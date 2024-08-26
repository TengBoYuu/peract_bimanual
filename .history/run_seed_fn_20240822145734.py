import os
import pickle
import gc
from typing import List
import filecmp
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator
from yarr.replay_buffer.task_uniform_replay_buffer import TaskUniformReplayBuffer

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
import random
from agents import agent_factory
from agents import replay_utils
from typing import Tuple, Optional
import peract_config
from functools import partial
import copy

from helpers.dataset_engine import RLBenchDataset
from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
class ReplayBuffer:
    def __init__(self):
        # 初始化 replay buffer，用于保存所有数据
        self.buffer = []

    def add_dataset(self, dataset):
        # 将整个数据集添加到 buffer 中
        self.buffer.extend(dataset)

    def get_all_data(self):
        # 返回所有缓冲的数据
        return self.buffer
    
def get_datasets(cfg):
    """Initialize datasets."""
    # Load instruction, based on which we load tasks/variations
    print("--------------in",cfg.rlbench.instructions)
    variations: Tuple[int, ...] = tuple(range(200))
    instruction = load_instructions(
        cfg.rlbench.instructions,
        tasks=cfg.rlbench.tasks,
        variations = variations
    )
    if instruction is None:
        raise NotImplementedError()
    else:
        taskvar = [
            (task, var)
            for task, var_instr in instruction.items()
            for var in var_instr.keys()
        ]
    
    # Initialize datasets with arguments
    train_dataset = RLBenchDataset(
        root="/mnt/disk_1/tengbo/bimanual_data/package/train",
        instructions=instruction,
        taskvar=taskvar,
        max_episode_length=5,
        cache_size=100,
        max_episodes_per_task=100,
        num_iters=200_000,
        cameras=("over_shoulder_left", "over_shoulder_right", "overhead", "wrist_right", "wrist_left", "front"),
        training=True,
        image_rescale=tuple(
            float(x) for x in str("0.75,1.25").split(",")
        ),
        return_low_lvl_trajectory=True,
        dense_interpolation=bool(1),
        interpolation_length=100
    )
    return train_dataset
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_loaders(cfg,collate_fn=default_collate):
    """Initialize data loaders."""
    # Datasets
    train_dataset = get_datasets(cfg)
    # for i, data in enumerate(train_dataset):
    #     print(f"Sample {i} shapes:")
    #     for key, value in data.items():
    #         print(f"{key}: {value.shape if isinstance(value, torch.Tensor) else type(value)}")
    # Samplers and loaders
    g = torch.Generator()
    g.manual_seed(0)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.replay.batch_size,
        shuffle=False,
        num_workers=1,
        worker_init_fn=seed_worker,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        generator=g
    )

    return train_loader

def pad_and_crop_tensors(tensors, target_size=128, pad_value=0, pad_first_dim=True):
    # 获取每个维度的最大大小，保持高宽为 target_size
    max_shape = [max(sizes) for sizes in zip(*[tensor.shape for tensor in tensors])]
    
    # 只保留宽度和高度维度为 target_size，其余保持原样
    target_shape = [min(dim_size, target_size) if i in [3, 4] else dim_size
                    for i, dim_size in enumerate(max_shape)]
    
    padded_tensors = []
    for tensor in tensors:
        # 填充或裁剪操作
        padding = []
        for dim in range(len(tensor.shape)-1, -1, -1):
            if tensor.shape[dim] < target_shape[dim]:
                # 如果维度小于目标大小，填充
                pad_before = 0
                pad_after = target_shape[dim] - tensor.shape[dim]
            elif tensor.shape[dim] > target_shape[dim]:
                # 如果维度大于目标大小，裁剪
                tensor = tensor.narrow(dim, 0, target_shape[dim])
                pad_before = 0
                pad_after = 0
            else:
                pad_before = 0
                pad_after = 0
            padding.insert(0, pad_after)
            padding.insert(0, pad_before)
        
        # 如果需要在第一个维度进行填充（通常是 batch 维度）
        if pad_first_dim:
            first_dim_padding = max_shape[0] - tensor.shape[0]
            padding = [0, 0] * (len(tensor.shape) - 1) + [0, first_dim_padding]
        
        padded_tensor = torch.nn.functional.pad(tensor, pad=padding, value=pad_value)
        padded_tensors.append(padded_tensor)
    
    return torch.stack(padded_tensors)

def custom_collate_fn(batch):
    batch_dict = {}
    
    for key in batch[0].keys():
        first_item = batch[0][key]
        
        if isinstance(first_item, torch.Tensor):
            # 对于 Tensor 类型的数据，进行正常的批处理操作
            if key in ['right_rgbs', 'left_rgbs', 'right_pcds', 'left_pcds',
                       'right_action', 'left_action', 'right_instr', 'left_instr',
                       'right_curr_gripper', 'left_curr_gripper',
                       'right_curr_gripper_history', 'left_curr_gripper_history',
                       'right_trajectory', 'left_trajectory', 'right_trajectory_mask', 'left_trajectory_mask']:
                # 对每个需要处理的键进行填充处理
                batch_dict[key] = pad_and_crop_tensors([item[key] for item in batch],target_size=128,pad_first_dim=True)
            else:
                # 直接堆叠 Tensors
                batch_dict[key] = torch.stack([item[key] for item in batch])
        
        elif isinstance(first_item, list):
            # 忽略 List 类型的数据
            continue
        
        else:
            # 对其他类型的数据保持原样
            batch_dict[key] = [item[key] for item in batch]
    
    return batch_dict

def run_seed(
    rank,
    cfg: DictConfig,
    obs_config: ObservationConfig,
    seed,
    world_size,
) -> None:
    

    peract_config.config_logging()
    
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    tasks = cfg.rlbench.tasks
    cams = cfg.rlbench.cameras

    # task_folder = "debug" if len(tasks) > 1 else tasks[0] 
    task_folder = cfg.replay.task_folder if len(tasks) > 1 else tasks[0] 
    # task_folder = cfg.rlbench.task_name
    replay_path = os.path.join(
        cfg.replay.path, task_folder, cfg.method.name, "seed%d" % seed
    )
    # to do create agent
    agent = agent_factory.create_agent(cfg)

    if not agent:
        print("Unable to create agent")
        return

    if cfg.method.name == "ARM":
        raise NotImplementedError("ARM is not supported yet")
    elif cfg.method.name == "BC_LANG":
        from agents.baselines import bc_lang

        assert cfg.ddp.num_devices == 1, "BC_LANG only supports single GPU training"
        replay_buffer = bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )


    elif cfg.method.name == "VIT_BC_LANG":
        from agents.baselines import vit_bc_lang

        assert cfg.ddp.num_devices == 1, "VIT_BC_LANG only supports single GPU training"
        replay_buffer = vit_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
        )

        vit_bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name.startswith("ACT_BC_LANG"):
        from agents import act_bc_lang

        assert cfg.ddp.num_devices == 1, "ACT_BC_LANG only supports single GPU training"
        replay_buffer = act_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.rlbench.camera_resolution,
            replay_size=3e5,
            prev_action_horizon=cfg.method.prev_action_horizon,
            next_action_horizon=cfg.method.next_action_horizon
        )

        act_bc_lang.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
        )

    elif cfg.method.name == "C2FARM_LINGUNET_BC":
        from agents import c2farm_lingunet_bc

        replay_buffer = c2farm_lingunet_bc.launch_utils.create_replay(
            cfg.replay.batch_size,
            cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,
            cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution,
        )

        c2farm_lingunet_bc.launch_utils.fill_multi_task_replay(
            cfg,
            obs_config,
            rank,
            replay_buffer,
            tasks,
            cfg.rlbench.demos,
            cfg.method.demo_augmentation,
            cfg.method.demo_augmentation_every_n,
            cams,
            cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes,
            cfg.method.bounds_offset,
            cfg.method.rotation_resolution,
            cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method,
        )


    elif cfg.method.name.startswith("BIMANUAL_PERACT") or cfg.method.name.startswith("RVT") or cfg.method.name.startswith("PERACT_BC"):
        print(replay_path)
        if os.path.exists(replay_path):
            print("Replay files found. Loading...")
            # 初始化 Replay Buffer
            # replay_buffer = TaskUniformReplayBuffer()
            replay_buffer = replay_utils.create_replay(cfg, replay_path)
            # 加载所有的 Replay 文件
            replay_files = [os.path.join(replay_path, f) for f in os.listdir(replay_path) if f.endswith('.replay')]
            for replay_file in replay_files:
                print(replay_file)
                with open(replay_file, 'rb') as f:
                    replay_data = pickle.load(f)
                replay_buffer.load_add(replay_data)  # 调用 _add 方法将数据加载到缓冲区中
        else:
            print("No replay files found. Creating replay...")
            replay_buffer = replay_utils.create_replay(cfg, replay_path)
            replay_utils.fill_multi_task_replay(
                cfg,
                obs_config,
                rank,
                replay_buffer,
                tasks
            )
    # elif cfg.method.name.startswith("DIFFUSER_ACTOR"):
    #     print("----------------3dda-------------------")
    #     # 这里是不是不需要传buffer
    #     replay_buffer = ReplayBuffer()
    #     # train_loader = get_loaders(cfg=cfg,collate_fn=custom_collate_fn)
    #     # replay_buffer.add_dataset(train_loader)
    #     # print(replay_buffer)

    elif cfg.method.name == "PERACT_RL":
        raise NotImplementedError("PERACT_RL is not supported yet")
    
    else:
        raise ValueError("Method %s does not exists." % cfg.method.name)

    wrapped_replay = PyTorchReplayBuffer(
        replay_buffer, num_workers=cfg.framework.num_workers
    )
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(cwd, "seed%d" % seed, "weights")
    logdir = os.path.join(cwd, "seed%d" % seed)

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=rank,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size,
        cfg=cfg
    )

    train_runner._on_thread_start = partial(peract_config.config_logging, cfg.framework.logging_level)
    
    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
