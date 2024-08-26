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

from agents import agent_factory
from agents import replay_utils

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
    
def get_datasets(self,cfg):
    """Initialize datasets."""
    # Load instruction, based on which we load tasks/variations
    instruction = load_instructions(
        cfg.rlbench.instructions,
        tasks=cfg.rlbench.tasks,
        variations=cfg.rlbench.variations
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
            float(x) for x in (0,75,1.25).split(",")
        ),
        return_low_lvl_trajectory=True,
        dense_interpolation=bool(0),
        interpolation_length=100
    )
    return train_dataset

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
    elif cfg.method.name.startswith("DIFFUSER_ACTOR"):
        print("----------------3dda-------------------")
        replay_buffer = ReplayBuffer()
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
