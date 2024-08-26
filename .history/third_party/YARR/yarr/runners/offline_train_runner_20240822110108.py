import copy
import logging
import os
import shutil
import time
from typing import List
from typing import Union

import psutil
import torch
import pandas as pd
from yarr.agents.agent import Agent
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import \
    PyTorchReplayBuffer
from yarr.utils.log_writer import LogWriter
from yarr.utils.stat_accumulator import StatAccumulator
from tqdm import tqdm
from omegaconf import DictConfig
import wandb
from termcolor import cprint

from typing import Tuple
from helpers.engine import BaseTrainTester
from helpers.dataset_engine import RLBenchDataset
from utils.common_utils import (
    load_instructions, count_parameters, get_gripper_loc_bounds
)
from torch.utils.data import DataLoader, default_collate
class OfflineTrainRunner(BaseTrainTester):

    def __init__(self,
                 agent: Agent,
                 wrapped_replay_buffer: PyTorchReplayBuffer,
                 train_device: torch.device,
                 stat_accumulator: Union[StatAccumulator, None] = None,
                 iterations: int = int(6e6),
                 logdir: str = '/tmp/yarr/logs',
                 logging_level: int = logging.INFO,
                 log_freq: int = 10,
                 weightsdir: str = '/tmp/yarr/weights',
                 num_weights_to_keep: int = 60,
                 save_freq: int = 100,
                 tensorboard_logging: bool = True,
                 csv_logging: bool = False,
                 load_existing_weights: bool = True,
                 rank: int = None,
                 world_size: int = None,
                 cfg: DictConfig = None,
                 ):
        super().__init__(cfg)
        self._agent = agent
        self._wrapped_buffer = wrapped_replay_buffer
        self._stat_accumulator = stat_accumulator
        self._iterations = iterations
        self._logdir = logdir
        self._logging_level = logging_level
        self._log_freq = log_freq
        self._weightsdir = weightsdir
        self._num_weights_to_keep = num_weights_to_keep
        self._save_freq = save_freq

        self._wrapped_buffer = wrapped_replay_buffer
        self._train_device = train_device
        self._tensorboard_logging = tensorboard_logging
        self._csv_logging = csv_logging
        self._load_existing_weights = load_existing_weights
        self._rank = rank
        self._world_size = world_size

        self.use_wandb = cfg.framework.use_wandb
        self.use_pretrained = cfg.framework.use_pretrained

        self.instructions = cfg.rlbench.instructions
        self.method = cfg.method.name
        self.tasks = cfg.rlbench.tasks

        if self.use_wandb and rank == 0:
            wandb_name = cfg.framework.wandb_name
            wandb.init(project=cfg.framework.wandb_project, group=str(cfg.framework.wandb_group), name=wandb_name, config=cfg)
            cprint(f'[wandb] init in {cfg.framework.wandb_project}/{cfg.framework.wandb_group}/{wandb_name}', 'cyan')
            
        self._writer = None
        if logdir is None:
            logging.info("'logdir' was None. No logging will take place.")
        else:
            self._writer = LogWriter(
                self._logdir, tensorboard_logging, csv_logging)

        if weightsdir is None:
            logging.info(
                "'weightsdir' was None. No weight saving will take place.")
        else:
            os.makedirs(self._weightsdir, exist_ok=True)

    def _save_model(self, i):
        d = os.path.join(self._weightsdir, str(i))
        os.makedirs(d, exist_ok=True)
        self._agent.save_weights(d)

        # remove oldest save
        prev_dir = os.path.join(self._weightsdir, str(
            i - self._save_freq * self._num_weights_to_keep))
        if os.path.exists(prev_dir):
            shutil.rmtree(prev_dir)

    def _step(self, i, sampled_batch):
        update_dict = self._agent.update(i, sampled_batch)
        total_losses = update_dict['total_losses']
        return total_losses

    def _get_resume_eval_epoch(self):
        starting_epoch = 0
        eval_csv_file = self._weightsdir.replace('weights', 'eval_data.csv') # TODO(mohit): check if it's supposed be 'env_data.csv'
        if os.path.exists(eval_csv_file):
             eval_dict = pd.read_csv(eval_csv_file).to_dict()
             epochs = list(eval_dict['step'].values())
             return epochs[-1] if len(epochs) > 0 else starting_epoch
        else:
            return starting_epoch

    def get_datasets(self):
        """Initialize datasets."""
        # Load instruction, based on which we load tasks/variations
        # print("--------------in",instructions)
        variations: Tuple[int, ...] = tuple(range(200))
        instruction = load_instructions(
            self.instructions,
            self.tasks,
            variations
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
    def start(self):

        if hasattr(self, "_on_thread_start"):
            self._on_thread_start()
        else:
            logging.getLogger().setLevel(self._logging_level)
         

        self._agent = copy.deepcopy(self._agent)
        self._agent.build(training=True, device=self._train_device)

        if self.use_pretrained:
            ##########################################Load Weights########################################################
            self._agent.load_weights("/mnt/disk_1/tengbo/peract_bimanual/ckpts/multi/PERACT_BC/seed0/weights/600000/")
            if self._rank == 0:
                print("-----------------Loading Success Pre-trained Peract Model----------------------")
                logging.info(f"Loading Success Pre-trained Peract Model")
            start_iter = 0
            ##########################################Load Weights########################################################
        else:
            if self._weightsdir is not None:
                existing_weights = sorted([int(f) for f in os.listdir(self._weightsdir)])
                if (not self._load_existing_weights) or len(existing_weights) == 0:
                    # self._save_model(0)
                    start_iter = 0
                else:
                    resume_iteration = existing_weights[-1]
                    self._agent.load_weights(os.path.join(self._weightsdir, str(resume_iteration)))
                    start_iter = resume_iteration
                    if self._rank == 0:
                        logging.info(f"Resuming training from iteration {resume_iteration} ...")

        if self.method.startswith("DIFFUSER_ACTOR"):
            # TO DO: 
            # 这里用base的iter导入data_iter
            print("------------------3dda----------------------")
            # for循环iteration
            # 修改step函数？？？ 或者这里不改，在agent里面去改，agent传入的step是iteration的轮次，和diffusion无关
            # agent大类应该不用动，需要改的是q_attention里面的update等，sample中的内容都会有变动
            # 改agent_fn_by_name Bimanualagnet
            # dataset = self.get_datasets()

            # data_loader = self.get_loaders(collate_fn=self.custom_collate_fn)
            # process = psutil.Process(os.getpid())
            # num_cpu = psutil.cpu_count()
            # data_iter = iter(data_loader)
            for i in tqdm(range(start_iter, self._iterations), mininterval=10):
                log_iteration = i % self._log_freq == 0 and i > 0
                if log_iteration:
                    process.cpu_percent(interval=None)
                t = time.time()
                sampled_batch = next(data_iter)
                print(sampled_batch)
                sample_time = time.time() - t
        else:
            dataset = self._wrapped_buffer.dataset()
            data_iter = iter(dataset)

            process = psutil.Process(os.getpid())
            num_cpu = psutil.cpu_count()

            for i in tqdm(range(start_iter, self._iterations), mininterval=10):
                log_iteration = i % self._log_freq == 0 and i > 0

                if log_iteration:
                    process.cpu_percent(interval=None)

                t = time.time()
                sampled_batch = next(data_iter)
                sample_time = time.time() - t

                batch = {k: v.to(self._train_device) for k, v in sampled_batch.items() if type(v) == torch.Tensor}
                t = time.time()
                loss = self._step(i, batch)
                step_time = time.time() - t

                if self._rank == 0:
                    if log_iteration and self._writer is not None:
                        # agent_summaries = self._agent.update_summaries()
                        # self._writer.add_summaries(i, agent_summaries)

                        # self._writer.add_scalar(
                        #     i, 'monitoring/memory_gb',
                        #     process.memory_info().rss * 1e-9)
                        # self._writer.add_scalar(
                        #     i, 'monitoring/cpu_percent',
                        #     process.cpu_percent(interval=None) / num_cpu)

                        logging.info(f"Train Step {i:06d} | Loss: {loss:0.5f} | Sample time: {sample_time:0.6f} | Step time: {step_time:0.4f}.")

                    # self._writer.end_iteration()

                    if i % self._save_freq == 0 and self._weightsdir is not None:
                        self._save_model(i)
                torch.cuda.empty_cache()
            
        if self._rank == 0 and self._writer is not None:
            self._writer.close()
            logging.info('Stopping envs ...')

            self._wrapped_buffer.replay_buffer.shutdown()
