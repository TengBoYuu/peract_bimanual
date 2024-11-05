'''
Control the ur robot arm with the learned policy (in ntu lab)

Usage:
conda activate manigaussian
CUDA_VISIBLE_DEVICES=6 python eval_real_agent_ntu_0919.py
'''
import os
import time
from PIL import Image
import numpy as np
import pickle
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf

from real_utils import image_to_float_array, float_array_to_rgb_image, pointcloud_from_depth_and_camera_params
DEPTH_SCALE = 2**24 - 1

from agents import any_bimanual

from scipy.spatial.transform import Rotation as R
from helpers import demo_loading_utils

import visdom
import einops
from transformers import CLIPTokenizer, CLIPTextModel


class AnyBimanualAgentInterface:
    def __init__(self, cfg, instruction):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.step = 0
    
        current_dir = os.path.dirname(os.path.realpath(__file__))
        print('current_dir: ', current_dir)

        self.extrinsics = {
            'front_rgb': [
                # [-0.98322948, 0.10974513, 0.14565641, 0.42687533],
                # [-0.05765898, 0.57064435, -0.8191706, 0.44493099],
                # [-0.17301799, -0.81383109, -0.55474656, 0.65768216],
                # [0.0, 0.0, 0.0, 1.0]
                [0.9984647, -0.01066009, -0.05435628, 0.41119387],
                [0.05308204, -0.09627122, 0.9939387, -1.28711702],
                [-0.01582842, -0.99529805, -0.09555755, 0.33819046],
                [0.0, 0.0, 0.0, 1.0]
            ],
        }
        self.extrinsics = {k:np.array(v) for k, v in self.extrinsics.items()}
        self.intrinsics = {
            'front_rgb': [
                # [601.51025390625, 0., 327.903076171875],
                # [0., 601.7446899414062, 242.809326171875],
                # [0, 0, 1],
                [603.2314453125, 0., 325.3480529785156],
                [0., 603.2608032226562, 251.1649932861328],
                [0, 0, 1],
            ],  # color
        }
        self.intrinsics = {k:np.array(v) for k, v in self.intrinsics.items()}

        self.z_near = 0.0
        self.z_far = 1.2

        self.lang_goal = instruction
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def _resize_if_needed(self, image, size):
        if image.size[0] != size[0] or image.size[1] != size[1]:
            image = image.resize(size)
        return image

    def _load_agent(self):
        cfg_path = os.path.join(self.cfg['agent']['seed_path'], 'config.yaml')
        cfg = OmegaConf.load(cfg_path)
        print('config: ', cfg)
        self.agent_cfg = cfg

        # load agent
        # cfg.method.use_neural_rendering = False # when doing eval, we suppress neural rendering
        self.agent = any_bimanual.launch_utils.create_agent(cfg)

        self.agent.build(training=False, device=self.device)

        # load pre-trained weights
        weights_path = os.path.join(self.cfg['agent']['seed_path'], 'weights',
                                    str(self.cfg['agent']['weight']))
        self.agent.load_weights(weights_path)

        print("Loaded: " + weights_path)

    def get_obs(self, img, depth, right_gripper_info, left_gripper_info, step):
        '''
        img: PIL image
        depth: PIL image
        gripper_info: float
        step: int
        '''
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        obs = {}
        # pre-calibrated camera info
        front_camera_intrinsics = self.intrinsics['front_rgb']
        front_camera_extrinsics = self.extrinsics['front_rgb']
        obs['front_camera_intrinsics'] = torch.tensor([front_camera_intrinsics], device=self.device).unsqueeze(0)
        obs['front_camera_extrinsics'] = torch.tensor([front_camera_extrinsics], device=self.device).unsqueeze(0)

        # real-time point cloud computation
        img = np.array(img)
        front_rgb = torch.tensor([img], device=self.device)   # [1, 480, 640, 3]
        # print(f"front_rgb.shape: {front_rgb.shape}")
        # front_rgb = front_rgb.permute(0, 3, 1, 2).permute(0, 1, 3, 2).unsqueeze(0)  # [1, 1, 3, 640, 480], min: 0, max: 255
        front_rgb = front_rgb.permute(0, 3, 1, 2).unsqueeze(0)
        # print(f"front_rgb.shape: {front_rgb.shape}")
        obs['front_rgb'] = front_rgb #torch.flip(front_rgb, dims=[2])

        front_depth = image_to_float_array(
            depth,
            DEPTH_SCALE)
        # print(f"front_rgb.shape: {front_depth.shape}")
        # print(f"front_depth.min(): {front_depth.min()}, front_depth.max(): {front_depth.max()}")
        # if self.agent_cfg.method.transform_augmentation.apply_se3:
        #     front_depth[front_depth == 1] = 2.0
        obs['front_depth'] = torch.tensor([front_depth], device=self.device)    # no channel dimension
        
        near = self.z_near
        far = self.z_far
        front_depth_m = near + front_depth * (far - near)

        front_point_cloud = pointcloud_from_depth_and_camera_params(
                                                            # front_depth,
                                                            front_depth_m,
                                                            front_camera_extrinsics,
                                                            front_camera_intrinsics)
        front_point_cloud = torch.tensor([front_point_cloud], device=self.device)
        # front_point_cloud = front_point_cloud.permute(0, 3, 1, 2).permute(0, 1, 3, 2).unsqueeze(0)   #[1, 1, 3, 640, 480]
        front_point_cloud = front_point_cloud.permute(0, 3, 1, 2).unsqueeze(0)   #[1, 3, 640, 480]
        # print(f"front_point_cloud.shape: {front_point_cloud.shape}")
        obs['front_point_cloud'] = front_point_cloud

        # collision
        obs['ignore_collisions'] = torch.tensor([[[1.0]]], device=self.device)

        # -------------------------需要确定 denghaoyuan-------------------------
        inputs = self.tokenizer(
            self.lang_goal,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )

        # language
        obs['lang_goal'] = self.lang_goal
        obs['lang_goal_tokens'] = inputs.input_ids.to(self.device)
        # print(f"lang_goal: {self.lang_goal}, lang_goal_tokens.shape: {last_hidden_state.shape}")

        # --------------------------------------------------------------------

        # robotic proprioception
        right_finger_positions = right_gripper_info
        left_finger_positions = left_gripper_info
        # gripper_open = (1.0 if (gripper_open_amount > 0.0385 + 0.0385) else 0.0)
        threshold = 90
        right_gripper_open = (1.0 if right_finger_positions[0] < threshold else 0.0)
        left_gripper_open = (1.0 if left_finger_positions[0] < threshold else 0.0)
        time = (1. - (step / float(self.cfg['agent']['episode_length'] - 1))) * 2. - 1.
        
        right_low_dim_state = torch.tensor([[[right_gripper_open, right_finger_positions[0],
                                        right_finger_positions[1],
                                        time]]])
        
        left_low_dim_state = torch.tensor([[[left_gripper_open,
                                        left_finger_positions[0],
                                        left_finger_positions[1],
                                        time]]])
        
        obs['right_low_dim_state'] = right_low_dim_state
        obs['left_low_dim_state'] = left_low_dim_state
        return obs
    
    def adjust_gripper_z(self, action, displacement):
        position = np.array(action[:3])  # x, y, z
        # print("!!!!!!!!!!!!!!!!!!!original position",position)
        quaternion = np.array(action[3:7])  # rx, ry, rz, w

        # Create a rotation object from the quaternion
        rotation = R.from_quat(quaternion)

        # Get the rotation matrix from the rotation object
        rotation_matrix = rotation.as_matrix()

        # Create the displacement vector along the gripper's z-axis
        z_displacement = rotation_matrix[:, 2] * displacement  # Third column is the z-axis

        # Adjust the position by the displacement along the gripper's z-axis
        adjusted_position = position + z_displacement

        # Return the new action with the adjusted position
        adjusted_action = np.concatenate([adjusted_position, quaternion, action[7:]])
        return adjusted_action[:8]

    def act(self, obs, step):
        self.act_result = self.agent.act(step, obs, deterministic=True)

        action = self.act_result.action
        # [x, y, z, rx, ry, rz, w, open, collision]

        # quaternion = np.array([action[3], action[4], action[5], action[6]]) # (x,y,z,w) format
        # euler = Rotation.from_quat(quaternion).as_euler('xyz', degrees=True)
        # rotvec = Rotation.from_quat(quaternion).as_rotvec()

        # act_res = [action[0], action[1], action[2], rotvec[0], rotvec[1], rotvec[2], action[7], action[8]] # for ur3 (rotvec)
        # adjusted_action = self.adjust_gripper_z(action, -0.16)
        right_act_res = action[:9]
        left_act_res = action[9:]
        right_act = self.adjust_gripper_z(right_act_res, -0.16)
        left_act = self.adjust_gripper_z(left_act_res, -0.16)
        # [x, y, z, theta_x, theta_y, theta_z, open, collision]
        # return right_act_res, left_act_res
        return right_act, left_act


task_to_instruction = {
    'lift': 'lift the box.',    
    'pnp': 'put the green cube in the green box and put the red cube in the orange box.',
    'press': 'press dish soap into the bowl.',
    'handover': 'handover the bowl to the other hand.',
    'pick_in_one': 'place the two cubes in the bowl.',
    'pick_in_two': 'place the two cubes in boxes of the corresponding color.'
}


def main():
    initialize(config_path="conf/real")
    config_name = "anybimanual_agent"
    cfg = compose(config_name=config_name)

    task_name = 'lift'

    instruction = task_to_instruction[task_name]
    print(f"task_name: {task_name}, instruction: {instruction}")

    # ------------------------------
    # initialize the agent interface
    agent = AnyBimanualAgentInterface(cfg, instruction=instruction)
    agent._load_agent()
    episode_list = [7]

    right_total_loss = []
    left_total_loss = []
    for episode_idx in episode_list:
        # episode_idx = 17
        # keypoint_idx = 0    

        episode_path = f'/home/pine/peract_bimanual_real/data/real_dual_ur_train_data/{task_name}/all_variations/episodes/episode20'
        # episode_path = f'/home/pine/peract_bimanual_real/data/real_dual_ur_train_data/handover_1014/all_variations/episodes/episode20'
        print(f"episode_path: {episode_path}")

        LOW_DIM_PICKLE = 'low_dim_obs.pkl'
        with open(os.path.join(episode_path, LOW_DIM_PICKLE), 'rb') as f:
            demo = pickle.load(f)

        episode_keypoints = demo_loading_utils._keypoint_discovery_dualarm(
            demo, 
            warm_up=42,
            cool_down=0,
            stopping_delta=0.00005,
            stopping_buffer=35,  
        )
        print(f"episode_keypoints: {episode_keypoints}")

        test_num = len(episode_keypoints)
        # test_num = 1
        step = 0

        for keypoint_idx in range(test_num):

            print('-----------------------------------')

            prev_step_idx = episode_keypoints[keypoint_idx - 1] if keypoint_idx != 0 else 0
            next_step_idx = episode_keypoints[keypoint_idx]
            print(f"prev_step_idx: {prev_step_idx}, next_step_idx: {next_step_idx}, step: {step}")

            image_path = os.path.join(episode_path, 'front_rgb', f'{prev_step_idx}.png')
            img = Image.open(image_path)
            # print(f"img.min(): {np.array(img).min()}, img.max(): {np.array(img).max()}")
            
            '''
            If the depth is preprocessed
            '''
            depth_path = image_path.replace('front_rgb', 'front_depth')
            depth = Image.open(depth_path)
            print(depth.size)

            '''
            If the depth (mm) is obtained on-the-fly, then use the following code:
            depth = depth / 1000.0  # convert to (m)
            depth = float_array_to_rgb_image(depth, scale_factor=DEPTH_SCALE)
            '''

            right_gripper_info = demo[prev_step_idx].right.gripper_joint_positions
            left_gripper_info = demo[prev_step_idx].left.gripper_joint_positions
            print(f"input riht gripper_info: {right_gripper_info}, left gripper_info: {left_gripper_info}")

            # step = prev_step_idx # NOTE: this is the counter of how much the agent has been called (not the step in the episode), so only '0' is valid now

            # prepare observation
            obs = agent.get_obs(img, depth, right_gripper_info, left_gripper_info, step)

            # inference
            right_act_res, left_act_res = agent.act(obs, step)

            step += 1

            print(f"right arm action: {right_act_res},left arm action: {left_act_res}")


            # get ground-truth
            right_gt_pose = demo[next_step_idx].right.gripper_pose
            left_gt_pose = demo[next_step_idx].left.gripper_pose
            right_gt_gripper = demo[next_step_idx].right.gripper_open
            left_gt_gripper = demo[next_step_idx].left.gripper_open

            # convert to euler
            # gt_pose[3:] = Rotation.from_quat(gt_pose[3:]).as_euler('xyz', degrees=True)
            # gt_pose[3:] = Rotation.from_quat(gt_pose[3:]).as_rotvec()

            right_gt_action = right_gt_pose + [right_gt_gripper, 1.0]
            left_gt_action = left_gt_pose + [left_gt_gripper, 1.0]
            print(f"ground-truth right action: {right_gt_action}, left action: {left_gt_action}")
            # print(f"ground-truth action: {gt_action}")

            # right_total_loss += np.linalg.norm(np.array(right_act_res) - np.array(right_gt_action))
            # left_total_loss += np.linalg.norm(np.array(left_act_res) - np.array(left_gt_action))
            # print(f"right_total_loss: {right_total_loss}, left_total_loss: {left_total_loss}")
            right_total_loss.append(np.abs(np.array(right_act_res) - np.array(right_gt_action)))  # [8]
            left_total_loss.append(np.abs(np.array(left_act_res) - np.array(left_gt_action)))

    right_total_loss_mean = np.array(right_total_loss).mean(axis=0)
    left_total_loss_mean = np.array(left_total_loss).mean(axis=0)
    print(f"average loss: {right_total_loss_mean}, {left_total_loss_mean}") 

    # benchmarking inference time
    BENCHMARK = False
    if BENCHMARK:
        print("Benchmarking...")
        for step in range(3):  # warm-up
            obs = agent.get_obs(img, depth, gripper_info, step)
            act_res = agent.act(obs, step)
        start_time = time.time()
        N = 20
        for step in range(N):
            obs = agent.get_obs(img, depth, gripper_info, step)
            act_res = agent.act(obs, step)
        end_time = time.time()
        print(f"average time: {(end_time - start_time) / N}")


if __name__ == "__main__":
   main() 
