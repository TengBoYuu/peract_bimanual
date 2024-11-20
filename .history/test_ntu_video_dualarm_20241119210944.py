'''
录制视频，用于debug数据，观察关键点
Usage:
conda activate manigaussian
cd /mnt/disk_1/guanxing/ManiGaussian
python test_ntu_video.py
'''
import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import natsort
import imageio
from helpers import demo_loading_utils
from tqdm import trange
import shutil

# from camera to world ??
camera_list = ["front_rgb"]
extrinsics = {
    camera_list[0]: [
        
        # [-0.98322948, 0.10974513, 0.14565641, 0.42687533],
        # [-0.05765898, 0.57064435, -0.8191706, 0.44493099],
        # [-0.17301799, -0.81383109, -0.55474656, 0.65768216],
        # [0.0, 0.0, 0.0, 1.0]
        # [0.9984647, -0.01066009, -0.05435628, 0.41119387],
        # [0.05308204, -0.09627122, 0.9939387, -1.28711702],
        # [-0.01582842, -0.99529805, -0.09555755, 0.33819046],
        # [0.0, 0.0, 0.0, 1.0]
        # [1.0, 0.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, 0.0],
        # [0.0, 0.0, 0.0, 1.0]
        [
            -2.384185791015625e-07,
            0.9063076972961426,
            0.4226186275482178,
            -0.17499709129333496
        ],
        [
            1.0000001192092896,
            1.1920928955078125e-07,
            -7.450580596923828e-08,
            0.0
        ],
        [
            -1.6391277313232422e-07,
            0.4226186275482178,
            -0.9063076972961426,
            2.4299938678741455
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0
        ]
    ],
    # camera_list[1]: [
    #     [-0.44906055,  0.20380977, -0.86994609,  1.12049788],
    #     [ 0.8932096,   0.1272763,  -0.43125091,  0.00939908],
    #     [ 0.02283037, -0.97070197, -0.23919961,  0.39543717],
    #     [ 0.,          0.,          0.,          1.        ],
    #     # [1.0, 0.0, 0.0, 0.0],
    #     # [0.0, 1.0, 0.0, 0.0],
    #     # [0.0, 0.0, 1.0, 0.0],
    #     # [0.0, 0.0, 0.0, 1.0]
    # ],
}
intrinsics = {
    camera_list[0]: [
        # [601.51025390625, 0., 327.903076171875],
        # [0., 601.7446899414062, 242.809326171875],
        # [0, 0, 1],
        # [603.2314453125, 0., 325.3480529785156],
        # [0., 603.2608032226562, 251.1649932861328],
        # [0, 0, 1],

    ],  # color
#     camera_list[1]: [
#         [601.51025390625, 0., 327.903076171875],
#         [0., 601.7446899414062, 242.809326171875],
#         [0, 0, 1],
#     ],
}
distortions = {
    'front_rgb': [0.1769687533378601, -0.5447412729263306, -0.0021821269765496254, 0.0002617577847559005, 0.478135347366333],
    # 'front_rgb': [0.15970464050769806, -0.5111035108566284, 0.00039717796607874334, 0.0006566409720107913, 0.4506976902484894],
}
extrinsics = {k:np.array(v) for k, v in extrinsics.items()}
intrinsics = {k:np.array(v) for k, v in intrinsics.items()}
distortions = {k:np.array(v) for k, v in distortions.items()}


# task_name = 'pick_and_place' pick in one
task_name = 'pick_in_two'
# task_name = 'lift'

camera_name = 'front_rgb'

extrinsic = extrinsics[camera_name]
extrinsic = np.linalg.inv(extrinsic)    # from world to camera
intrinsic = intrinsics[camera_name]

keypoint_method = 'heuristic_real'

# open_cabinet
# stopping_delta = 0.001
# stopping_buffer = 50    # default: 4
# warm_up = 60
# cool_down = 35
# keypoint_number_should_be_min = 4
# keypoint_number_should_be_max = 5

# all
stopping_delta = 0.00005
stopping_buffer = 35    # default: 4
warm_up = 42
cool_down = 0


N = 50
M = 50 # the number of episodes we want to keep
save_folder = f'data/debug/{task_name}_keypoint_video'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
os.makedirs(save_folder, exist_ok=True)

episode_id_that_should_skip = []

if task_name == 'press':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 3
    keypoint_number_should_be_max = 5
elif task_name == 'pnp':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 5
elif task_name == 'lift':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 2
    keypoint_number_should_be_max = 4
elif task_name == 'pick_in_two':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 4
    keypoint_number_should_be_max = 6
elif task_name == 'pick_in_one':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 4
    keypoint_number_should_be_max = 6
elif task_name == 'handover_item':
    episode_id_that_should_skip = []
    keypoint_number_should_be_min = 4
    keypoint_number_should_be_max = 15


for episode_id in trange(N):
    # episode_id = 9
    episode_path = f'data/real_dual_ur_train_data/{task_name}/all_variations/episodes/episode{episode_id}/'
    if not os.path.exists(episode_path):
        print(f"Skipping episode {episode_id}: Path does not exist")
        continue
    description_path = os.path.join(episode_path, 'variation_descriptions.pkl')
    description = pickle.load(open(description_path, 'rb'))
    description = description[0]
    
    # loop through all the observation to record the video
    rgb_path = os.path.join(episode_path, camera_name)
    demo_path = os.path.join(episode_path, 'low_dim_obs.pkl')

    img_files = os.listdir(rgb_path)
    img_files = natsort.natsorted(img_files)

    demo = pickle.load(open(demo_path, 'rb'))

    episode_keypoints = demo_loading_utils._keypoint_discovery_bimanual(
        demo, 
        # method=keypoint_method, 
        stopping_delta=stopping_delta,
        stopping_buffer=stopping_buffer,
        warm_up=warm_up,
        cool_down=cool_down,
    )
    # NOTE: the last is always a keypoint

    # print(f"episode_keypoints: {episode_keypoints}")
    # print(f"found {len(episode_keypoints)} keypoints")

    if episode_id in episode_id_that_should_skip:
        print(f"episode {episode_id} already in should skip list for other reasons")
        save_path = f'{save_folder}/episode{episode_id}_keynum_{keypoint_num}_failure.mp4'

    else:
        keypoint_num = len(episode_keypoints)
        if keypoint_num < keypoint_number_should_be_min or keypoint_num > keypoint_number_should_be_max:
            episode_id_that_should_skip.append(episode_id)
            print(f"episode {episode_id} should skip, found {keypoint_num} keypoints")
            save_path = f'{save_folder}/episode{episode_id}_keynum_{keypoint_num}_failure.mp4'
        else:
            save_path = f'{save_folder}/episode{episode_id}_keynum_{keypoint_num}_success.mp4'

    video_writer = imageio.get_writer(save_path, fps=20)

    for img_file in img_files:
        img_path = os.path.join(rgb_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # print velocity on the image
        step = int(img_file.split('.')[0])
        
        vel_left = demo[step].left.joint_velocities
        vel_right = demo[step].right.joint_velocities
        vel_left = np.max(np.abs(vel_left))
        vel_right = np.max(np.abs(vel_right))

        gripper_openness_right = demo[step].right.gripper_open
        gripper_openness_left = demo[step].left.gripper_open

        # plot the trajectory for the right gripper on the image
        pos_world_right = demo[step].right.gripper_pose[:3]
        # print(f"pos_world_right: {pos_world_right}")
        pos_world_right = np.array(pos_world_right).reshape(3, 1)
        pos_cam_right = extrinsic[:3, :3] @ pos_world_right + extrinsic[:3, 3:]
        pos_cam_right = intrinsic @ pos_cam_right
        pos_cam_right = pos_cam_right / pos_cam_right[2]
        pos_cam_right = pos_cam_right[:2].flatten()
        pos_cam_right = pos_cam_right.astype(np.int32)

        # plot the trajectory for the left gripper on the image
        pos_world_left = demo[step].left.gripper_pose[:3]
        # print(f"pos_world_left: {pos_world_left}")
        pos_world_left = np.array(pos_world_left).reshape(3, 1)
        pos_cam_left = extrinsic[:3, :3] @ pos_world_left + extrinsic[:3, 3:]
        pos_cam_left = intrinsic @ pos_cam_left
        pos_cam_left = pos_cam_left / pos_cam_left[2]
        pos_cam_left = pos_cam_left[:2].flatten()
        pos_cam_left = pos_cam_left.astype(np.int32)

        # Draw circles at the projected positions
        color_right = (255, 0, 0)  # Red for the right gripper
        color_left = (0, 0, 255)   # Blue for the left gripper
        cv2.circle(img, tuple(pos_cam_right), 5, color_right, -1)
        cv2.circle(img, tuple(pos_cam_left), 5, color_left, -1)
        # print(f"pos_cam: {pos_cam}")

        color = (255, 0, 0)  if step in episode_keypoints else (0, 255, 0)

        cv2.putText(img, f"step: {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"right_velocity: {vel_right:.4f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"left_velocity: {vel_left:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"right_gripper_openness: {gripper_openness_right}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img, f"left_gripper_openness: {gripper_openness_left}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # cv2.putText(img, f"instruction: {description}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


        # cv2.circle(img, tuple(pos_cam), 5, color, -1)

        # keep this frame for a while   
        if step in episode_keypoints:
            for _ in range(10):
                video_writer.append_data(img)

        video_writer.append_data(img)

    video_writer.close()

print(f"video saved to {save_path}")
episode_id_that_should_skip = sorted(episode_id_that_should_skip)
print(f"should skip: {episode_id_that_should_skip}")

should_keep_list = list(set(range(N)) - set(episode_id_that_should_skip))

if len(episode_id_that_should_skip) < N-M:
    episode_id_that_should_skip.extend(list(np.random.choice(should_keep_list, N-M-len(episode_id_that_should_skip), replace=False)))
    episode_id_that_should_skip = sorted(episode_id_that_should_skip)

print(f"selected should skip: {episode_id_that_should_skip}, len: {len(episode_id_that_should_skip)}")

should_keep_list = list(set(range(N)) - set(episode_id_that_should_skip))
print(f"selected should keep: {should_keep_list}, len: {len(should_keep_list)}")

