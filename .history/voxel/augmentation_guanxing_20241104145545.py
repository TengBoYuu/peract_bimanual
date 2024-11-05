import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
import time
import os




if __name__ == "__main__":
    from helpers.utils import visualise_voxel, stack_on_channel
    import visdom
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R
    def convert_to_numpy(pcd_tensor):
        pcd_np = pcd_tensor.squeeze().cpu().numpy() # [B, C, H, W]
        pcd_np = pcd_np.transpose(1, 2, 0) # [H, W, C]
        pcd_np = pcd_np.reshape(-1, 3) # [H*W, C]
        return pcd_np

    def plot_point_cloud(pcd, left_pose, right_pose, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points and end-effector poses
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=1, alpha=0.01)
        ax.scatter(left_pose[:, 0], left_pose[:, 1], left_pose[:, 2], c='r', marker='o', s=40)
        ax.scatter(right_pose[:, 0], right_pose[:, 1], right_pose[:, 2], c='g', marker='o', s=40)

        # Draw left and right gripper axes
        for pose in left_pose:
            trans = pose[:3]
            rot = pose[3:]  # wxyz
            rotation = R.from_quat(rot).as_matrix()
            for i, color in zip(range(3), ['r', 'g', 'b']):
                start = trans
                end = trans + rotation[:, i] * 0.1
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

        for pose in right_pose:
            trans = pose[:3]
            rot = pose[3:]
            rotation = R.from_quat(rot).as_matrix()
            for i, color in zip(range(3), ['r', 'g', 'b']):
                start = trans
                end = trans + rotation[:, i] * 0.1
                ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")

    # Sample inputs
    batch_size = 1
    # pcd = [torch.rand(batch_size, 3, 128, 128) for _ in range(1)]

    # numpoints = 128 * 128
    # pcd_flat = torch.zeros(batch_size, 3, numpoints)
    # pcd_flat[0, 0, :] = torch.linspace(-1, 1, numpoints)
    # pcd_flat[0, 1, :] = torch.linspace(-1, 1, numpoints)
    # pcd_flat[0, 2, :] = 0.0
    # pcd_flat = pcd_flat.reshape(batch_size, 3, 128, 128)
    # pcd = [torch.rand(batch_size, 3, 1, 1) for _ in range(1)]
    # pcd = [pcd_flat]
    # right_action_gripper_pose = torch.tensor([[0.8, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    # right_action_trans = right_action_gripper_pose[:, :3]
    # right_action_rot_grip = right_action_gripper_pose[:, 3:]
    # left_action_gripper_pose = torch.tensor([[0.2, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]])
    # left_action_trans = left_action_gripper_pose[:, :3]
    # left_action_rot_grip = left_action_gripper_pose[:, 3:]

    root_path = "/mnt/disk_1/tengbo/peract_bimanual-real/voxel/debug"
    pcd = np.load(os.path.join(root_path, "pcd.npy"))
    pcd = [torch.tensor(pcd)]
    right_action_gripper_pose = np.load(os.path.join(root_path, "right_action_gripper_pose.npy"))
    right_action_trans = np.load(os.path.join(root_path, "right_action_trans.npy"))
    right_action_rot_grip = np.load(os.path.join(root_path, "right_action_rot_grip.npy"))
    left_action_gripper_pose = np.load(os.path.join(root_path, "left_action_gripper_pose.npy"))
    left_action_trans = np.load(os.path.join(root_path, "left_action_trans.npy"))
    left_action_rot_grip = np.load(os.path.join(root_path, "left_action_rot_grip.npy"))

    right_action_gripper_pose = torch.tensor(right_action_gripper_pose)
    right_action_trans = torch.tensor(right_action_trans)
    right_action_rot_grip = torch.tensor(right_action_rot_grip)
    left_action_gripper_pose = torch.tensor(left_action_gripper_pose)
    left_action_trans = torch.tensor(left_action_trans)
    left_action_rot_grip = torch.tensor(left_action_rot_grip)

    # bounds = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    bounds = torch.tensor([[0.0, -1.0, -0.2, 1.0, 0.0, 0.8]])
    layer = 0
    trans_aug_range = torch.tensor([0.125, 0.125, 0.125])
    rot_aug_range = [0.0, 0.0, 45.0]
    rot_aug_resolution = 5
    voxel_size = 100
    rot_resolution = 5
    device = 'cpu'

    before_np = convert_to_numpy(pcd[0])
    plot_point_cloud(before_np, left_action_gripper_pose, right_action_gripper_pose, "Before Augmentation", os.path.join(root_path, "before.png"))

    # Call the function
    outputs = bimanual_apply_se3_augmentation(
        pcd,
        right_action_gripper_pose,
        right_action_trans,
        right_action_rot_grip,
        left_action_gripper_pose,
        left_action_trans,
        left_action_rot_grip,
        bounds,
        layer,
        trans_aug_range,
        rot_aug_range,
        rot_aug_resolution,
        voxel_size,
        rot_resolution,
        device,
    )
    # Unpack outputs
    right_action_trans_out, right_action_rot_grip_out, left_action_trans_out, left_action_rot_grip_out, pcd_out = outputs

    # Visualize point clouds

    after_np = convert_to_numpy(pcd_out[0])

    # Visualize and save before and after augmentation

    left_action_gripper_pose = np.concatenate([left_action_trans_out, left_action_rot_grip_out], axis=1)
    right_action_gripper_pose = np.concatenate([right_action_trans_out, right_action_rot_grip_out], axis=1)
    plot_point_cloud(after_np, left_action_gripper_pose, right_action_gripper_pose, "After Augmentation", os.path.join(root_path, "after.png"))


