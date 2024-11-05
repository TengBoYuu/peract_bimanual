import numpy as np
import torch
from helpers import utils
from pytorch3d import transforms as torch3d_tf
import time


# def perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds):
#     """Perturb point clouds with given transformation.
#     :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
#     :param trans_shift_4x4: translation matrix [bs, 4, 4]
#     :param rot_shift_4x4: rotation matrix [bs, 4, 4]
#     :param action_gripper_4x4: original keyframe action gripper pose [bs, 4, 4]
#     :param bounds: metric scene bounds [bs, 6]
#     :return: peturbed point clouds
#     """
#     # batch bounds if necessary
#     bs = pcd[0].shape[0]
#     if bounds.shape[0] != bs:
#         bounds = bounds.repeat(bs, 1)

#     perturbed_pcd = []
#     for p in pcd:
#         p_shape = p.shape
#         num_points = p_shape[-1] * p_shape[-2]

#         action_trans_3x1 = (
#             action_gripper_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
#         )
#         trans_shift_3x1 = (
#             trans_shift_4x4[:, 0:3, 3].unsqueeze(-1).repeat(1, 1, num_points)
#         )

#         # flatten point cloud
#         p_flat = p.reshape(bs, 3, -1)
#         p_flat_4x1_action_origin = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)

#         # shift points to have action_gripper pose as the origin
#         p_flat_4x1_action_origin[:, :3, :] = p_flat - action_trans_3x1

#         # apply rotation
#         perturbed_p_flat_4x1_action_origin = torch.bmm(
#             p_flat_4x1_action_origin.transpose(2, 1), rot_shift_4x4
#         ).transpose(2, 1)

#         # apply bounded translations
#         bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
#         bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
#         bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

#         action_then_trans_3x1 = action_trans_3x1 + trans_shift_3x1
#         action_then_trans_3x1_x = torch.clamp(
#             action_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
#         )
#         action_then_trans_3x1_y = torch.clamp(
#             action_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
#         )
#         action_then_trans_3x1_z = torch.clamp(
#             action_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
#         )
#         action_then_trans_3x1 = torch.stack(
#             [action_then_trans_3x1_x, action_then_trans_3x1_y, action_then_trans_3x1_z],
#             dim=1,
#         )

#         # shift back the origin
#         perturbed_p_flat_3x1 = (
#             perturbed_p_flat_4x1_action_origin[:, :3, :] + action_then_trans_3x1
#         )

#         perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
#         perturbed_pcd.append(perturbed_p)
#     return perturbed_pcd

def bimanual_perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, left_action_gripper_4x4, bounds):
    """Perturb point clouds with given transformation centered around the middle of left and right grippers.
    
    :param pcd: list of point clouds [[bs, 3, N], ...] for N cameras
    :param trans_shift_4x4: translation matrix [bs, 4, 4]
    :param rot_shift_4x4: rotation matrix [bs, 4, 4]
    :param right_action_gripper_4x4: right arm keyframe action gripper pose [bs, 4, 4]
    :param left_action_gripper_4x4: left arm keyframe action gripper pose [bs, 4, 4]
    :param bounds: metric scene bounds [bs, 6]
    :return: perturbed point clouds
    """
    # Batch bounds if necessary
    bs = pcd[0].shape[0]
    if bounds.shape[0] != bs:
        bounds = bounds.repeat(bs, 1)

    center_trans_3x1 = (
        (right_action_gripper_4x4[:, :3, 3] + left_action_gripper_4x4[:, :3, 3]) / 2
    ).unsqueeze(-1)
    
    perturbed_pcd = []
    for p in pcd:
        p_shape = p.shape
        num_points = p_shape[-1] * p_shape[-2]

        center_trans_3x1_repeated = center_trans_3x1.repeat(1, 1, num_points)
        trans_shift_3x1 = (
            trans_shift_4x4[:, :3, 3].unsqueeze(-1).repeat(1, 1, num_points)
        )

        p_flat = p.reshape(bs, 3, -1)
        p_flat_4x1_centered = torch.ones(bs, 4, p_flat.shape[-1]).to(p_flat.device)
        p_flat_4x1_centered[:, :3, :] = p_flat - center_trans_3x1_repeated

        perturbed_p_flat_4x1_centered = torch.bmm(
            p_flat_4x1_centered.transpose(2, 1), rot_shift_4x4
        ).transpose(2, 1)

        bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
        bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
        bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

        center_then_trans_3x1 = center_trans_3x1_repeated + trans_shift_3x1
        center_then_trans_3x1_x = torch.clamp(
            center_then_trans_3x1[:, 0], min=bounds_x_min, max=bounds_x_max
        )
        center_then_trans_3x1_y = torch.clamp(
            center_then_trans_3x1[:, 1], min=bounds_y_min, max=bounds_y_max
        )
        center_then_trans_3x1_z = torch.clamp(
            center_then_trans_3x1[:, 2], min=bounds_z_min, max=bounds_z_max
        )
        center_then_trans_3x1 = torch.stack(
            [center_then_trans_3x1_x, center_then_trans_3x1_y, center_then_trans_3x1_z],
            dim=1,
        )

        perturbed_p_flat_3x1 = (
            perturbed_p_flat_4x1_centered[:, :3, :] + center_then_trans_3x1
        )

        perturbed_p = perturbed_p_flat_3x1.reshape(p_shape)
        perturbed_pcd.append(perturbed_p)

    return perturbed_pcd


def apply_transformation_to_action(
    right_action_trans,
    left_action_trans,
    right_action_rot_grip,
    left_action_rot_grip,
    center_4x4,
    trans_shift_4x4,
    rot_shift_4x4,
    bounds
):
    # print(trans_shift_4x4)
    right_action_rot_grip[:, :3] = torch.bmm(
        rot_shift_4x4[:, :3, :3], (right_action_rot_grip[:, :3] - center_4x4[:, :3, 3]).unsqueeze(-1)
    ).squeeze(-1) + center_4x4[:, :3, 3]
    
    left_action_rot_grip[:, :3] = torch.bmm(
        rot_shift_4x4[:, :3, :3], (left_action_rot_grip[:, :3] - center_4x4[:, :3, 3]).unsqueeze(-1)
    ).squeeze(-1) + center_4x4[:, :3, 3]

    trans_shift_4x4 = trans_shift_4x4.int()
    right_action_trans = right_action_trans.int()
    left_action_trans = left_action_trans.int()

    right_action_trans += trans_shift_4x4[:, :3, 3]
    left_action_trans += trans_shift_4x4[:, :3, 3]

    # right_action_trans = apply_transformation_to_single_point(right_action_trans, center_4x4[:, :3, 3], trans_shift_4x4, rot_shift_4x4, bounds)

    # print(right_action_trans.shape)
    # print(right_action_rot_grip.shape)
    # print(left_action_trans.shape)
    # print(left_action_rot_grip.shape)
    print(right_action_trans)
    print(right_action_rot_grip)
    # print(left_action_trans)
    # print(left_action_rot_grip)
    return right_action_trans, left_action_trans, right_action_rot_grip, left_action_rot_grip

def apply_transformation_to_single_point(single_point, center_trans_3x1, trans_shift_4x4, rot_shift_4x4, bounds):

    single_point_centered = single_point - center_trans_3x1.squeeze(-1)
    print(single_point)
    print(center_trans_3x1)

    single_point_rotated = torch.bmm(
        rot_shift_4x4[:, :3, :3], single_point_centered.unsqueeze(-1)
    ).squeeze(-1)
    

    trans_shift_3x1 = trans_shift_4x4[:, :3, 3]
    single_point_transformed = single_point_rotated + center_trans_3x1.squeeze(-1) + trans_shift_3x1


    bounds_x_min, bounds_x_max = bounds[:, 0].min(), bounds[:, 3].max()
    bounds_y_min, bounds_y_max = bounds[:, 1].min(), bounds[:, 4].max()
    bounds_z_min, bounds_z_max = bounds[:, 2].min(), bounds[:, 5].max()

    single_point_transformed[:, 0] = torch.clamp(single_point_transformed[:, 0], min=bounds_x_min, max=bounds_x_max)
    single_point_transformed[:, 1] = torch.clamp(single_point_transformed[:, 1], min=bounds_y_min, max=bounds_y_max)
    single_point_transformed[:, 2] = torch.clamp(single_point_transformed[:, 2], min=bounds_z_min, max=bounds_z_max)

    return single_point_transformed

def bimanual_apply_se3_augmentation(
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
):
    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    right_action_gripper_trans = right_action_gripper_pose[:, :3]
    right_action_gripper_quat_wxyz = torch.cat(
        (
            right_action_gripper_pose[:, 6].unsqueeze(1),
            right_action_gripper_pose[:, 3:6],
        ),
        dim=1,
    )

    right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        right_action_gripper_quat_wxyz
    )
    right_action_gripper_4x4 = identity_4x4.detach().clone()
    right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
    right_action_gripper_4x4[:, 0:3, 3] = right_action_gripper_trans

    right_perturbed_trans = torch.full_like(right_action_trans, -1.0)
    right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1.0)

    left_action_gripper_trans = left_action_gripper_pose[:, :3]
    left_action_gripper_quat_wxyz = torch.cat(
        (left_action_gripper_pose[:, 6].unsqueeze(1), left_action_gripper_pose[:, 3:6]),
        dim=1,
    )

    left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
        left_action_gripper_quat_wxyz
    )
    left_action_gripper_4x4 = identity_4x4.detach().clone()
    left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
    left_action_gripper_4x4[:, 0:3, 3] = left_action_gripper_trans

    left_perturbed_trans = torch.full_like(left_action_trans, -1.0)
    left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0

    # print(right_action_gripper_4x4)
    # while torch.any(right_perturbed_trans < 0) and torch.any(left_perturbed_trans < 0):
    while torch.any(right_perturbed_trans < 0) or torch.any(left_perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        #########################################

        average_action_gripper_4x4 = (right_action_gripper_4x4 + left_action_gripper_4x4) / 2
        
        # right_trans_shifted = right_action_trans - average_action_gripper_4x4[:, :3, 3]
        # left_trans_shifted = left_action_trans - average_action_gripper_4x4[:, :3, 3]
        # rotated_right_action_trans = torch.bmm(rot_shift_4x4[:, :3, :3], right_trans_shifted.unsqueeze(-1)).squeeze(-1)
        # rotated_left_action_trans = torch.bmm(rot_shift_4x4[:, :3, :3], left_trans_shifted.unsqueeze(-1)).squeeze(-1)
        # right_action_trans = rotated_right_action_trans + trans_shift + average_action_gripper_4x4[:, :3, 3]
        # left_action_trans = rotated_left_action_trans + trans_shift + average_action_gripper_4x4[:, :3, 3]
        # right_action_rot_grip[:, :3] = torch.bmm(rot_shift_4x4[:, :3, :3], right_action_rot_grip[:, :3].unsqueeze(-1)).squeeze(-1)
        # left_action_rot_grip[:, :3] = torch.bmm(rot_shift_4x4[:, :3, :3], left_action_rot_grip[:, :3].unsqueeze(-1)).squeeze(-1)

        # rotate then translate the 4x4 keyframe action
        right_perturbed_action_gripper_4x4 = torch.bmm(
            right_action_gripper_4x4, rot_shift_4x4
        )

        right_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        right_perturbed_action_trans = (
            right_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        right_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            right_perturbed_action_gripper_4x4[:, :3, :3]
        )
        right_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    right_perturbed_action_quat_wxyz[:, 1:],
                    right_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # rotate then translate the 4x4 keyframe action
        left_perturbed_action_gripper_4x4 = torch.bmm(
            left_action_gripper_4x4, rot_shift_4x4
        )

        left_perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        left_perturbed_action_trans = (
            left_perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        )
        left_perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            left_perturbed_action_gripper_4x4[:, :3, :3]
        )
        left_perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    left_perturbed_action_quat_wxyz[:, 1:],
                    left_perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        ####################################
        ####################################
        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        right_trans_indicies, right_rot_grip_indicies = [], []
        left_trans_indicies, left_rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            right_trans_idx = utils.point_to_voxel_index(
                right_perturbed_action_trans[b], voxel_size, bounds_np
            )
            right_trans_indicies.append(right_trans_idx.tolist())

            right_quat = right_perturbed_action_quat_xyzw[b]
            right_quat = utils.normalize_quaternion(right_perturbed_action_quat_xyzw[b])
            if right_quat[-1] < 0:
                right_quat = -right_quat
            right_disc_rot = utils.quaternion_to_discrete_euler(
                right_quat, rot_resolution
            )
            right_rot_grip_indicies.append(
                right_disc_rot.tolist()
                + [int(right_action_rot_grip[b, 3].cpu().numpy())]
            )

            left_trans_idx = utils.point_to_voxel_index(
                left_perturbed_action_trans[b], voxel_size, bounds_np
            )
            left_trans_indicies.append(left_trans_idx.tolist())

            left_quat = left_perturbed_action_quat_xyzw[b]
            left_quat = utils.normalize_quaternion(left_perturbed_action_quat_xyzw[b])
            if left_quat[-1] < 0:
                left_quat = -left_quat
            left_disc_rot = utils.quaternion_to_discrete_euler(
                left_quat, rot_resolution
            )
            left_rot_grip_indicies.append(
                left_disc_rot.tolist() + [int(left_action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        right_perturbed_trans = torch.from_numpy(np.array(right_trans_indicies)).to(
            device=device
        )
        right_perturbed_rot_grip = torch.from_numpy(
            np.array(right_rot_grip_indicies)
        ).to(device=device)

        left_perturbed_trans = torch.from_numpy(np.array(left_trans_indicies)).to(
            device=device
        )
        left_perturbed_rot_grip = torch.from_numpy(np.array(left_rot_grip_indicies)).to(
            device=device
        )

    right_action_trans = right_perturbed_trans
    right_action_rot_grip = right_perturbed_rot_grip

    left_action_trans = left_perturbed_trans
    left_action_rot_grip = left_perturbed_rot_grip

    # # print(right_action_trans.shape)
    # # print(right_action_rot_grip.shape)
    # # print(left_action_trans.shape)
    # # print(left_action_rot_grip.shape)
    # print(right_action_trans)
    # print(right_action_rot_grip)
    # print(left_action_trans)
    # print(left_action_rot_grip)
    # right_action_trans, left_action_trans, right_action_rot_grip, left_action_rot_grip = apply_transformation_to_action(
    #     right_action_trans,
    #     left_action_trans,
    #     right_action_rot_grip,
    #     left_action_rot_grip,
    #     average_action_gripper_4x4, 
    #     trans_shift_4x4,
    #     rot_shift_4x4,
    #     bounds
    # )
    # apply perturbation to pointclouds
    pcd = bimanual_perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, left_action_gripper_4x4, bounds)
    # print(pcd[0][0][0])
    # pcd = perturb_se3(
    #     pcd, trans_shift_4x4, rot_shift_4x4, right_action_gripper_4x4, bounds
    # )

    return (
        right_action_trans,
        right_action_rot_grip,
        left_action_trans,
        left_action_rot_grip,
        pcd,
    )

# def bimanual_apply_se3_augmentation(
#     pcd,
#     right_action_gripper_pose,
#     right_action_trans,
#     right_action_rot_grip,
#     left_action_gripper_pose,
#     left_action_trans,
#     left_action_rot_grip,
#     bounds,
#     layer,
#     trans_aug_range,
#     rot_aug_range,
#     rot_aug_resolution,
#     voxel_size,
#     rot_resolution,
#     device,
# ):
#     # batch size
#     bs = pcd[0].shape[0]

#     # identity matrix
#     identity_4x4 = torch.eye(4, device=device).unsqueeze(0).repeat(bs, 1, 1)

#     # Prepare gripper poses for right arm
#     right_action_gripper_trans = right_action_gripper_pose[:, :3]
#     right_action_gripper_quat_wxyz = torch.cat(
#         (
#             right_action_gripper_pose[:, 6].unsqueeze(1),
#             right_action_gripper_pose[:, 3:6],
#         ),
#         dim=1,
#     )

#     right_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
#         right_action_gripper_quat_wxyz
#     )
#     right_action_gripper_4x4 = identity_4x4.clone()
#     right_action_gripper_4x4[:, :3, :3] = right_action_gripper_rot
#     right_action_gripper_4x4[:, :3, 3] = right_action_gripper_trans

#     # Prepare gripper poses for left arm
#     left_action_gripper_trans = left_action_gripper_pose[:, :3]
#     left_action_gripper_quat_wxyz = torch.cat(
#         (
#             left_action_gripper_pose[:, 6].unsqueeze(1),
#             left_action_gripper_pose[:, 3:6],
#         ),
#         dim=1,
#     )

#     left_action_gripper_rot = torch3d_tf.quaternion_to_matrix(
#         left_action_gripper_quat_wxyz
#     )
#     left_action_gripper_4x4 = identity_4x4.clone()
#     left_action_gripper_4x4[:, :3, :3] = left_action_gripper_rot
#     left_action_gripper_4x4[:, :3, 3] = left_action_gripper_trans

#     # Initialize perturbed action variables
#     right_perturbed_trans = torch.full_like(right_action_trans, -1)
#     right_perturbed_rot_grip = torch.full_like(right_action_rot_grip, -1)
#     left_perturbed_trans = torch.full_like(left_action_trans, -1)
#     left_perturbed_rot_grip = torch.full_like(left_action_rot_grip, -1)

#     perturb_attempts = 0
#     max_attempts = 100

#     # Ensure voxel_size is a tensor
#     if not isinstance(voxel_size, torch.Tensor):
#         voxel_size_tensor = torch.tensor(voxel_size, device=device)
#     else:
#         voxel_size_tensor = voxel_size.to(device)

#     while torch.any(right_perturbed_trans < 0) or torch.any(left_perturbed_trans < 0):
#         perturb_attempts += 1
#         if perturb_attempts > max_attempts:
#             raise Exception("Failing to perturb action and keep it within bounds.")

#         # Sample translation perturbation with specified range
#         trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(device)
#         trans_shift = (torch.rand(bs, 3, device=device) * 2 - 1) * trans_range
#         trans_shift_4x4 = identity_4x4.clone()
#         trans_shift_4x4[:, :3, 3] = trans_shift

#         # Sample rotation perturbation at specified resolution and range
#         roll_aug_steps = int(rot_aug_range[0] / rot_aug_resolution)
#         pitch_aug_steps = int(rot_aug_range[1] / rot_aug_resolution)
#         yaw_aug_steps = int(rot_aug_range[2] / rot_aug_resolution)

#         roll = (torch.randint(-roll_aug_steps, roll_aug_steps + 1, (bs, 1), device=device)
#                 * np.deg2rad(rot_aug_resolution))
#         pitch = (torch.randint(-pitch_aug_steps, pitch_aug_steps + 1, (bs, 1), device=device)
#                  * np.deg2rad(rot_aug_resolution))
#         yaw = (torch.randint(-yaw_aug_steps, yaw_aug_steps + 1, (bs, 1), device=device)
#                * np.deg2rad(rot_aug_resolution))

#         euler_angles = torch.cat((roll, pitch, yaw), dim=1)
#         rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(euler_angles, convention='XYZ')
#         rot_shift_4x4 = identity_4x4.clone()
#         rot_shift_4x4[:, :3, :3] = rot_shift_3x3

#         # Compute the center point between the two grippers
#         center_trans_3x1 = (
#             (right_action_gripper_4x4[:, :3, 3] + left_action_gripper_4x4[:, :3, 3]) / 2
#         ).unsqueeze(-1)  # [bs, 3, 1]

#         # Apply rotation and translation to the right action gripper pose
#         right_action_trans_centered = (
#             right_action_gripper_4x4[:, :3, 3].unsqueeze(-1) - center_trans_3x1
#         )  # [bs, 3, 1]
#         right_action_trans_rotated = torch.bmm(
#             rot_shift_3x3, right_action_trans_centered
#         )  # [bs, 3, 1]
#         right_action_trans_transformed = (
#             right_action_trans_rotated.squeeze(-1)
#             + center_trans_3x1.squeeze(-1)
#             + trans_shift
#         )  # [bs, 3]

#         # Apply rotation and translation to the left action gripper pose
#         left_action_trans_centered = (
#             left_action_gripper_4x4[:, :3, 3].unsqueeze(-1) - center_trans_3x1
#         )  # [bs, 3, 1]
#         left_action_trans_rotated = torch.bmm(
#             rot_shift_3x3, left_action_trans_centered
#         )  # [bs, 3, 1]
#         left_action_trans_transformed = (
#             left_action_trans_rotated.squeeze(-1)
#             + center_trans_3x1.squeeze(-1)
#             + trans_shift
#         )  # [bs, 3]

#         # Apply rotation to the orientations
#         right_action_rot_mat = right_action_gripper_4x4[:, :3, :3]
#         left_action_rot_mat = left_action_gripper_4x4[:, :3, :3]

#         right_action_rot_transformed = torch.bmm(
#             rot_shift_3x3, right_action_rot_mat
#         )  # [bs, 3, 3]
#         left_action_rot_transformed = torch.bmm(
#             rot_shift_3x3, left_action_rot_mat
#         )  # [bs, 3, 3]

#         # Convert rotation matrices to quaternions
#         right_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
#             right_action_rot_transformed
#         )  # [bs, 4]
#         left_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
#             left_action_rot_transformed
#         )  # [bs, 4]

#         # Convert quaternions to xyzw format
#         right_action_quat_xyzw = torch.cat(
#             [right_action_quat_wxyz[:, 1:], right_action_quat_wxyz[:, 0:1]], dim=1
#         )  # [bs, 4]
#         left_action_quat_xyzw = torch.cat(
#             [left_action_quat_wxyz[:, 1:], left_action_quat_wxyz[:, 0:1]], dim=1
#         )  # [bs, 4]

#         # Discretize perturbed translation and rotation
#         right_trans_indices, right_rot_grip_indices = [], []
#         left_trans_indices, left_rot_grip_indices = [], []

#         for b in range(bs):
#             bounds_idx = b if layer > 0 else 0
#             bounds_tensor = bounds[bounds_idx].to(device)

#             # Right arm
#             print("right_action_trans_transformed[b] device:", right_action_trans_transformed[b].device)
#             print("voxel_size_tensor device:", voxel_size_tensor.device)
#             print("bounds_tensor device:", bounds_tensor.device)
#             right_trans_idx = utils.point_to_voxel_index(
#                 right_action_trans_transformed[b], voxel_size_tensor, bounds_tensor
#             )
#             right_trans_indices.append(right_trans_idx)

#             right_quat = right_action_quat_xyzw[b]
#             right_quat = utils.normalize_quaternion(right_quat)
#             if right_quat[-1] < 0:
#                 right_quat = -right_quat
#             right_disc_rot = utils.quaternion_to_discrete_euler(
#                 right_quat, rot_resolution, device
#             )
#             right_rot_grip_indices.append(
#                 torch.cat(
#                     [right_disc_rot, right_action_rot_grip[b, -1:].long()]
#                 )
#             )

#             # Left arm
#             left_trans_idx = utils.point_to_voxel_index(
#                 left_action_trans_transformed[b], voxel_size_tensor, bounds_tensor
#             )
#             left_trans_indices.append(left_trans_idx)

#             left_quat = left_action_quat_xyzw[b]
#             left_quat = utils.normalize_quaternion(left_quat)
#             if left_quat[-1] < 0:
#                 left_quat = -left_quat
#             left_disc_rot = utils.quaternion_to_discrete_euler(
#                 left_quat, rot_resolution, device
#             )
#             left_rot_grip_indices.append(
#                 torch.cat(
#                     [left_disc_rot, left_action_rot_grip[b, -1:].long()]
#                 )
#             )

#         # Convert lists to tensors
#         right_perturbed_trans = torch.stack(right_trans_indices).to(device)
#         right_perturbed_rot_grip = torch.stack(right_rot_grip_indices).to(device)
#         left_perturbed_trans = torch.stack(left_trans_indices).to(device)
#         left_perturbed_rot_grip = torch.stack(left_rot_grip_indices).to(device)

#         # Check if perturbed actions are within bounds
#         right_in_bounds = torch.all(right_perturbed_trans >= 0, dim=1)
#         left_in_bounds = torch.all(left_perturbed_trans >= 0, dim=1)

#         if torch.all(right_in_bounds) and torch.all(left_in_bounds):
#             break

#     # Update action variables
#     right_action_trans = right_perturbed_trans
#     right_action_rot_grip = right_perturbed_rot_grip
#     left_action_trans = left_perturbed_trans
#     left_action_rot_grip = left_perturbed_rot_grip

#     # Apply perturbation to pointclouds
#     pcd = bimanual_perturb_se3(
#         pcd,
#         trans_shift_4x4,
#         rot_shift_4x4,
#         right_action_gripper_4x4,
#         left_action_gripper_4x4,
#         bounds,
#     )

#     return (
#         right_action_trans,
#         right_action_rot_grip,
#         left_action_trans,
#         left_action_rot_grip,
#         pcd,
#     )

def apply_se3_augmentation(
    pcd,
    action_gripper_pose,
    action_trans,
    action_rot_grip,
    bounds,
    layer,
    trans_aug_range,
    rot_aug_range,
    rot_aug_resolution,
    voxel_size,
    rot_resolution,
    device,
):
    """Apply SE3 augmentation to a point clouds and actions.
    :param pcd: list of point clouds [[bs, 3, H, W], ...] for N cameras
    :param action_gripper_pose: 6-DoF pose of keyframe action [bs, 7]
    :param action_trans: discretized translation action [bs, 3]
    :param action_rot_grip: discretized rotation and gripper action [bs, 4]
    :param bounds: metric scene bounds of voxel grid [bs, 6]
    :param layer: voxelization layer (always 1 for PerAct)
    :param trans_aug_range: range of translation augmentation [x_range, y_range, z_range]
    :param rot_aug_range: range of rotation augmentation [x_range, y_range, z_range]
    :param rot_aug_resolution: degree increments for discretized augmentation rotations
    :param voxel_size: voxelization resoltion
    :param rot_resolution: degree increments for discretized rotations
    :param device: torch device
    :return: perturbed action_trans, action_rot_grip, pcd
    """

    # batch size
    bs = pcd[0].shape[0]

    # identity matrix
    identity_4x4 = torch.eye(4).unsqueeze(0).repeat(bs, 1, 1).to(device=device)

    # 4x4 matrix of keyframe action gripper pose
    action_gripper_trans = action_gripper_pose[:, :3]
    action_gripper_quat_wxyz = torch.cat(
        (action_gripper_pose[:, 6].unsqueeze(1), action_gripper_pose[:, 3:6]), dim=1
    )
    action_gripper_rot = torch3d_tf.quaternion_to_matrix(action_gripper_quat_wxyz)
    action_gripper_4x4 = identity_4x4.detach().clone()
    action_gripper_4x4[:, :3, :3] = action_gripper_rot
    action_gripper_4x4[:, 0:3, 3] = action_gripper_trans

    perturbed_trans = torch.full_like(action_trans, -1.0)
    perturbed_rot_grip = torch.full_like(action_rot_grip, -1.0)

    # perturb the action, check if it is within bounds, if not, try another perturbation
    perturb_attempts = 0
    while torch.any(perturbed_trans < 0):
        # might take some repeated attempts to find a perturbation that doesn't go out of bounds
        perturb_attempts += 1
        if perturb_attempts > 100:
            print("perturbation can not within bounds")
            return action_trans, action_gripper_pose, pcd
            # raise Exception("Failing to perturb action and keep it within bounds.")

        # sample translation perturbation with specified range
        trans_range = (bounds[:, 3:] - bounds[:, :3]) * trans_aug_range.to(
            device=device
        )
        trans_shift = trans_range * utils.rand_dist((bs, 3)).to(device=device)
        trans_shift_4x4 = identity_4x4.detach().clone()
        trans_shift_4x4[:, 0:3, 3] = trans_shift

        # sample rotation perturbation at specified resolution and range
        roll_aug_steps = int(rot_aug_range[0] // rot_aug_resolution)
        pitch_aug_steps = int(rot_aug_range[1] // rot_aug_resolution)
        yaw_aug_steps = int(rot_aug_range[2] // rot_aug_resolution)

        roll = utils.rand_discrete(
            (bs, 1), min=-roll_aug_steps, max=roll_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        pitch = utils.rand_discrete(
            (bs, 1), min=-pitch_aug_steps, max=pitch_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        yaw = utils.rand_discrete(
            (bs, 1), min=-yaw_aug_steps, max=yaw_aug_steps
        ) * np.deg2rad(rot_aug_resolution)
        rot_shift_3x3 = torch3d_tf.euler_angles_to_matrix(
            torch.cat((roll, pitch, yaw), dim=1), "XYZ"
        )
        rot_shift_4x4 = identity_4x4.detach().clone()
        rot_shift_4x4[:, :3, :3] = rot_shift_3x3

        # rotate then translate the 4x4 keyframe action
        perturbed_action_gripper_4x4 = torch.bmm(action_gripper_4x4, rot_shift_4x4)
        perturbed_action_gripper_4x4[:, 0:3, 3] += trans_shift

        # convert transformation matrix to translation + quaternion
        perturbed_action_trans = perturbed_action_gripper_4x4[:, 0:3, 3].cpu().numpy()
        perturbed_action_quat_wxyz = torch3d_tf.matrix_to_quaternion(
            perturbed_action_gripper_4x4[:, :3, :3]
        )
        perturbed_action_quat_xyzw = (
            torch.cat(
                [
                    perturbed_action_quat_wxyz[:, 1:],
                    perturbed_action_quat_wxyz[:, 0].unsqueeze(1),
                ],
                dim=1,
            )
            .cpu()
            .numpy()
        )

        # discretize perturbed translation and rotation
        # TODO(mohit): do this in torch without any numpy.
        trans_indicies, rot_grip_indicies = [], []
        for b in range(bs):
            bounds_idx = b if layer > 0 else 0
            bounds_np = bounds[bounds_idx].cpu().numpy()

            trans_idx = utils.point_to_voxel_index(
                perturbed_action_trans[b], voxel_size, bounds_np
            )
            trans_indicies.append(trans_idx.tolist())

            quat = perturbed_action_quat_xyzw[b]
            quat = utils.normalize_quaternion(perturbed_action_quat_xyzw[b])
            if quat[-1] < 0:
                quat = -quat
            disc_rot = utils.quaternion_to_discrete_euler(quat, rot_resolution)
            rot_grip_indicies.append(
                disc_rot.tolist() + [int(action_rot_grip[b, 3].cpu().numpy())]
            )

        # if the perturbed action is out of bounds,
        # the discretized perturb_trans should have invalid indicies
        perturbed_trans = torch.from_numpy(np.array(trans_indicies)).to(device=device)
        perturbed_rot_grip = torch.from_numpy(np.array(rot_grip_indicies)).to(
            device=device
        )

    action_trans = perturbed_trans
    action_rot_grip = perturbed_rot_grip

    # apply perturbation to pointclouds
    pcd = perturb_se3(pcd, trans_shift_4x4, rot_shift_4x4, action_gripper_4x4, bounds)

    return action_trans, action_rot_grip, pcd



if __name__ == "__main__":
    from helpers.utils import visualise_voxel, stack_on_channel
    import visdom
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.spatial.transform import Rotation as R

    # def convert_to_visdom_format(pcd_tensor):
    #     # Convert tensor to numpy and reshape for Visdom
    #     pcd_np = pcd_tensor.squeeze().cpu().numpy()
    #     pcd_np = pcd_np.reshape(-1, 3)
    #     return pcd_np
    # def visualize_point_clouds(before_pcd, after_pcd):
    #     # vis = visdom.Visdom(port=8098)
    #     vis = visdom.Visdom(port=8097)

    #     # Convert point clouds to Visdom-compatible format
    #     before_np = convert_to_visdom_format(before_pcd)
    #     after_np = convert_to_visdom_format(after_pcd)

    #     # Visualize before point cloud
    #     vis.scatter(
    #         X=before_np,
    #         opts=dict(
    #             title="Before Augmentation",
    #             markersize=2,
    #             markercolor=np.array([[255, 0, 0]]),  # Red
    #         )
    #     )
    #     # Visualize after point cloud
    #     vis.scatter(
    #         X=after_np,
    #         opts=dict(
    #             title="After Augmentation",
    #             markersize=2,
    #             markercolor=np.array([[0, 255, 0]]),  # Green
    #         )
    #     )
    def convert_to_numpy(pcd_tensor):
        pcd_np = pcd_tensor.squeeze().cpu().numpy()
        pcd_np = pcd_np.reshape(-1, 3)
        return pcd_np

    def plot_point_cloud(pcd, gripper_pose, title, filename):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c='b', marker='o', s=1)

        # Draw gripper axes
        for pose in gripper_pose:
            trans = pose[:3]
            rot = pose[3:]
            rotation = R.from_quat(rot).as_matrix()

            # Create axis lines
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
    pcd = [torch.rand(batch_size, 3, 1, 1) for _ in range(1)]
    right_action_gripper_pose = torch.rand(batch_size, 7)
    right_action_trans = torch.rand(batch_size, 3)
    right_action_rot_grip = torch.rand(batch_size, 4)
    left_action_gripper_pose = torch.rand(batch_size, 7)
    left_action_trans = torch.rand(batch_size, 3)
    left_action_rot_grip = torch.rand(batch_size, 4)
    bounds = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    layer = 0
    trans_aug_range = torch.tensor([0.125, 0.125, 0.125])
    rot_aug_range = [0.0, 0.0, 45.0]
    rot_aug_resolution = 5
    voxel_size = [100]
    rot_resolution = 5
    device = 'cpu'

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

    # Visualize before and after point clouds
    # visualize_point_clouds(pcd[0], pcd_out[0])
    before_np = convert_to_numpy(pcd[0])
    after_np = convert_to_numpy(pcd_out[0])

    # Visualize and save before and after augmentation
    plot_point_cloud(before_np, right_action_gripper_pose, "Before Augmentation", "before.png")
    plot_point_cloud(after_np, right_action_gripper_pose, "After Augmentation", "after.png")

    # Display results
    # rendered_img_right = visualise_voxel(
    #     voxel_grid[0].cpu().detach().numpy(),    # [10, 100, 100, 100]
    #     None,
    #     None,
    #     right_action_trans_out,
    #     voxel_size=0.045,
    #     rotation_amount=np.deg2rad(-90),
    #     highlight_alpha=1.0,
    #     alpha=0.4,
    # )

    # hold on to visualize
    # while True:
    #     print("Hold on to visualize")
    #     time.sleep(100)
