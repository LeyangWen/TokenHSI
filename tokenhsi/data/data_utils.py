import sys
sys.path.append("./")

import numpy as np
import torch
import pickle

from lpanlib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState, SkeletonMotion
from lpanlib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion_interactive
from lpanlib.poselib.core.rotation3d import quat_mul, quat_from_angle_axis, quat_mul_norm, quat_rotate, quat_identity

def process_smplest_seq(fname, output_path, visualize=False, target_fps=20):

    # load raw params from smplest batch inference
    with open(fname, "rb") as f:
        raw_params = pickle.load(f)
        
    
    # dict(np.load(fname, allow_pickle=True))

    poses = raw_params["poses"]
    trans = raw_params["trans"]
    J = raw_params.get("smplx_joints_cam", None)
    fps = raw_params["fps"]
    assert J is not None, "Missing key 'smplx_joints_cam' in input pkl"
    
    # center
    # trans = trans - trans[0]


    # downsample from 20hz to 20hz
    source_fps = fps
    
    assert source_fps % target_fps == 0, f"source_fps {source_fps} not divisible by target_fps {target_fps}"
    skip = int(source_fps // target_fps)
    poses = poses[::skip]
    trans = trans[::skip]
    J = J[::skip]
    # extract 24 SMPL joints from 55 SMPL-X joints
    joints_to_use = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
    )
    joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
    poses = poses[:, joints_to_use]
    
    # Y up to Z-up
    # Rx = rot_x(-np.pi / 2.0).astype(np.float32)      # MOD: choose sign per your viewer

    # root_aa = poses[:, :3]                           # MOD
    # R_root = axis_angle_to_matrix_batch(root_aa)     # MOD
    # R_root_new = Rx[None] @ R_root                   # MOD: left-multiply
    # poses[:, :3] = matrix_to_axis_angle_batch(R_root_new)  # MOD

    # Smooth root, use this axis rotate instead to avoid sudden spin & jerk
    root_aa_smoothed, jerk_mask, theta = smooth_root_with_jerk_fix(
        poses[:, :3],
        fps=target_fps,
        cutoff_hz=1.0,       # try 0.8–2.0
        order=2,
        apply_rx=True,      # keep False if you rotate elsewhere
        jerk_abs_thresh=0.1, # absolute threshold in rad/frame
        jerk_mad_k=4.0,      # robust (MAD) multiplier
        search_radius=2,     # search up to ~2 frames each side for clean neighbors
        return_format='aa'
    )
    poses[:, :3] = root_aa_smoothed

    for j in range(1, 24):
        seg = poses[:, j*3:(j+1)*3]
        seg = smooth_axis_angle_butter(seg, fps=target_fps, cutoff_hz=1.0, order=4)
        seg = smooth_axis_angle_butter(seg, fps=target_fps, cutoff_hz=1.0, order=4)  # second pass
        poses[:, j*3:(j+1)*3] = seg
    print(poses[:, :3])
    ##### trans from foot contact: assume walking
    # Y-up → Z-up: (x, y, z) -> (x, z, y)
    J_zup = np.stack([J[..., 0], J[..., 2], -J[..., 1]], axis=-1)  # (T, J, 3)
    
    if visualize:
        # plot all J[frame] with index, to see which joints to use for foot contact
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        J = J_zup[:15, :,:]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for j in range(J.shape[1]):
            ax.plot(J[:, j, 0], J[:, j, 1], J[:, j, 2], label=str(j))
            ax.text(J[0, j, 0], J[0, j, 1], J[0, j, 2], str(j))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        # xyz all same scale
        max_range = np.array([J[:, :, 0].max()-J[:, :, 0].min(), J[:, :, 1].max()-J[:, :, 1].min(), J[:, :, 2].max()-J[:, :, 2].min()]).max() / 2.0
        mid_x = (J[:, :, 0].max()+J[:, :, 0].min()) * 0.5
        mid_y = (J[:, :, 1].max()+J[:, :, 1].min()) * 0.5
        mid_z = (J[:, :, 2].max()+J[:, :, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        plt.title('Joint Trajectories')
        plt.legend()
        # plt.show()
    
    
    L_ANK, R_ANK = 5, 6
    L_KNEE, R_KNEE = 3, 4
    foot_ids = [19,18,17,16,14,15]
    foot_ids = [L_ANK, R_ANK]   # 4 joints for foot contact
    smoothed = {}
    for idx in foot_ids:
        traj = J_zup[:, idx, :]                          # (T,3) for one joint
        # smoothed[idx] = smooth_trans_butter(traj, target_fps, cutoff_hz=4.0, order=2)
        smoothed[idx] = traj
    
    # Stack the smoothed candidate joints into (T, len(foot_ids), 3)
    foot_stack = np.stack([smoothed[idx] for idx in foot_ids], axis=1)  # (T,4,3)
    T = foot_stack.shape[0]
    
    foot_height = foot_stack[..., 2]    # (T,4)
    
    
    # Find per-frame min speed foot
    idx_min_per_frame = np.argmin(foot_height, axis=1)     # (T,)
    min_speed_per_frame = np.min(foot_height, axis=1)    # (T,)
    
    # Height offsets (meters) per foot_id
    toe_height   = 0.025
    heel_height  = 0.15
    
    add_height = heel_height
    
    # Frame 0: place chosen contact at (0,0,0) after subtracting its height
    idx0 = int(idx_min_per_frame[0])
    trans[0] = np.array([0.0, 0.0, foot_stack[0, idx0, 2] - add_height], dtype=np.float32)
    
    # Frames 1..T-1: carry previous frame's contact
    for t in range(1, T):
        idx_prev = int(idx_min_per_frame[t-1])
        foot_movement = foot_stack[t, idx_prev, :] - foot_stack[t-1, idx_prev, :]
        trans[t] = trans[t-1] - foot_movement

    # Optional: smooth translation
    trans = smooth_trans_butter(trans, fps=target_fps, cutoff_hz=2.0, order=2)

    if visualize:
        J_zup = trans.reshape(-1,1,3)   # visualize
        J = J_zup[:, :1,:]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for j in range(J.shape[1]):
            ax.plot(J[:, j, 0], J[:, j, 1], J[:, j, 2], label=str(j))
            ax.text(J[0, j, 0], J[0, j, 1], J[0, j, 2], str(j))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Joint Trajectories')
        plt.legend()
        plt.show()
        
    required_params = {}
    required_params["poses"] = poses
    required_params["trans"] = trans
    required_params["fps"] = target_fps
    
    # save
    np.save(output_path, required_params)
    
    return

def process_VEHS7M_seq(fname, output_path, start_end = None):

    # load raw params from AMASS dataset
    with open(fname, "rb") as f:
        raw_params = pickle.load(f)
        
    
    # dict(np.load(fname, allow_pickle=True))

    poses = raw_params["fullpose"]
    trans = raw_params["trans"]
    
    # clip
    if start_end is not None:
        start = int(start_end[0])
        end = int(start_end[1])
        poses = poses[start:end]
        trans = trans[start:end]
        
    # downsample from 100hz to 25hz
    source_fps = 100
    target_fps = 25
    skip = int(source_fps // target_fps)
    poses = poses[::skip]
    trans = trans[::skip]

    # extract 24 SMPL joints from 55 SMPL-X joints
    joints_to_use = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
    )
    joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
    poses = poses[:, joints_to_use]

    required_params = {}
    required_params["poses"] = poses
    required_params["trans"] = trans
    required_params["fps"] = target_fps
    
    # save
    np.save(output_path, required_params)
    
    return

def process_amass_seq(fname, output_path):

    # load raw params from AMASS dataset
    raw_params = dict(np.load(fname, allow_pickle=True))

    poses = raw_params["poses"]
    trans = raw_params["trans"]

    # downsample from 120hz to 30hz
    source_fps = raw_params["mocap_frame_rate"]
    target_fps = 30
    skip = int(source_fps // target_fps)
    poses = poses[::skip]
    trans = trans[::skip]

    # extract 24 SMPL joints from 55 SMPL-X joints
    joints_to_use = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 40]
    )
    joints_to_use = np.arange(0, 156).reshape((-1, 3))[joints_to_use].reshape(-1)
    poses = poses[:, joints_to_use]

    required_params = {}
    required_params["poses"] = poses
    required_params["trans"] = trans
    required_params["fps"] = target_fps
    
    # save
    np.save(output_path, required_params)
    
    return

    
def project_joints(motion):
    """ This is the original function used by ASE, designed for amp_humanoid.xml """

    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)
    
    # right leg
    right_thigh_pos = motion.global_translation[..., right_thigh_id, :]
    right_shin_pos = motion.global_translation[..., right_shin_id, :]
    right_foot_pos = motion.global_translation[..., right_foot_id, :]
    right_hip_rot = motion.local_rotation[..., right_thigh_id, :]
    right_knee_rot = motion.local_rotation[..., right_shin_id, :]
    
    right_leg_delta0 = right_thigh_pos - right_shin_pos
    right_leg_delta1 = right_foot_pos - right_shin_pos
    right_leg_delta0 = right_leg_delta0 / torch.norm(right_leg_delta0, dim=-1, keepdim=True)
    right_leg_delta1 = right_leg_delta1 / torch.norm(right_leg_delta1, dim=-1, keepdim=True)
    right_knee_dot = torch.sum(-right_leg_delta0 * right_leg_delta1, dim=-1)
    right_knee_dot = torch.clamp(right_knee_dot, -1.0, 1.0)
    right_knee_theta = torch.acos(right_knee_dot)
    right_knee_q = quat_from_angle_axis(torch.abs(right_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    right_knee_local_dir = motion.skeleton_tree.local_translation[right_foot_id]
    right_knee_local_dir = right_knee_local_dir / torch.norm(right_knee_local_dir)
    right_knee_local_dir_tile = torch.tile(right_knee_local_dir.unsqueeze(0), [right_knee_rot.shape[0], 1])
    right_knee_local_dir0 = quat_rotate(right_knee_rot, right_knee_local_dir_tile)
    right_knee_local_dir1 = quat_rotate(right_knee_q, right_knee_local_dir_tile)
    right_leg_dot = torch.sum(right_knee_local_dir0 * right_knee_local_dir1, dim=-1)
    right_leg_dot = torch.clamp(right_leg_dot, -1.0, 1.0)
    right_leg_theta = torch.acos(right_leg_dot)
    right_leg_theta = torch.where(right_knee_local_dir0[..., 1] >= 0, right_leg_theta, -right_leg_theta)
    right_leg_q = quat_from_angle_axis(right_leg_theta, right_knee_local_dir.unsqueeze(0))
    right_hip_rot = quat_mul(right_hip_rot, right_leg_q)
    
    # left leg
    left_thigh_pos = motion.global_translation[..., left_thigh_id, :]
    left_shin_pos = motion.global_translation[..., left_shin_id, :]
    left_foot_pos = motion.global_translation[..., left_foot_id, :]
    left_hip_rot = motion.local_rotation[..., left_thigh_id, :]
    left_knee_rot = motion.local_rotation[..., left_shin_id, :]
    
    left_leg_delta0 = left_thigh_pos - left_shin_pos
    left_leg_delta1 = left_foot_pos - left_shin_pos
    left_leg_delta0 = left_leg_delta0 / torch.norm(left_leg_delta0, dim=-1, keepdim=True)
    left_leg_delta1 = left_leg_delta1 / torch.norm(left_leg_delta1, dim=-1, keepdim=True)
    left_knee_dot = torch.sum(-left_leg_delta0 * left_leg_delta1, dim=-1)
    left_knee_dot = torch.clamp(left_knee_dot, -1.0, 1.0)
    left_knee_theta = torch.acos(left_knee_dot)
    left_knee_q = quat_from_angle_axis(torch.abs(left_knee_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))
    
    left_knee_local_dir = motion.skeleton_tree.local_translation[left_foot_id]
    left_knee_local_dir = left_knee_local_dir / torch.norm(left_knee_local_dir)
    left_knee_local_dir_tile = torch.tile(left_knee_local_dir.unsqueeze(0), [left_knee_rot.shape[0], 1])
    left_knee_local_dir0 = quat_rotate(left_knee_rot, left_knee_local_dir_tile)
    left_knee_local_dir1 = quat_rotate(left_knee_q, left_knee_local_dir_tile)
    left_leg_dot = torch.sum(left_knee_local_dir0 * left_knee_local_dir1, dim=-1)
    left_leg_dot = torch.clamp(left_leg_dot, -1.0, 1.0)
    left_leg_theta = torch.acos(left_leg_dot)
    left_leg_theta = torch.where(left_knee_local_dir0[..., 1] >= 0, left_leg_theta, -left_leg_theta)
    left_leg_q = quat_from_angle_axis(left_leg_theta, left_knee_local_dir.unsqueeze(0))
    left_hip_rot = quat_mul(left_hip_rot, left_leg_q)
    

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., right_thigh_id, :] = right_hip_rot
    new_local_rotation[..., right_shin_id, :] = right_knee_q
    new_local_rotation[..., left_thigh_id, :] = left_hip_rot
    new_local_rotation[..., left_shin_id, :] = left_knee_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)
    
    return new_motion

def project_joints_simple(motion):
    """ This is the our revised function used by TokenHSI, designed for phys_humanoid_v3.xml 

    The difference is that we only project the arms, not the legs.
    The reason is that the leg joints have been modified to 3 DoF spherical joints.

    """

    right_upper_arm_id = motion.skeleton_tree._node_indices["right_upper_arm"]
    right_lower_arm_id = motion.skeleton_tree._node_indices["right_lower_arm"]
    right_hand_id = motion.skeleton_tree._node_indices["right_hand"]
    left_upper_arm_id = motion.skeleton_tree._node_indices["left_upper_arm"]
    left_lower_arm_id = motion.skeleton_tree._node_indices["left_lower_arm"]
    left_hand_id = motion.skeleton_tree._node_indices["left_hand"]
    
    right_thigh_id = motion.skeleton_tree._node_indices["right_thigh"]
    right_shin_id = motion.skeleton_tree._node_indices["right_shin"]
    right_foot_id = motion.skeleton_tree._node_indices["right_foot"]
    left_thigh_id = motion.skeleton_tree._node_indices["left_thigh"]
    left_shin_id = motion.skeleton_tree._node_indices["left_shin"]
    left_foot_id = motion.skeleton_tree._node_indices["left_foot"]
    
    device = motion.global_translation.device

    # right arm
    right_upper_arm_pos = motion.global_translation[..., right_upper_arm_id, :]
    right_lower_arm_pos = motion.global_translation[..., right_lower_arm_id, :]
    right_hand_pos = motion.global_translation[..., right_hand_id, :]
    right_shoulder_rot = motion.local_rotation[..., right_upper_arm_id, :]
    right_elbow_rot = motion.local_rotation[..., right_lower_arm_id, :]
    
    right_arm_delta0 = right_upper_arm_pos - right_lower_arm_pos
    right_arm_delta1 = right_hand_pos - right_lower_arm_pos
    right_arm_delta0 = right_arm_delta0 / torch.norm(right_arm_delta0, dim=-1, keepdim=True)
    right_arm_delta1 = right_arm_delta1 / torch.norm(right_arm_delta1, dim=-1, keepdim=True)
    right_elbow_dot = torch.sum(-right_arm_delta0 * right_arm_delta1, dim=-1)
    right_elbow_dot = torch.clamp(right_elbow_dot, -1.0, 1.0)
    right_elbow_theta = torch.acos(right_elbow_dot)
    right_elbow_q = quat_from_angle_axis(-torch.abs(right_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                            device=device, dtype=torch.float32))
    
    right_elbow_local_dir = motion.skeleton_tree.local_translation[right_hand_id]
    right_elbow_local_dir = right_elbow_local_dir / torch.norm(right_elbow_local_dir)
    right_elbow_local_dir_tile = torch.tile(right_elbow_local_dir.unsqueeze(0), [right_elbow_rot.shape[0], 1])
    right_elbow_local_dir0 = quat_rotate(right_elbow_rot, right_elbow_local_dir_tile)
    right_elbow_local_dir1 = quat_rotate(right_elbow_q, right_elbow_local_dir_tile)
    right_arm_dot = torch.sum(right_elbow_local_dir0 * right_elbow_local_dir1, dim=-1)
    right_arm_dot = torch.clamp(right_arm_dot, -1.0, 1.0)
    right_arm_theta = torch.acos(right_arm_dot)
    right_arm_theta = torch.where(right_elbow_local_dir0[..., 1] <= 0, right_arm_theta, -right_arm_theta)
    right_arm_q = quat_from_angle_axis(right_arm_theta, right_elbow_local_dir.unsqueeze(0))
    right_shoulder_rot = quat_mul(right_shoulder_rot, right_arm_q)
    
    # left arm
    left_upper_arm_pos = motion.global_translation[..., left_upper_arm_id, :]
    left_lower_arm_pos = motion.global_translation[..., left_lower_arm_id, :]
    left_hand_pos = motion.global_translation[..., left_hand_id, :]
    left_shoulder_rot = motion.local_rotation[..., left_upper_arm_id, :]
    left_elbow_rot = motion.local_rotation[..., left_lower_arm_id, :]
    
    left_arm_delta0 = left_upper_arm_pos - left_lower_arm_pos
    left_arm_delta1 = left_hand_pos - left_lower_arm_pos
    left_arm_delta0 = left_arm_delta0 / torch.norm(left_arm_delta0, dim=-1, keepdim=True)
    left_arm_delta1 = left_arm_delta1 / torch.norm(left_arm_delta1, dim=-1, keepdim=True)
    left_elbow_dot = torch.sum(-left_arm_delta0 * left_arm_delta1, dim=-1)
    left_elbow_dot = torch.clamp(left_elbow_dot, -1.0, 1.0)
    left_elbow_theta = torch.acos(left_elbow_dot)
    left_elbow_q = quat_from_angle_axis(-torch.abs(left_elbow_theta), torch.tensor(np.array([[0.0, 1.0, 0.0]]), 
                                        device=device, dtype=torch.float32))

    left_elbow_local_dir = motion.skeleton_tree.local_translation[left_hand_id]
    left_elbow_local_dir = left_elbow_local_dir / torch.norm(left_elbow_local_dir)
    left_elbow_local_dir_tile = torch.tile(left_elbow_local_dir.unsqueeze(0), [left_elbow_rot.shape[0], 1])
    left_elbow_local_dir0 = quat_rotate(left_elbow_rot, left_elbow_local_dir_tile)
    left_elbow_local_dir1 = quat_rotate(left_elbow_q, left_elbow_local_dir_tile)
    left_arm_dot = torch.sum(left_elbow_local_dir0 * left_elbow_local_dir1, dim=-1)
    left_arm_dot = torch.clamp(left_arm_dot, -1.0, 1.0)
    left_arm_theta = torch.acos(left_arm_dot)
    left_arm_theta = torch.where(left_elbow_local_dir0[..., 1] <= 0, left_arm_theta, -left_arm_theta)
    left_arm_q = quat_from_angle_axis(left_arm_theta, left_elbow_local_dir.unsqueeze(0))
    left_shoulder_rot = quat_mul(left_shoulder_rot, left_arm_q)

    new_local_rotation = motion.local_rotation.clone()
    new_local_rotation[..., right_upper_arm_id, :] = right_shoulder_rot
    new_local_rotation[..., right_lower_arm_id, :] = right_elbow_q
    new_local_rotation[..., left_upper_arm_id, :] = left_shoulder_rot
    new_local_rotation[..., left_lower_arm_id, :] = left_elbow_q
    
    new_local_rotation[..., left_hand_id, :] = quat_identity([1])
    new_local_rotation[..., right_hand_id, :] = quat_identity([1])

    new_sk_state = SkeletonState.from_rotation_and_root_translation(motion.skeleton_tree, new_local_rotation, motion.root_translation, is_local=True)
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion


# wen helper for smplest output
def axis_angle_to_matrix_batch(aa):  # (T,3)
    theta = np.linalg.norm(aa, axis=1, keepdims=True) + 1e-12
    k = aa / theta
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    z = np.zeros_like(kx)
    K = np.stack([
        np.stack([z,   -kz,  ky], axis=1),
        np.stack([kz,   z,  -kx], axis=1),
        np.stack([-ky,  kx,  z],  axis=1),
    ], axis=1)                               # (T,3,3)
    I = np.eye(3)[None]
    s = np.sin(theta)[:, None]
    c = np.cos(theta)[:, None]
    return I + s*K + (1 - c)*(K @ K)         # (T,3,3)

def matrix_to_axis_angle_batch(R):           # (T,3,3) -> (T,3)
    tr = np.clip(R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2], -1.0, 3.0)
    cos_t = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_t)
    rx = R[:, 2, 1] - R[:, 1, 2]
    ry = R[:, 0, 2] - R[:, 2, 0]
    rz = R[:, 1, 0] - R[:, 0, 1]
    axis = np.stack([rx, ry, rz], axis=1)
    out = np.zeros_like(axis)
    big = theta >= 1e-6
    if np.any(big):
        out[big] = axis[big] / (2.0*np.sin(theta[big]))[:, None] * theta[big][:, None]
    if np.any(~big):
        out[~big] = axis[~big] * 0.5  # series approx
    return out

def rot_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]], dtype=np.float32)
    
from scipy.signal import butter, filtfilt

def butter_lowpass_filtfilt(x, cutoff_hz, fps, order=2):
    # x: (T,) 1D
    nyq = 0.5 * fps
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, x, method="gust", padlen=min(31, len(x)-1))

def smooth_trans_butter(trans, fps, cutoff_hz=4.0, order=2):
    out = np.zeros_like(trans)
    for d in range(3):
        out[:, d] = butter_lowpass_filtfilt(trans[:, d], cutoff_hz, fps, order)
    return out

def smooth_axis_angle_butter(aa, fps, cutoff_hz=4.0, order=2):
    out = np.zeros_like(aa)
    for d in range(3):
        out[:, d] = butter_lowpass_filtfilt(aa[:, d], cutoff_hz, fps, order)
    return out


# smooth root axis angle

import numpy as np
from scipy.signal import butter, filtfilt

# ---------- SO(3) helpers ----------
def aa_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12: return np.eye(3)
    k = v / th
    K = np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]], float)
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*(K@K)

def R_to_aa(R):
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-12: return np.zeros(3)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2.0*np.sin(th))
    return v * th

def so3_log(R):          # mat -> axis-angle
    return R_to_aa(R)

def so3_exp(v):          # axis-angle -> mat
    return aa_to_R(v)

def Rx(angle):
    c,s = np.cos(angle), np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], float)

# ---------- filtering ----------
def butter_lowpass_filtfilt_vec(x, cutoff_hz, fps, order=2):
    nyq = 0.5 * fps
    b, a = butter(order, cutoff_hz/nyq, btype='low')
    y = np.empty_like(x)
    for d in range(x.shape[1]):
        y[:, d] = filtfilt(b, a, x[:, d], method="gust", padlen=min(31, len(x)-1))
    return y

# ---------- SLERP for rotations (via squad-lite on edges) ----------
def slerp_R(R0, R1, u):
    # SLERP using exp/log: R(u) = R0 * exp( u * log(R0^T R1) )
    dR = R0.T @ R1
    v  = so3_log(dR)       # axis-angle
    return R0 @ so3_exp(u * v)

# ---------- jerk detection (on relative angles) ----------
def detect_jerks(R, thresh_rad=0.6, mad_k=4.0):
    """
    Returns boolean mask of frames that look jerky based on |log(dR)|.
    thresh_rad: absolute threshold in rad/frame (e.g., 0.5–1.0)
    mad_k: robust MAD multiplier on top (catches context-dependent spikes)
    """
    T = R.shape[0]
    theta = np.zeros(T)
    theta[0] = 0.0
    for t in range(1, T):
        dR = R[t-1].T @ R[t]
        theta[t] = np.linalg.norm(so3_log(dR))

    # robust threshold
    med = np.median(theta)
    mad = np.median(np.abs(theta - med)) + 1e-9
    rob_thresh = med + mad_k * 1.4826 * mad

    mask = theta > max(thresh_rad, rob_thresh)
    # we mark the *current* frame as jerky if its increment is too large
    return mask, theta

# ---------- repair jerks by inpainting with SLERP ----------
def repair_rot_jerks(R, jerk_mask, radius=2):
    """
    Replace jerky frames by slerp between nearest clean neighbors.
    radius: how far to search for clean neighbors on each side.
    """
    T = R.shape[0]
    R_fixed = R.copy()
    idx = np.where(jerk_mask)[0]
    if len(idx) == 0:
        return R_fixed

    clean = np.ones(T, dtype=bool)
    clean[idx] = False

    for t in idx:
        # find left clean
        l = t-1
        while l >= max(0, t-10*radius) and not clean[l]:
            l -= 1
        # find right clean
        r = t+1
        while r <= min(T-1, t+10*radius) and not clean[r]:
            r += 1
        if l < 0 or r >= T or not(clean[l] and clean[r]):
            # cannot repair cleanly -> leave as is
            continue
        # interpolate proportionally in time
        u = (t - l) / max(1e-6, (r - l))
        R_fixed[t] = slerp_R(R_fixed[l], R_fixed[r], u)

    return R_fixed

# ---------- main smoother with jerk handling ----------
def smooth_root_with_jerk_fix(aa_root, fps, cutoff_hz=1.0, order=2,
                              apply_rx=False, world_pre=True,
                              jerk_abs_thresh=0.6, jerk_mad_k=4.0,
                              search_radius=2, return_format='aa'):
    """
    1) Build R from AA
    2) Detect jerks on increments
    3) Repair jerks by SLERP inpainting
    4) Low-pass the *increments* and reintegrate
    5) Optional Rx for Y-up -> Z-up
    6) Return AA / R / quat
    """
    T = aa_root.shape[0]
    R = np.stack([aa_to_R(aa_root[t]) for t in range(T)], axis=0)  # raw

    # (a) jerk detection
    mask, theta = detect_jerks(R, thresh_rad=jerk_abs_thresh, mad_k=jerk_mad_k)

    # (b) repair by SLERP between nearest clean neighbors
    R_rep = repair_rot_jerks(R, mask, radius=search_radius)

    # (c) build relative increments from repaired R
    omega = np.zeros((T,3))
    for t in range(1, T):
        dR = R_rep[t-1].T @ R_rep[t]
        omega[t] = so3_log(dR)

    # (d) low-pass the increments (zero-phase)
    omega_f = butter_lowpass_filtfilt_vec(omega, cutoff_hz, fps, order)

    # (e) re-integrate
    A = np.zeros_like(R_rep)
    A[0] = R_rep[0]
    for t in range(1, T):
        A[t] = A[t-1] @ so3_exp(omega_f[t])

    # (f) optional Y-up -> Z-up
    if apply_rx:
        RX = Rx(-np.pi/2)
        A = (RX[None, ...] @ A) if world_pre else (A @ RX[None, ...])

    if return_format == 'R':
        return A.astype(np.float32), mask, theta
    elif return_format == 'quat':
        out = []
        for t in range(T):
            Rt = A[t]
            tr = np.trace(Rt)
            if tr > 0:
                s = np.sqrt(tr + 1.0) * 2
                w = 0.25 * s
                x = (Rt[2,1]-Rt[1,2]) / s
                y = (Rt[0,2]-Rt[2,0]) / s
                z = (Rt[1,0]-Rt[0,1]) / s
            else:
                i = np.argmax([Rt[0,0], Rt[1,1], Rt[2,2]])
                if i == 0:
                    s = np.sqrt(1.0 + Rt[0,0] - Rt[1,1] - Rt[2,2]) * 2
                    w = (Rt[2,1]-Rt[1,2]) / s; x = 0.25*s
                    y = (Rt[0,1]+Rt[1,0]) / s; z = (Rt[0,2]+Rt[2,0]) / s
                elif i == 1:
                    s = np.sqrt(1.0 + Rt[1,1] - Rt[0,0] - Rt[2,2]) * 2
                    w = (Rt[0,2]-Rt[2,0]) / s; x = (Rt[0,1]+Rt[1,0]) / s
                    y = 0.25*s;             z = (Rt[1,2]+Rt[2,1]) / s
                else:
                    s = np.sqrt(1.0 + Rt[2,2] - Rt[0,0] - Rt[1,1]) * 2
                    w = (Rt[1,0]-Rt[0,1]) / s; x = (Rt[0,2]+Rt[2,0]) / s
                    y = (Rt[1,2]+Rt[2,1]) / s; z = 0.25*s
            q = np.array([w,x,y,z]); q /= (np.linalg.norm(q)+1e-12)
            out.append(q)
        return np.stack(out, 0).astype(np.float32), mask, theta
    else:
        aa_out = np.stack([R_to_aa(A[t]) for t in range(T)], axis=0)
        return aa_out.astype(np.float32), mask, theta

