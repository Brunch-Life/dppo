import numpy as np
from scipy.spatial.transform import Rotation as R


def get_pose_from_rot_pos(mat: np.ndarray, pos: np.ndarray):
    return np.concatenate(
        [
            np.concatenate([mat, pos.reshape(3, 1)], axis=-1),
            np.array([0.0, 0.0, 0.0, 1.0]).reshape(1, 4),
        ],
        axis=0,
    )


def quat_to_matrix(quaternions):
    """将四元数转换为旋转矩阵 (w, x, y, z) 或 (x, y, z, w)"""
    batch_size = quaternions.shape[0]
    rot_mats = np.zeros((batch_size, 3, 3))
    for i in range(batch_size):
        # 假设四元数格式为 (x, y, z, w)，如果是 (w, x, y, z) 请修改
        rot = R.from_quat(quaternions[i])
        rot_mats[i] = rot.as_matrix()
    return rot_mats


def get_44(state):
    """从状态中获取4x4变换矩阵"""
    rot = R.from_quat(state[3:7])
    pos = state[:3]
    return get_pose_from_rot_pos(rot.as_matrix(), pos)  # (4, 4)


def eight_dim_to_ten_dim_delta(state_all, action_all, chunk_size=20):
    """
    将8维状态和8维动作转换为10维delta action

    参数:
        current_state: 8维状态 [B, 8]
            结构: [x, y, z, qx, qy, qz, qw, gripper]
        target_action: 8维目标动作 [B, 8]
            结构: [x, y, z, qx, qy, qz, qw, gripper]

    返回:
        delta_action: 10维delta action [B, 10]
    """
    length = state_all.shape[0]

    end_idx = length - chunk_size

    for i in range(end_idx):
        state_at_obs = state_all[i]
        current_action = action_all[i : i + chunk_size][:, :7]
        action_gripper = action_all[i : i + chunk_size][:, 7:8]
        delta_action_44 = []
        state_44 = get_44(state_at_obs)
        for j in range(20):
            action_44 = get_44(current_action[j])
            delta_44 = np.linalg.inv(state_44) @ action_44
            delta_action_44.append(delta_44)  # list of (4,4)
        delta_action_44 = np.stack(delta_action_44)  # (20, 4, 4)
        mat_6 = delta_action_44[:, :3, :2].reshape(delta_action_44.shape[0], 6)
        delta_pos = delta_action_44[:, :3, 3].reshape(delta_action_44.shape[0], 3)
        delta_action_10 = np.concatenate(
            [
                delta_pos,
                mat_6,
                action_gripper,
            ],
            axis=-1,
        )

        # save current_obs, delta_action_10, predict_image
        # self.dataset.append(
        #     {
        #         "current_obs": current_obs,
        #         "delta_action_10": delta_action_10,
        #         "predict_image": predict_image,
        #     }
        # )

    return delta_action


video_path = "/manifold-obs/bingwen/Datasets/wooden/bowl/TabletopPickPlaceEnv-v1/20250907_154533/lemon_bowl_wooden/success/success_optimal_lemon_bowl_wooden_proc_3_num_9_trynum_10_epsid_38.npz"
data = np.load(video_path, allow_pickle=True)["arr_0"].tolist()
actions = data["action"]
states = data["state"]

action_all = np.concatenate(
    [
        actions["end"]["position"],  # (..., 3)
        actions["end"]["orientation"],  # (..., 4)
        actions["effector"]["position_gripper"].reshape(
            actions["effector"]["position_gripper"].shape + (1,)
        ),
    ],
    axis=-1,
).reshape(
    -1, 8
)  # (n_steps, 8)
state_all = np.concatenate(
    [
        states["end"]["position"],  # (..., 3)
        states["end"]["orientation"],  # (..., 4)
        states["effector"]["position_gripper"].reshape(
            states["effector"]["position_gripper"].shape + (1,)
        ),
    ],
    axis=-1,
).reshape(
    -1, 8
)  # (n_steps, 8)

delta_actions = eight_dim_to_ten_dim_delta(state_all=state_all, action_all=action_all)
print(delta_actions.shape)
