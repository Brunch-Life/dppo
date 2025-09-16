import numpy as np
from scipy.spatial.transform import Rotation as R


def get_pose_from_rot_pos_batch(rot, pos):
    """从旋转矩阵和位置向量构建变换矩阵"""
    batch_size = rot.shape[0]
    pose_mat = np.zeros((batch_size, 4, 4))
    pose_mat[:, :3, :3] = rot
    pose_mat[:, :3, 3] = pos
    pose_mat[:, 3, 3] = 1.0
    return pose_mat


def quat_to_matrix(quaternions):
    """将四元数转换为旋转矩阵 (w, x, y, z) 或 (x, y, z, w)"""
    batch_size = quaternions.shape[0]
    rot_mats = np.zeros((batch_size, 3, 3))
    for i in range(batch_size):
        # 假设四元数格式为 (x, y, z, w)，如果是 (w, x, y, z) 请修改
        rot = R.from_quat(quaternions[i])
        rot_mats[i] = rot.as_matrix()
    return rot_mats


def eight_dim_to_ten_dim_delta(current_state, target_action):
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
    B = current_state.shape[0]

    # 解析当前状态
    current_pos = current_state[:, :3]  # [B, 3] 位置
    current_quat = current_state[:, 3:7]  # [B, 4] 四元数姿态
    current_gripper = current_state[:, 7:8]  # [B, 1] 夹爪状态

    # 解析目标动作
    target_pos = target_action[:, :3]  # [B, 3] 目标位置
    target_quat = target_action[:, 3:7]  # [B, 4] 目标姿态
    target_gripper = target_action[:, 7:8]  # [B, 1] 目标夹爪状态

    # 计算位置delta
    delta_pos = target_pos - current_pos  # [B, 3]

    # 将四元数转换为旋转矩阵
    current_rot = quat_to_matrix(current_quat)  # [B, 3, 3]
    target_rot = quat_to_matrix(target_quat)  # [B, 3, 3]

    # 计算相对旋转 (目标旋转相对于当前旋转)
    # 相对旋转 = 目标旋转 * 当前旋转的逆
    delta_rot = np.matmul(target_rot, np.transpose(current_rot, (0, 2, 1)))  # [B, 3, 3]

    # 提取旋转矩阵的前两列
    rot_col0 = delta_rot[:, :, 0]  # [B, 3]
    rot_col1 = delta_rot[:, :, 1]  # [B, 3]

    # 计算夹爪delta
    delta_gripper = target_gripper - current_gripper  # [B, 1]

    # 组合成10维delta action
    delta_action = np.concatenate(
        [
            delta_pos,  # 3维位置delta
            rot_col0,  # 3维旋转第一列
            rot_col1,  # 3维旋转第二列
            delta_gripper,  # 1维夹爪delta
        ],
        axis=1,
    )  # [B, 10]

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
).reshape(-1, 8)
state_all = np.concatenate(
    [
        states["end"]["position"],  # (..., 3)
        states["end"]["orientation"],  # (..., 4)
        states["effector"]["position_gripper"].reshape(
            states["effector"]["position_gripper"].shape + (1,)
        ),
    ],
    axis=-1,
).reshape(-1, 8)
delta_actions = eight_dim_to_ten_dim_delta(
    current_state=state_all, target_action=action_all
)
print(delta_actions.shape)
