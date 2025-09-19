import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from magvit2.models.lfqgan import VQModel
from magvit2.config import VQConfig
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import gc
import torchvision.io as io
import cv2
import json
from PIL import Image
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.transform import Rotation as R


# ✅ 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Encoding with MagViT on Multiple GPUs"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/manifold-obs/bingwen/Datasets/wooden/bowl/TabletopPickPlaceEnv-v1",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/ML-vePFS/tangyinzhou/encoded_maniskill_delta_multi_view_temp",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--task", type=str, default="20250907_154533", help="each part is 1w"
    )
    parser.add_argument("--sigma", type=float, default=-1, help="each part is 1w")
    parser.add_argument("--multiview", action="store_true")
    return parser.parse_args()


# ✅ 加载 tokenizer 模型并移动到指定 GPU
def load_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    tokenizer = VQModel(
        VQConfig(),
        ckpt_path="/ML-vePFS/tangyinzhou/observation-genie/genie/magvit2.ckpt",
    ).cuda()
    tokenizer.eval()
    print(f"✅ 模型已加载到 GPU {gpu_id}")
    return tokenizer


# ✅ 加载 tokenizer 模型并移动到指定 GPU
def load_text_model(gpu_id):
    from transformers import T5Tokenizer, T5Model

    torch.cuda.set_device(gpu_id)
    tokenizer = None
    model = None
    print(f"✅ text模型已加载到 GPU {gpu_id}")
    return tokenizer, model


# ✅ 读取视频路径并按 `item_range` 选择数据
def load_data(train_dir, tid):
    data = []
    for task in os.listdir(os.path.join(train_dir, tid)):
        data.extend(
            [
                os.path.join(train_dir, tid, task, "success", x)
                for x in os.listdir(os.path.join(train_dir, tid, task, "success"))
            ]
        )
    return data


def exp_grow(N, rate=4):
    """返回长度为 N 的软 delta，峰值在最后一个元素处为 1"""
    k = np.arange(N)  # 0, 1, ..., n-1
    seq = np.exp(rate * (k / (N - 1) - 1))  # 最后一个元素为 exp(0)=1
    return seq / max(seq)


def get_dones(pos, grip):
    dones = np.zeros((pos.shape[0],))
    t_high = (pos[:, -1].reshape(-1) * np.abs(grip - 1).reshape(-1)).argmax()
    pick = dones[:t_high]
    place = dones[t_high:]
    pick_delta = exp_grow(len(pick))
    place_delta = exp_grow(len(place))
    dones = np.concatenate([pick_delta, place_delta + 10])
    return dones


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


def eight_dim_to_ten_dim_delta(state_all, action_all, chunk_size=60):
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
    delta_actions = []
    for i in range(length):
        state_at_obs = state_all[i]
        end = min(i + chunk_size, length)
        current_action = action_all[i:end][:, :7]
        action_gripper = action_all[i:end][:, 7:8]
        delta_action_44 = []
        state_44 = get_44(state_at_obs)
        for j in range(end - i):
            action_44 = get_44(current_action[j])
            delta_44 = np.linalg.inv(state_44) @ action_44
            delta_action_44.append(delta_44)  # list of (4,4)
        delta_action_44 = np.stack(delta_action_44)  # (60, 4, 4)
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
        # 如果不够chunk_size，则用-10补齐
        if delta_action_10.shape[0] < chunk_size:
            delta_action_10 = np.pad(
                delta_action_10,
                ((0, chunk_size - delta_action_10.shape[0]), (0, 0)),
                "constant",
                constant_values=-10,
            )
        delta_actions.append(delta_action_10)
    delta_actions = np.stack(delta_actions, axis=0)  # (B, 60, 10)
    return delta_actions


# ✅ 定义 VideoDataset 类
class VideoDataset(Dataset):
    def __init__(self, video_paths, sigma, multiview=False):
        self.video_paths = video_paths
        self.resize_transform = transforms.Resize((256, 256))
        self.sigma = sigma
        self.multiview = multiview

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
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
        delta_actions = eight_dim_to_ten_dim_delta(
            state_all=state_all, action_all=action_all
        )
        return (
            delta_actions,
            idx,
        )


def get_frame_count(video_path):
    """获取单个视频的帧数"""
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


# ✅ 计算当前 GPU 处理的数据总帧数
def calculate_total_frames(data, max_workers=8):
    video_paths = [
        item.replace(".npz", ".mp4").replace("/success/", "/videos/") for item in data
    ]
    total_frame = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(get_frame_count, path): path for path in video_paths
        }
        for future in tqdm(
            as_completed(future_to_path), total=len(video_paths), desc="统计帧数"
        ):
            total_frame += future.result() - 1

    print(f"✅ 计算总帧数: {total_frame}")
    return total_frame


# ✅ 创建 memmap 文件路径，根据 `item_range` 创建不同的文件
def create_memmap_files(output_dir, total_frame, is_multiview=False):
    """
    video_data_path = f"{output_dir}/video_item{item_range[0]}_{item_range[1]}.bin"
    segment_data_path = f"{output_dir}/segment_ids_item{item_range[0]}_{item_range[1]}.bin"
    action_data_path = f"{output_dir}/action_item{item_range[0]}_{item_range[1]}.bin"
    """
    os.makedirs(output_dir, exist_ok=True)
    action_data_path = f"{output_dir}/delta_action_10dim.bin"
    action_data = np.memmap(
        action_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 60, 10)
    )

    print(f"✅ 创建 memmap 文件：\n {action_data_path}")
    return action_data


def get_text_embedding(text_tokenizer, text_model, clip_text):
    # 使用分词器对文本进行编码
    inputs = text_tokenizer.encode_plus(
        clip_text,
        add_special_tokens=True,  # 添加特殊标记
        max_length=512,  # 设置最大长度
        padding="max_length",  # 填充到最大长度
        truncation=True,  # 截断过长的文本
        return_attention_mask=True,  # 返回注意力掩码
        return_tensors="pt",  # 返回PyTorch张量
    )

    # 获取输入ID和注意力掩码
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 使用模型的编码器部分进行编码
    with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
        outputs = text_model.encoder(input_ids=input_ids, attention_mask=attention_mask)

    # 获取最后一层的隐藏状态
    last_hidden_state = outputs.last_hidden_state

    text_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]
    return text_embedding


# ✅ 处理视频数据并编码
def process_videos(
    data_loader,
    tokenizer,
    text_tokenizer,
    text_model,
    action_data,
    output_dir,
):
    frame_idx = 0
    batch_size = 128

    for index, (clip_actions, video_id) in enumerate(data_loader):
        print(f"{index}/{len(data_loader)}")
        clip_actions = clip_actions[0].numpy()
        num_batches = (clip_actions.shape[0] + batch_size - 1) // batch_size
        for i in range(num_batches):
            start, end = i * batch_size, min(
                (i + 1) * batch_size, clip_actions.shape[0]
            )
            batch_actions = clip_actions[start:end]
            action_data[frame_idx : frame_idx + batch_actions.shape[0]] = (
                batch_actions.reshape(-1, 60, 10)
            )
            frame_idx += batch_actions.shape[0]
    print(f"✅ 处理完成，总帧数: {frame_idx}")


# ✅ 主函数：协调各个模块
def main():
    # 解析参数
    args = parse_args()
    args.multiview = True
    gpu_id = args.gpu_id

    # 加载模型和数据
    tokenizer = load_model(gpu_id)
    text_tokenizer, text_encoder = load_text_model(gpu_id)
    data = load_data(args.source_path, args.task)
    train_data = data[: int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)) :]

    # 加载 Dataset 和 DataLoader
    train_video_dataset = VideoDataset(train_data, args.sigma, args.multiview)
    val_video_dataset = VideoDataset(val_data, args.sigma, args.multiview)

    train_data_loader = DataLoader(
        train_video_dataset, batch_size=1, num_workers=4, pin_memory=True
    )
    val_data_loader = DataLoader(
        val_video_dataset, batch_size=1, num_workers=4, pin_memory=True
    )

    # 计算总帧数
    train_total_frame = calculate_total_frames(train_data)
    val_total_frame = calculate_total_frames(val_data)

    # 创建 memmap 文件
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    train_action_data = create_memmap_files(
        os.path.join(output_dir, f"{args.task}", "train"),
        train_total_frame,
        args.multiview,
    )

    val_action_data = create_memmap_files(
        os.path.join(output_dir, f"{args.task}", "val"),
        val_total_frame,
        args.multiview,
    )

    # 处理视频数据
    process_videos(
        train_data_loader,
        tokenizer,
        text_tokenizer,
        text_encoder,
        train_action_data,
        os.path.join(output_dir, f"{args.task}", "train"),
    )
    process_videos(
        val_data_loader,
        tokenizer,
        text_tokenizer,
        text_encoder,
        val_action_data,
        os.path.join(output_dir, f"{args.task}", "val"),
    )

    print(
        f"🎉 任务完成！GPU {gpu_id} 处理 {len(data)} 个视频，数据已保存到 {output_dir}"
    )


# ✅ 启动主程序
if __name__ == "__main__":
    main()
