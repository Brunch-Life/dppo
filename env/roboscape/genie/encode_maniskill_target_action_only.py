import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from magvit2.models.lfqgan import VQModel
from magvit2.config import VQConfig
from torchvision import transforms
from decord import VideoReader, cpu
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
    tokenizer = T5Tokenizer.from_pretrained(
        "/manifold-obs/tangyinzhou/hf_weights/t5-base"
    )
    model = model = T5Model.from_pretrained(
        "/manifold-obs/tangyinzhou/hf_weights/t5-base"
    )
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
        quat = actions["end"]["orientation"]  # shape: (..., 4)
        # 转成欧拉角 (roll, pitch, yaw)，默认是 'xyz' 顺序
        euler = np.stack(
            [R.from_quat(x).as_euler("xyz") for x in quat]
        )  # shape: (..., 3)

        # 然后你可以替换原来的四元数部分
        action_all = np.concatenate(
            [
                actions["end"]["position"],  # (..., 3)
                euler,  # (..., 3)
                actions["effector"]["position_gripper"].reshape(
                    actions["effector"]["position_gripper"].shape + (1,)
                ),
            ],
            axis=-1,
        )  # 最终 shape: (..., 7)
        return (
            action_all,
            idx,
        )


def get_frame_count_ffprobe(video_path):
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=nb_read_frames",
        "-of",
        "csv=p=0",
        video_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        return int(result.stdout.strip())
    except ValueError:
        raise ValueError(f"无法读取帧数: {video_path}")


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
    action_data_path = f"{output_dir}/target_pose.bin"
    action_data = np.memmap(
        action_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 7)
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
                batch_actions.reshape(-1, 7)
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
