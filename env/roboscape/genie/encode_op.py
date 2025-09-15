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


# ✅ 加载 tokenizer 模型并移动到指定 GPU
def load_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    tokenizer = VQModel(
        VQConfig(),
        ckpt_path="/iag_ad_01/ad/zhangxin11/tyz/observation-reward-model/genie/magvit2.ckpt",
    ).cuda()
    tokenizer.eval()
    print(f"✅ 模型已加载到 GPU {gpu_id}")
    return tokenizer


# ✅ 读取视频路径并按 `item_range` 选择数据
def load_data(train_dir):
    data = [os.path.join(train_dir, video_dir) for video_dir in os.listdir(train_dir)]
    return data


def output_video(
    video_frames,
    output_path="/iag_ad_01/ad/zhangxin11/tyz/observation-genie/temp/temp.mp4",
    fps=30,
    frame_width=640,
    frame_height=480,
):
    # 定义视频编码格式和创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # 遍历每一帧并写入视频
    for i in range(len(video_frames)):
        frame = video_frames[i]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # 确保帧是 uint8 类型
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # 写入帧到视频
        out.write(frame)

    # 释放视频写入器
    out.release()


# ✅ 定义 VideoDataset 类
class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.resize_transform = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        data = np.load(video_path, allow_pickle=True)["arr_0"].tolist()
        temp_path = (
            f"/iag_ad_01/ad/zhangxin11/tyz/observation-genie/temp/temp_{idx}.mp4"
        )
        output_video(data["observation"]["rgb"], output_path=temp_path)
        # 加载视频帧并进行 Resize 处理
        vr = VideoReader(
            temp_path,
            ctx=cpu(0),
            num_threads=4,
        )
        frames = vr.get_batch(range(len(vr))).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) / 255.0  # [N, 3, H, W]
        resized_frames = torch.stack([self.resize_transform(frame) for frame in frames])

        actions = data["action"]
        action_all = np.concatenate(
            [
                actions["end"]["position"],
                actions["end"]["orientation"],
                actions["effector"]["position_gripper"].reshape(
                    actions["effector"]["position_gripper"].shape + (1,)
                ),
            ],
            axis=2,
        )  # N, 1, 8

        return (
            resized_frames,
            action_all,
            int(os.path.basename(video_path).split(".")[0].split("_")[-1]),
        )


# ✅ 计算当前 GPU 处理的数据总帧数
def calculate_total_frames(data):
    total_frame = 0
    for item in tqdm(data):
        total_frame += (
            np.load(item, allow_pickle=True)["arr_0"]
            .tolist()["observation"]["rgb"]
            .shape[0]
        )
    print(f"✅ 计算总帧数: {total_frame}")
    return total_frame


# ✅ 创建 memmap 文件路径，根据 `item_range` 创建不同的文件
def create_memmap_files(output_dir, total_frame):
    """
    video_data_path = f"{output_dir}/video_item{item_range[0]}_{item_range[1]}.bin"
    segment_data_path = f"{output_dir}/segment_ids_item{item_range[0]}_{item_range[1]}.bin"
    action_data_path = f"{output_dir}/action_item{item_range[0]}_{item_range[1]}.bin"
    """
    os.makedirs(output_dir, exist_ok=True)
    video_data_path = f"{output_dir}/video.bin"
    segment_data_path = f"{output_dir}/segment_ids.bin"
    action_data_path = f"{output_dir}/action.bin"
    video_data = np.memmap(
        video_data_path, dtype=np.uint32, mode="w+", shape=(total_frame, 16, 16)
    )
    segment_data = np.memmap(
        segment_data_path, dtype=np.int32, mode="w+", shape=(total_frame, 1)
    )
    action_data = np.memmap(
        action_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 8)
    )
    print(
        f"✅ 创建 memmap 文件：\n  - {video_data_path}\n  - {segment_data_path}\n  - {action_data_path}\n"
    )

    # Dictionary with the specified data
    metadata = {
        "token_dtype": "uint32",
        "s": 16,
        "h": 16,
        "w": 16,
        "vocab_size": 262144,
        "hz": 30,
        "tokenizer_ckpt": "data/magvit2.ckpt",
        "num_images": total_frame,
    }

    # Path to save the JSON file
    file_path = f"{output_dir}/metadata.json"

    # Writing the dictionary to a JSON file
    with open(file_path, "w") as file:
        json.dump(metadata, file)

    return video_data, segment_data, action_data


# ✅ 处理视频数据并编码
def process_videos(data_loader, tokenizer, video_data, segment_data, action_data):
    frame_idx = 0
    batch_size = 32

    for resized_frames, clip_actions, video_id in tqdm(data_loader):
        resized_frames = resized_frames[0].cuda(non_blocking=True)
        clip_actions = clip_actions[0].numpy()

        # 分批次处理帧，防止 GPU 内存溢出
        num_batches = (resized_frames.size(0) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start, end = i * batch_size, min(
                (i + 1) * batch_size, resized_frames.size(0)
            )
            batch_frames = resized_frames[start:end].cuda(non_blocking=True)
            batch_actions = clip_actions[start:end]
            # FP16 加速推理
            with torch.no_grad():
                quant, _, _, _ = tokenizer.encode(batch_frames * 2 - 1)
                token_ids = (
                    tokenizer.quantize.bits_to_indices(quant.permute(0, 2, 3, 1) > 0)
                    .cpu()
                    .numpy()
                )

            # ✅ 写入 memmap 文件
            video_data[frame_idx : frame_idx + batch_frames.size(0)] = token_ids
            action_data[frame_idx : frame_idx + batch_frames.size(0)] = (
                batch_actions.reshape(-1, 8)
            )
            segment_data[frame_idx : frame_idx + batch_frames.size(0)] = video_id
            # 更新帧索引
            frame_idx += batch_frames.size(0)
            # 释放内存
            del quant, token_ids, batch_frames, batch_actions
            torch.cuda.empty_cache()
            gc.collect()

    print(f"✅ 处理完成，总帧数: {frame_idx}")


# ✅ 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="Video Encoding with MagViT on Multiple GPUs"
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="/iag_ad_01/ad/zhangxin11/tyz/observation-reward-model/data/sample/tomato_random/data",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/iag_ad_01/ad/zhangxin11/tyz/observation-reward-model/data/genie_data_debug",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    return parser.parse_args()


# ✅ 主函数：协调各个模块
def main():
    # 解析参数
    args = parse_args()
    gpu_id = args.gpu_id

    # 加载模型和数据
    tokenizer = load_model(gpu_id)
    data = load_data(args.source_path)  # npz paths
    train_data = data[: int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)) :]

    # 加载 Dataset 和 DataLoader
    train_video_dataset = VideoDataset(train_data)
    val_video_dataset = VideoDataset(val_data)

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
    train_video_data, train_segment_data, train_action_data = create_memmap_files(
        os.path.join(output_dir, "train"), train_total_frame
    )
    val_video_data, val_segment_data, val_action_data = create_memmap_files(
        os.path.join(output_dir, "val"), val_total_frame
    )

    # 处理视频数据
    process_videos(
        train_data_loader,
        tokenizer,
        train_video_data,
        train_segment_data,
        train_action_data,
    )
    process_videos(
        val_data_loader,
        tokenizer,
        val_video_data,
        val_segment_data,
        val_action_data,
    )

    print(
        f"🎉 任务完成！GPU {gpu_id} 处理 {len(data)} 个视频，数据已保存到 {output_dir}"
    )


# ✅ 启动主程序
if __name__ == "__main__":
    main()
