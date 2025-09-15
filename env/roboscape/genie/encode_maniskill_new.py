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
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed


# ✅ 加载 tokenizer 模型并移动到指定 GPU
def load_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    tokenizer = VQModel(
        VQConfig(),
        ckpt_path="/iag_ad_01/ad/tangyinzhou/tyz/observation-genie-finetune/genie/magvit2.ckpt",
    ).cuda()
    tokenizer.eval()
    print(f"✅ 模型已加载到 GPU {gpu_id}")
    return tokenizer


# ✅ 加载 tokenizer 模型并移动到指定 GPU
def load_text_model(gpu_id):
    from transformers import T5Tokenizer, T5Model

    torch.cuda.set_device(gpu_id)
    tokenizer = T5Tokenizer.from_pretrained(
        "/tangyinzhou-tos-volc-engine/tyz/hf_weights/t5-base"
    )
    model = model = T5Model.from_pretrained(
        "/tangyinzhou-tos-volc-engine/tyz/hf_weights/t5-base"
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
    def __init__(self, video_paths, sigma):
        self.video_paths = video_paths
        self.resize_transform = transforms.Resize((256, 256))
        self.sigma = sigma

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        data = np.load(video_path, allow_pickle=True)["arr_0"].tolist()
        vr = VideoReader(
            video_path.replace("/success/", "/videos/").replace(".npz", ".mp4"),
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

        obj = video_path.split("/")[-1].split("_")[1]
        tgt = video_path.split("/")[-1].split("_")[2]
        text = f"Pick up the {obj}.|Place the {obj} to the {tgt}."

        dones = get_dones(
            data["state"]["end"]["position"].reshape(-1, 3),
            data["action"]["effector"]["position_gripper"],
        )
        frame_num = min(frames.shape[0], action_all.shape[0])
        if not frames.shape[0] - 1 == action_all.shape[0]:
            print(1)
        return (
            resized_frames[:frame_num],
            action_all[:frame_num],
            text,
            dones[:frame_num],
            idx,
        )

        # text = data["instruction"]

        # return (resized_frames, action_all, text, idx)


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
    text_data_path = f"{output_dir}/text.bin"
    done_path = f"{output_dir}/done.bin"

    video_data = np.memmap(
        video_data_path, dtype=np.uint32, mode="w+", shape=(total_frame, 16, 16)
    )
    segment_data = np.memmap(
        segment_data_path, dtype=np.int32, mode="w+", shape=(total_frame, 1)
    )
    action_data = np.memmap(
        action_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 8)
    )
    text_data = np.memmap(
        text_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 768)
    )
    done_data = np.memmap(
        done_path, dtype=np.float32, mode="w+", shape=(total_frame, 1)
    )

    print(
        f"✅ 创建 memmap 文件：\n  - {video_data_path}\n  - {segment_data_path}\n  - {action_data_path}\n  - {text_data_path}\n  - {done_path}\n"
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

    return (video_data, segment_data, action_data, text_data, done_data)


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
    video_data,
    segment_data,
    action_data,
    text_data,
    done_data,
    output_dir,
):
    if os.path.exists(f"{output_dir}/text.txt"):
        print(f"{output_dir}/text.txt exists, removed")
        os.remove(f"{output_dir}/text.txt")
    frame_idx = 0
    batch_size = 128

    for resized_frames, clip_actions, clip_texts, clip_dones, video_id in tqdm(
        data_loader
    ):
        resized_frames = resized_frames[0].cuda(non_blocking=True)
        clip_actions = clip_actions[0].numpy()
        clip_dones = clip_dones[0].numpy()
        clip_texts = clip_texts[0].split("|")
        text_embeddings = [
            get_text_embedding(text_tokenizer, text_model, text) for text in clip_texts
        ]
        torch.cuda.empty_cache()
        gc.collect()

        # 分批次处理帧，防止 GPU 内存溢出
        num_batches = (resized_frames.size(0) + batch_size - 1) // batch_size
        for i in range(num_batches):
            start, end = i * batch_size, min(
                (i + 1) * batch_size, resized_frames.size(0)
            )
            batch_frames = resized_frames[start:end].cuda(non_blocking=True)
            batch_actions = clip_actions[start:end]
            batch_dones = clip_dones[start:end]
            # batch_text = clip_texts

            # FP16 加速推理
            with torch.no_grad():
                quant, _, _, _ = tokenizer.encode(batch_frames * 2 - 1)
                token_ids = (
                    tokenizer.quantize.bits_to_indices(quant.permute(0, 2, 3, 1) > 0)
                    .cpu()
                    .numpy()
                )

            # ✅ 写入 memmap 文件
            segnum1 = np.where(batch_dones < 10)[0].shape[0]
            segnum2 = np.where(batch_dones >= 10)[0].shape[0]
            video_data[frame_idx : frame_idx + batch_frames.size(0)] = token_ids

            action_data[frame_idx : frame_idx + batch_frames.size(0)] = (
                batch_actions.reshape(-1, 8)
            )
            segs = segnum1 * [2 * video_id] + segnum2 * [2 * video_id + 1]
            segment_data[frame_idx : frame_idx + batch_frames.size(0)] = segs

            if segnum1 == batch_frames.size(0) or segnum2 == batch_frames.size(0):
                text_output = np.tile(text_embeddings[0], (batch_frames.size(0), 1))
            else:
                text_output = np.concatenate(
                    [
                        np.tile(text_embeddings[0], (segnum1, 1)),
                        np.tile(text_embeddings[1], (segnum2, 1)),
                    ],
                    axis=0,
                )
            text_data[frame_idx : frame_idx + batch_frames.size(0)] = text_output

            done_output = (batch_dones % 10).reshape(-1, 1)
            done_data[frame_idx : frame_idx + batch_frames.size(0)] = done_output
            with open(f"{output_dir}/text.txt", "a", encoding="utf-8") as f:
                f.write((clip_texts[0] + "\n") * segnum1)
                f.write((clip_texts[1] + "\n") * segnum2)
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
        default="/tangyinzhou-tos-volc-engine/bingwen/ms_data/TabletopPickPlaceEnv-v1",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument(
        "--task", type=str, default="20250714_100315", help="each part is 1w"
    )
    parser.add_argument("--sigma", type=float, default=-1, help="each part is 1w")
    return parser.parse_args()


# ✅ 主函数：协调各个模块
def main():
    # 解析参数
    args = parse_args()
    gpu_id = args.gpu_id

    # 加载模型和数据
    tokenizer = load_model(gpu_id)
    text_tokenizer, text_encoder = load_text_model(gpu_id)
    data = load_data(args.source_path, args.task)
    train_data = data[: int(0.8 * len(data))]
    val_data = data[int(0.8 * len(data)) :]

    # 加载 Dataset 和 DataLoader
    train_video_dataset = VideoDataset(train_data, args.sigma)
    val_video_dataset = VideoDataset(val_data, args.sigma)

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
    (
        train_video_data,
        train_segment_data,
        train_action_data,
        train_text_data,
        train_done_data,
    ) = create_memmap_files(
        os.path.join(output_dir, f"{args.task}", "train"),
        train_total_frame,
    )
    (
        val_video_data,
        val_segment_data,
        val_action_data,
        val_text_data,
        val_done_data,
    ) = create_memmap_files(
        os.path.join(output_dir, f"{args.task}", "val"),
        val_total_frame,
    )

    # 处理视频数据
    process_videos(
        train_data_loader,
        tokenizer,
        text_tokenizer,
        text_encoder,
        train_video_data,
        train_segment_data,
        train_action_data,
        train_text_data,
        train_done_data,
        os.path.join(output_dir, f"{args.task}", "train"),
    )
    process_videos(
        val_data_loader,
        tokenizer,
        text_tokenizer,
        text_encoder,
        val_video_data,
        val_segment_data,
        val_action_data,
        val_text_data,
        val_done_data,
        os.path.join(output_dir, f"{args.task}", "val"),
    )

    print(
        f"🎉 任务完成！GPU {gpu_id} 处理 {len(data)} 个视频，数据已保存到 {output_dir}"
    )


# ✅ 启动主程序
if __name__ == "__main__":
    main()
