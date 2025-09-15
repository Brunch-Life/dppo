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
def load_data(train_dir, part):
    data = []
    with open(train_dir, "r") as f:
        for idx, line in tqdm(enumerate(f), total=100000):
            if not 10000 * part <= idx < 10000 * (part + 1):
                continue
            # if idx > 10:
            #     break
            line = json.loads(line)
            if os.path.exists(
                line["clip_path"].replace(
                    "/iag_ad_01/ad/zhangxin11/tyz/AgiBotWorld-Beta",
                    "/tangyinzhou-tos-volc-engine/tyz/AgiBotWorld-Beta/AgiBotWorld-Beta",
                )
            ) and os.path.exists(
                line["action_path"].replace(
                    "/iag_ad_01/ad/zhangxin11/tyz/AgiBotWorld-Beta",
                    "/tangyinzhou-tos-volc-engine/tyz/AgiBotWorld-Beta/AgiBotWorld-Beta",
                )
            ):
                data.append(
                    {
                        "clip_path": line["clip_path"].replace(
                            "/iag_ad_01/ad/zhangxin11/tyz/AgiBotWorld-Beta",
                            "/tangyinzhou-tos-volc-engine/tyz/AgiBotWorld-Beta/AgiBotWorld-Beta",
                        ),
                        "action_path": line["action_path"].replace(
                            "/iag_ad_01/ad/zhangxin11/tyz/AgiBotWorld-Beta",
                            "/tangyinzhou-tos-volc-engine/tyz/AgiBotWorld-Beta/AgiBotWorld-Beta",
                        ),
                        "keyframe": line["keyframe"],
                        "instruction": line["instruction"],
                    }
                )
    return data


# ✅ 定义 VideoDataset 类
class VideoDataset(Dataset):
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.resize_transform = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        data = self.video_paths[idx]
        video_path = data["clip_path"]
        # 加载视频帧并进行 Resize 处理
        vr = VideoReader(
            video_path,
            ctx=cpu(0),
            num_threads=4,
        )
        frames = vr.get_batch(range(len(vr))).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) / 255.0  # [N, 3, H, W]
        resized_frames = torch.stack([self.resize_transform(frame) for frame in frames])

        actions = np.load(data["action_path"], allow_pickle=True).tolist()
        try:
            action_all = np.concatenate(
                [
                    actions["end"]["position"],
                    actions["end"]["orientation"],
                    actions["effector"]["position"].reshape(
                        actions["effector"]["position"].shape + (1,)
                    ),
                ],
                axis=2,
            )  # N, 2, 8
        except:
            print("fail to concat")
            action_all = np.zeros((frames.shape[0], 2, 8))
        text = data["instruction"]

        return (resized_frames, action_all, text, idx)


# ✅ 计算当前 GPU 处理的数据总帧数
def calculate_total_frames(data):
    total_frame = 0
    for item in tqdm(data):
        # 打开视频文件
        video = cv2.VideoCapture(item["clip_path"])

        # 检查视频是否成功打开
        if not video.isOpened():
            raise ValueError("无法打开视频文件")

        # 获取视频的总帧数
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

        # 关闭视频文件
        video.release()
        total_frame += total_frames
    print(f"✅ 计算总帧数: {total_frame}")
    return total_frame, len(data)


# ✅ 创建 memmap 文件路径，根据 `item_range` 创建不同的文件
def create_memmap_files(output_dir, total_frame, clip_num):
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
    done_dense_path = f"{output_dir}/done_dense.bin"
    done_sparse_path = f"{output_dir}/done_sparse.bin"

    video_data = np.memmap(
        video_data_path, dtype=np.uint32, mode="w+", shape=(total_frame, 16, 16)
    )
    segment_data = np.memmap(
        segment_data_path, dtype=np.int32, mode="w+", shape=(total_frame, 1)
    )
    action_data = np.memmap(
        action_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 16)
    )
    text_data = np.memmap(
        text_data_path, dtype=np.float32, mode="w+", shape=(total_frame, 768)
    )
    done_dense_data = np.memmap(
        done_dense_path, dtype=np.float32, mode="w+", shape=(total_frame, 1)
    )
    done_sparse_data = np.memmap(
        done_sparse_path, dtype=np.float32, mode="w+", shape=(total_frame, 1)
    )

    print(
        f"✅ 创建 memmap 文件：\n  - {video_data_path}\n  - {segment_data_path}\n  - {action_data_path}\n  - {text_data_path}\n  - {done_dense_path}\n  - {done_sparse_path}"
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

    return (
        video_data,
        segment_data,
        action_data,
        text_data,
        done_dense_data,
        done_sparse_data,
    )


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
    done_dense_data,
    done_sparse_data,
):
    frame_idx = 0
    batch_size = 256

    for resized_frames, clip_actions, clip_texts, video_id in tqdm(data_loader):
        resized_frames = resized_frames[0].cuda(non_blocking=True)
        clip_actions = clip_actions[0].numpy()
        clip_texts = clip_texts[0]

        # 使用分词器对文本进行编码
        inputs = text_tokenizer.encode_plus(
            clip_texts,
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
            outputs = text_model.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # 获取最后一层的隐藏状态
        last_hidden_state = outputs.last_hidden_state

        text_embedding = last_hidden_state[:, 0, :].cpu().numpy()[0]
        del inputs, input_ids, attention_mask, last_hidden_state, outputs
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
            video_data[frame_idx : frame_idx + batch_frames.size(0)] = token_ids
            action_data[frame_idx : frame_idx + batch_frames.size(0)] = (
                batch_actions.reshape(-1, 16)
            )
            segment_data[frame_idx : frame_idx + batch_frames.size(0)] = video_id
            # text_embedding_concat = [text_embedding]
            # text_embedding_concat.extend(
            #     [
            #         np.array([0 for x in range(len(text_embedding))])
            #         for _ in range(batch_frames.size(0) - 1)
            #     ]
            # )
            text_data[frame_idx : frame_idx + batch_frames.size(0)] = np.tile(
                text_embedding, (batch_frames.size(0), 1)
            )
            frames = np.array(list(range(start + 1, end + 1)))
            done_sparse = np.array(frames == resized_frames.size(0)).astype(int)
            done_dense = 1 - (resized_frames.size(0) - frames) / resized_frames.size(0)

            done_dense_data[frame_idx : frame_idx + batch_frames.size(0)] = (
                done_dense.reshape(-1, 1)
            )
            done_sparse_data[frame_idx : frame_idx + batch_frames.size(0)] = (
                done_sparse.reshape(-1, 1)
            )

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
        default="/tangyinzhou-tos-volc-engine/tyz/AgiBotWorld-Beta/AgiBotWorld-Beta/processed_jsonl/clips_10000_clean.jsonl",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/tangyinzhou-tos-volc-engine/tyz/text_action_split_10000_new",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--part", type=int, default=0, help="each part is 1w")
    return parser.parse_args()


# ✅ 主函数：协调各个模块
def main():
    # 解析参数
    args = parse_args()
    gpu_id = args.gpu_id

    # 加载模型和数据
    tokenizer = load_model(gpu_id)
    text_tokenizer, text_encoder = load_text_model(gpu_id)
    data = load_data(args.source_path, args.part)
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
    train_total_frame, train_clip_num = calculate_total_frames(train_data)
    val_total_frame, val_clip_num = calculate_total_frames(val_data)

    # 创建 memmap 文件
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    (
        train_video_data,
        train_segment_data,
        train_action_data,
        train_text_data,
        train_done_dense,
        train_done_sparse,
    ) = create_memmap_files(
        os.path.join(output_dir, f"part_{args.part}", "train"),
        train_total_frame,
        train_clip_num,
    )
    (
        val_video_data,
        val_segment_data,
        val_action_data,
        val_text_data,
        val_done_dense,
        val_done_sparse,
    ) = create_memmap_files(
        os.path.join(output_dir, f"part_{args.part}", "val"),
        val_total_frame,
        val_clip_num,
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
        train_done_dense,
        train_done_sparse,
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
        val_done_dense,
        val_done_sparse,
    )

    print(
        f"🎉 任务完成！GPU {gpu_id} 处理 {len(data)} 个视频，数据已保存到 {output_dir}"
    )


# ✅ 启动主程序
if __name__ == "__main__":
    main()
