"""
Example usage: See https://github.com/1x-technologies/1xgpt?tab=readme-ov-file#1x-genie-baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import torch
import numpy as np

sys.path.append(os.getcwd())
from data import RawTokenDataset
from genie.st_mask_git import STMaskGIT, GenieConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates samples (as tokens) from GENIE model. "
        "Optionally visualizes these tokens as GIFs or comics."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="data/val_v1.1",
        help="A directory with video data, should have a `metadata.json` and `video.bin` We generate using the first frames of this dataset.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/genie_generated",
        help="Directory to save generated outputs.",
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=1, help="The number of context frames."
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="Will generate `window_size - num_prompt_frames` frames.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Will generate `window_size - num_prompt_frames` frames.",
    )

    parser.add_argument(
        "--example_ind",
        type=int,
        default=0,
        help="The index in the dataset of the example to generate on.",
    )
    parser.add_argument(
        "--teacher_force_time",
        action="store_true",
        help="If True, teacher-forces generation in time dimension.",
    )
    parser.add_argument(
        "--maskgit_steps", type=int, default=2, help="Number of MaskGIT sampling steps."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature. If `temperature` <= 1e-8, will do greedy sampling.",
    )
    parser.add_argument("--genie_config", type=str, help="GenieConfig json.")
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    assert args.num_prompt_frames <= args.window_size
    config = GenieConfig.from_pretrained(args.genie_config)
    val_dataset = RawTokenDataset(
        args.val_data_dir,
        config=config,
        window_size=args.window_size,
        stride=args.stride,
        split="val",
        filter_overlaps=True,
        rollout=True,
    )
    latent_side_len = val_dataset.metadata["s"]
    action_dim = 8
    text_dim = 768
    # Get single example
    # print(len(val_dataset))

    # 假设 val_dataset、args、latent_side_len、action_dim 等已经定义
    example_THW = (
        val_dataset[args.example_ind]["input_ids"]
        .reshape(1, -1, latent_side_len, latent_side_len)
        .to("cuda")
    )
    example_action = (
        val_dataset[args.example_ind]["actions"].reshape(1, -1, action_dim).to("cuda")
    )
    example_action = torch.empty_like(example_action).uniform_(
        example_action.min(), example_action.max()
    )
    example_done = val_dataset[args.example_ind]["done"].reshape(1, -1, 1).to("cuda")
    example_text = (
        val_dataset[args.example_ind]["texts"].reshape(1, -1, text_dim).to("cuda")
    )
    # 加载模型检查点
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to("cuda")
    model.eval()

    all_samples = []
    # 每次推理的窗口大小
    window_size = 16
    # 总的帧数
    total_frames = example_THW.shape[1]

    for start_frame in range(0, total_frames, window_size - 1):
        print(start_frame)
        end_frame = min(start_frame + window_size, total_frames)
        # 截取当前窗口的输入
        if start_frame == 0:
            current_THW = example_THW[:, start_frame:end_frame].clone()
            prompt_THW = current_THW.clone()
            prompt_THW[:, 1:] = model.mask_token_id
        else:
            current_THW = example_THW[:, start_frame:end_frame].clone()
            prompt_THW = current_THW.clone()
            prompt_THW[:, 0] = last_frame
            prompt_THW[:, 1:] = model.mask_token_id

        current_action = example_action[:, start_frame:end_frame].clone()
        prompt_action = current_action.clone()
        current_text = example_text[:, start_frame:end_frame].clone()
        prompt_text = current_text.clone()
        if not prompt_THW.shape[1] == window_size:
            pad_len = window_size - prompt_THW.shape[1]
            prompt_THW = torch.cat(
                [
                    current_THW.clone(),
                    torch.zeros(
                        1,
                        pad_len,
                        16,
                        16,
                        dtype=current_THW.dtype,
                        device=current_THW.device,
                    ),
                ],
                dim=1,
            )
            prompt_THW[:, 1:] = model.mask_token_id
            prompt_action = torch.cat(
                [
                    current_action.clone(),
                    torch.zeros(
                        1,
                        pad_len,
                        action_dim,
                        dtype=current_action.dtype,
                        device=current_action.device,
                    ),
                ],
                dim=1,
            )
            prompt_text = torch.cat(
                [
                    current_text.clone(),
                    torch.zeros(
                        1,
                        pad_len,
                        text_dim,
                        dtype=current_text.dtype,
                        device=current_text.device,
                    ),
                ],
                dim=1,
            )
        # # 只取首帧作为提示
        # prompt_THW = current_THW.clone()
        # prompt_THW[:, 1:] = model.mask_token_id

        samples = []
        for timestep in range(1, window_size):
            if (
                end_frame - start_frame < window_size
                and timestep >= end_frame - start_frame
            ):
                break
            # Teacher-forced, maskgit generation
            if args.teacher_force_time:
                prompt_THW = current_THW.clone()
                # 仅对当前时间步进行掩码预测，之后提供真实值
                prompt_THW[:, timestep:] = model.image_mask_token

            samples_HW, _, done_pred = model.maskgit_generate(
                prompt_THW,
                prompt_action,
                prompt_text,
                out_t=timestep,
                maskgit_steps=args.maskgit_steps,
                temperature=args.temperature,
            )

            samples.append(samples_HW)
            if not args.teacher_force_time:
                # autoregressive
                prompt_THW[:, timestep] = samples_HW
            # print(samples_HW.shape)
        last_frame = samples_HW.clone()
        if len(samples) > 0:
            outputs = torch.stack(samples, dim=1)
            # print(outputs.size())
            all_samples.append(outputs)

    # 合并所有生成的帧
    final_outputs = torch.cat(all_samples, dim=1)
    # prepend prompt sequence
    # print(final_outputs.shape)
    outputs = torch.cat([example_THW[:, :1], final_outputs], dim=1)
    print(outputs.size())
    print(example_THW.size())
    # append ground-truth targets next to generated outputs for comic strip generation
    # [<prompt frames><predicted frames><ground truth frames>]
    outputs = torch.cat([outputs, example_THW], dim=1)

    # write to output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs.cpu().numpy().astype(np.dtype(val_dataset.metadata["token_dtype"])).tofile(
        output_dir / "video.bin"
    )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(
            vars(args)
            | val_dataset.metadata
            | {
                "num_images": outputs.shape[1],
                "h": latent_side_len,
                "w": latent_side_len,
                "t": args.window_size,
            },
            f,
        )


if __name__ == "__main__":
    main()
