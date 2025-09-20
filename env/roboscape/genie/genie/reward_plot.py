import argparse
import json
import os
import sys
from pathlib import Path
import warnings
import torch
import numpy as np
import lpips
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import *
import imageio
import warnings

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from env.roboscape.genie.data import RawTokenDataset
from env.roboscape.genie.genie.st_mask_git import STMaskGIT, GenieConfig
from env.roboscape.genie.eval_utils import compute_lpips, decode_tokens
from env.roboscape.genie.visualize import decode_latents_wrapper


def cal_reward(samples_HW, done, target_observations):
    lpips_net = lpips.LPIPS(net="alex")
    decoded_pred = decode_tokens(samples_HW.cpu(), decode_latents_wrapper())
    target_observations = decode_tokens(
        target_observations.cpu(), decode_latents_wrapper()
    )
    dense_reward = 1 - compute_lpips(decoded_pred, target_observations, lpips_net)[0]
    sparse_reward = [10 if x >= 0.9 else 0 for x in done[:, 0]]
    return (
        np.array(dense_reward).reshape(-1) + np.array(sparse_reward).reshape(-1),
        decoded_pred,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates samples (as tokens) from GENIE model. "
        "Optionally visualizes these tokens as GIFs or comics."
    )
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/iag_ad_01/ad/zhangxin11/tyz/observation-reward-model/data/pick_tomato/val",
        help="A directory with video data, should have a `metadata.json` and `video.bin` We generate using the first frames of this dataset.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/iag_ad_01/ad/zhangxin11/tyz/observation-reward-model/data/pick_tomato_only_reward/step_60000",
        help="Path to a HuggingFace-style checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/genie_generated",
        help="Directory to save generated outputs.",
    )
    parser.add_argument(
        "--num_prompt_frames", type=int, default=8, help="The number of context frames."
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
        default=1,
        help="Will generate `window_size - num_prompt_frames` frames.",
    )

    parser.add_argument(
        "--example_ind",
        type=int,
        default=0,
        help="The index in the dataset of the example to generate on.",
    )
    parser.add_argument(
        "--action_ind",
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
    parser.add_argument(
        "--action_noise",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--noise_dim",
        type=int,
        default=-1,
        help="0-7, -1 refers to all",
    )
    parser.add_argument(
        "--action_random",
        action="store_true",
    )
    parser.add_argument(
        "--force_end",
        action="store_true",
    )
    parser.add_argument(
        "--use_bin_action",
        action="store_true",
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
        use_bin_action=args.use_bin_action,
        norm_7_dim=False,
    )
    choose_idx = []  # 临时存储当前片段的有效索引
    current_id = 0  # 当前处理的片段ID
    ptr = 0  # 指针，用于遍历数据集
    valid_start_inds = []
    # 遍历数据集的有效起始索引，筛选出每个片段的有效起始点
    while ptr < len(val_dataset.valid_start_inds):
        # 如果当前指针指向的索引属于当前片段ID
        if val_dataset.segment_ids[val_dataset.valid_start_inds[ptr]] == current_id:
            choose_idx.append(ptr)
            ptr += val_dataset.video_len
        # 如果当前指针指向的索引属于更大的片段ID（说明当前片段已处理完）
        elif val_dataset.segment_ids[val_dataset.valid_start_inds[ptr]] > current_id:
            # 调整指针位置，找到当前片段的最后一个有效索引
            ptr -= val_dataset.video_len
            while (
                val_dataset.segment_ids[val_dataset.valid_start_inds[ptr]] == current_id
            ):
                ptr += val_dataset.stride
            ptr -= val_dataset.stride
            choose_idx.append(ptr)  # 记录当前片段的最后一个有效索引

            # 移动指针到下一个片段
            while (
                val_dataset.segment_ids[val_dataset.valid_start_inds[ptr]] == current_id
            ):
                ptr += 1
            # 更新当前片段ID为新片段ID
            current_id = val_dataset.segment_ids[val_dataset.valid_start_inds[ptr]]
            valid_start_inds.append(choose_idx)  # 保存当前片段的有效索引列表
            choose_idx = []  # 重置选择列表
        else:
            ptr += 1

    example = valid_start_inds[0]
    init_frames = []
    init_actions = []
    init_texts = []
    gt_actions = []
    gt_observations = []
    latent_side_len = val_dataset.metadata["s"]
    # Get single example
    # print(len(val_dataset))

    for i, example_ind in tqdm(enumerate(example)):
        example_THW = (
            val_dataset[example_ind]["input_ids"].reshape(16, -1, 16).to("cuda")
        )
        example_action = val_dataset[example_ind]["actions"].reshape(16, 7).to("cuda")
        example_text = val_dataset[example_ind]["texts"].reshape(16, 768).to("cuda")
        if i == 0:
            init_frames.append(example_THW)  # 编码后的首帧
            # 处理action
            init_actions.append(example_action)
            init_texts.append(example_text)
            gt_action = example_action
            gt_obs = example_THW
        elif example[i] - example[i - 1] == val_dataset.video_len:
            gt_action = torch.concat([gt_action, example_action], dim=0)
            gt_obs = torch.concat([gt_obs, example_THW], dim=0)
        else:
            gap = int((example[i] - example[i - 1]) / val_dataset.stride)
            gt_action = torch.concat([gt_action, example_action[gap:]], dim=0)
            gt_obs = torch.concat([gt_obs, example_THW[gap:]], dim=0)
            gt_actions.append(gt_action)
            gt_observations.append(gt_obs)
    target_observation = gt_observations[0][-1]
    # Load the model checkpoint
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to("cuda")
    model.eval()
    rewards = []
    os.makedirs("/ML-vePFS/tangyinzhou/RoboScape-R/dppo/vis_reward", exist_ok=True)
    for ptr, obs in enumerate(gt_observations[0]):
        reward, decoded_pred = cal_reward(
            obs.unsqueeze(0).unsqueeze(0),
            np.array([0.0]).reshape(1, 1),
            target_observation.unsqueeze(0).unsqueeze(0),
        )
        imageio.imwrite(
            "/ML-vePFS/tangyinzhou/RoboScape-R/dppo/vis_reward/{}.png".format(ptr),
            decoded_pred.squeeze(0).squeeze(0).permute(1, 2, 0).cpu().numpy(),
        )
        rewards.append(reward)
    plt.plot(rewards)
    plt.savefig("/ML-vePFS/tangyinzhou/RoboScape-R/dppo/reward.png")


if __name__ == "__main__":
    main()
