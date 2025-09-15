"""
Example usage: See https://github.com/1x-technologies/1xgpt?tab=readme-ov-file#1x-genie-baseline
"""

import argparse
import json
import os
import sys
from pathlib import Path

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
        "--not_use_bin_action",
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
        noise_ratio=args.action_noise,
        noise_dim=args.noise_dim,
        use_bin_action=not args.not_use_bin_action,
        norm_7_dim=False,
    )
    latent_side_len = val_dataset.metadata["s"]
    # action_dim = 7
    action_dim = 8
    text_dim = 768
    # Get single example
    # print(len(val_dataset))
    example_ind = args.example_ind
    action_example_ind = args.action_ind
    if args.force_end:
        while True:
            example_done = (
                val_dataset[example_ind]["done"]
                .reshape(1, args.window_size, 1)
                .to("cuda")
            )

            if 1 in example_done:
                # if True:
                example_THW = (
                    val_dataset[example_ind]["input_ids"]
                    .reshape(1, args.window_size, latent_side_len, latent_side_len)
                    .to("cuda")
                )
                if args.action_random:
                    example_action = (
                        val_dataset[example_ind]["random_actions"]
                        .reshape(1, args.window_size, action_dim)
                        .to("cuda")
                    )
                else:
                    example_action = (
                        val_dataset[example_ind]["actions"]
                        .reshape(1, args.window_size, action_dim)
                        .to("cuda")
                    )
                # example_action = torch.random(example_action.shape).to("cuda")

                example_text = (
                    val_dataset[example_ind]["texts"]
                    .reshape(1, args.window_size, text_dim)
                    .to("cuda")
                )
                # example_text = torch.zeros(example_text.shape).to("cuda")
                break
            else:
                example_ind += 1
                continue
    else:
        example_THW = (
            val_dataset[example_ind]["input_ids"]
            .reshape(1, args.window_size, -1, latent_side_len)
            .to("cuda")
        )
        if args.action_random:
            example_action = (
                val_dataset[example_ind]["actions"]
                .reshape(1, args.window_size, action_dim)
                .to("cuda")
            )
            new_scene_action = (
                val_dataset[action_example_ind]["actions"]
                .reshape(1, args.window_size, action_dim)
                .to("cuda")
            )
            example_action[:, args.num_prompt_frames :] = new_scene_action[
                :, args.num_prompt_frames :
            ]
            # example_action = torch.zeros(example_action.shape).to("cuda")
            # example_action = torch.empty_like(example_action).uniform_(
            #     example_action.min(), example_action.max()
            # )
        else:
            example_action = (
                val_dataset[example_ind]["actions"]
                .reshape(1, args.window_size, action_dim)
                .to("cuda")
            )
            # 生成与 x 同形状的高斯噪声 N(0, 1)，再乘比例
            # ratio = args.action_noise
            # eps = torch.randn_like(example_action)
            # n_eps, n_act = eps.norm(2), example_action.norm(2)
            # if n_eps.item() < 1e-12:  # 防 0
            #     eps = torch.randn_like(example_action)
            #     n_eps = eps.norm(2)
            # eps = eps * (ratio * n_act / n_eps)
            # if args.noise_dim == -1:
            #     example_action = example_action + eps
            # else:
            #     if args.noise_dim == 123:
            #         args.noise_dim = [0, 1, 2]
            #     if args.noise_dim == 456:
            #         args.noise_dim = [3, 4, 5]
            #     result = torch.zeros_like(eps)
            #     result[:, :, args.noise_dim] = eps[:, :, args.noise_dim]
            #     eps = result
        # example_action = torch.random(example_action.shape).to("cuda")

        example_text = (
            val_dataset[example_ind]["texts"]
            .reshape(1, args.window_size, text_dim)
            .to("cuda")
        )

    # Load the model checkpoint
    model = STMaskGIT.from_pretrained(args.checkpoint_dir).to("cuda")
    model.eval()

    samples = []
    prompt_THW = example_THW.clone()
    prompt_THW[:, args.num_prompt_frames :] = model.mask_token_id

    prompt_action = example_action.clone()
    prompt_text = example_text.clone()
    # prompt_action[:, args.num_prompt_frames:] = model.mask_token_id
    for timestep in range(args.num_prompt_frames, args.window_size):
        # Teacher-forced, maskgit generation
        if args.teacher_force_time:
            prompt_THW = example_THW.clone()
            # Masked prediction for this timestep only, after which we provide ground-truth
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

    outputs = torch.stack(samples, dim=1)
    # prepend prompt sequence
    outputs = torch.cat([example_THW[:, : args.num_prompt_frames], outputs], dim=1)

    # append ground-truth targets next to generated outputs for comic strip generation
    # [<prompt frames><predicted frames><ground truth frames>]
    outputs = torch.cat([outputs, example_THW[:, args.num_prompt_frames :]], dim=1)

    # write to output
    output_dir = Path(args.output_dir)
    import json

    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "action.json", "w") as f:
        prompt_action = [
            list(map(float, list(x.cpu().numpy()))) for x in prompt_action[0]
        ]
        json.dump(prompt_action, f)
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
                # "h": 2 * latent_side_len,
                "h": latent_side_len,
                "w": latent_side_len,
                "t": args.window_size,
            },
            f,
        )


if __name__ == "__main__":
    main()
