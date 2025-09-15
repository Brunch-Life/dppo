"""
Example usage:
`python genie/evaluate.py --checkpoint_dir 1x-technologies/GENIE_35M`
"""

import argparse
import time
import os
import sys
from collections import defaultdict
from pathlib import Path

import lpips
import torch
import transformers
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import default_data_collator
from skimage.metrics import peak_signal_noise_ratio as psnr

# 1xgpt imports
sys.path.append(os.getcwd())
from data import RawTokenDataset
from visualize import decode_latents_wrapper
from eval_utils import decode_tokens, compute_lpips, AvgMetric, compute_loss
from genie.st_mask_git import STMaskGIT, GenieConfig


# Hardcoded values for the v1.1 dataset
# WINDOW_SIZE = 16
# STRIDE = 15  # Data is 30 Hz so with stride 15, video is 2 Hz


def compute_psnr(frame1, frame2):
    return psnr(frame1, frame2)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GENIE-style models.")
    parser.add_argument(
        "--val_data_dir",
        type=str,
        default="/mnt/iag/user/zhangxin/shangyu/challenge-0308/data/val_v1.1",
        help="A directory with video data, should have a `metadata.json` and `video.bin`.",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, help="Path to a HuggingFace-style checkpoint."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size, current script only supports a single GPU.",
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
        "--save_outputs_dir",
        type=str,
        help="Debug option. If specified, will save model predictions and ground truths to this directory. "
        "Specifically, will save `{pred_frames,pred_logits,gtruth_frames,gtruth_tokens}.pt`",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        help="If specified, will stop evaluation early after `max_examples` examples.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=16,
        help="If specified, will stop evaluation early after `max_examples` examples.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="If specified, will stop evaluation early after `max_examples` examples.",
    )
    parser.add_argument(
        "--genie_config", type=str, help="Path to a HuggingFace-style checkpoint."
    )
    return parser.parse_args()


class GenieEvaluator:
    def __init__(self, args, decode_latents, device="cuda"):
        super().__init__()

        self.model = STMaskGIT.from_pretrained(args.checkpoint_dir)

        self.model = self.model.to(device=device)
        self.model.eval()

        self.decode_latents = decode_latents
        self.device = device
        self.args = args

    def predict_zframe_logits(
        self, input_ids: torch.LongTensor, actions, texts
    ) -> tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ... frame_{T-1}],
        predict the tokens in the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        Image logits are denoised in parallel across spatial dimension and teacher-forced
        across the time dimension. To compute logits, we save both the samples and logits as we do MaskGIT generation.

        Total number of forward passes is (T-1) * maskgit steps.

        Args:
            input_ids: LongTensor of size (B, T*H*W) corresponding to flattened, tokenized images.

        Returns: (samples_THW, factored_logits)
            samples_THW:
                size (B, T, H, W) corresponding to the token ids of the predicted frames.
                May differ from the argmax of `factored_logits` if not greedy sampling.
            factored_logits:
                size (B, 512, 2, T-1, H, W) corresponding to the predicted logits.
                Note that we are factorizing the 2**18 vocabulary into two separate vocabularies of size 512 each.
        """
        inputs_THW = rearrange(
            input_ids,
            "b (t h w) -> b t h w",
            t=self.args.window_size,
            h=self.args.latent_h,
            w=self.args.latent_w,
        ).to(self.device)
        input_action = actions.reshape(-1, self.args.window_size, 8).to(self.device)
        input_text = texts.reshape(-1, self.args.window_size, 768).to(self.device)
        all_samples = []
        all_logits = []
        for timestep in range(1, self.args.window_size):
            print(f"Generating frame {timestep}")
            inputs_masked = inputs_THW.clone()
            inputs_masked[:, timestep:] = self.model.mask_token_id

            # MaskGIT sampling
            samples_HW, factored_logits, done_pred = self.model.maskgit_generate(
                inputs_masked,
                input_action,
                input_text,
                out_t=timestep,
                maskgit_steps=self.args.maskgit_steps,
                temperature=self.args.temperature,
            )

            all_samples.append(samples_HW)
            all_logits.append(factored_logits)

        samples_THW = torch.stack(all_samples, dim=1)
        return samples_THW, torch.stack(all_logits, dim=3)

    def predict_next_frames(self, samples_THW) -> torch.Tensor:
        """
        All model submissions should have this defined.

        Like predict_next_frames, this is teacher-forced along time dimension, autoregressive along spatial dimension.

        Conditioned on each prefix: [frame_0], [frame_0, frame_1], ..., [frame_0, frame_1, ..., frame_{T-1}],
        predict the following frame: [pred_frame_1, pred_frame_2, ..., pred_frame_T].

        For this model, the frames are generated by using the argmax of `predict_zframe_logits`
        and decoding the quantized latent space tokens back to the original image space.

        Args:
            samples_THW: LongTensor of size (B, T, H, W) corresponding to sampled images in the quantized latent space.

        Returns:
            LongTensor of size (B, T-1, 3, 256, 256) corresponding to the predicted frames.
        """
        return decode_tokens(samples_THW.cpu(), self.decode_latents)


import cv2
import torch
import numpy as np


def mean_filtering(predicted_images, kernel_size=3):
    """
    对预测图像进行均值滤波
    :param predicted_images: 预测图像，形状为 [batch_size, num_frames, 3, 256, 256]
    :param kernel_size: 滤波核的大小，默认为 3
    :return: 滤波后的预测图像，形状为 [batch_size, num_frames, 3, 256, 256]
    """
    batch_size, num_frames, _, _, _ = predicted_images.shape
    filtered_images = []
    for batch in range(batch_size):
        batch_images = []
        for frame in range(num_frames):
            img = predicted_images[batch, frame].permute(1, 2, 0).cpu().numpy()
            # 进行均值滤波
            filtered_img = cv2.blur(img, (kernel_size, kernel_size))
            filtered_img = torch.from_numpy(filtered_img).permute(2, 0, 1)
            batch_images.append(filtered_img)
        batch_images = torch.stack(batch_images)
        filtered_images.append(batch_images)
    filtered_images = torch.stack(filtered_images)
    return filtered_images


def gaussian_filtering(predicted_images, kernel_size=3, sigma=0):
    """
    对预测图像进行高斯滤波
    :param predicted_images: 预测图像，形状为 [batch_size, num_frames, 3, 256, 256]
    :param kernel_size: 滤波核的大小，默认为 3
    :param sigma: 高斯核的标准差，默认为 0（自动计算）
    :return: 滤波后的预测图像，形状为 [batch_size, num_frames, 3, 256, 256]
    """
    batch_size, num_frames, _, _, _ = predicted_images.shape
    filtered_images = []
    for batch in range(batch_size):
        batch_images = []
        for frame in range(num_frames):
            img = predicted_images[batch, frame].permute(1, 2, 0).cpu().numpy()
            # 进行高斯滤波
            filtered_img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
            filtered_img = torch.from_numpy(filtered_img).permute(2, 0, 1)
            batch_images.append(filtered_img)
        batch_images = torch.stack(batch_images)
        filtered_images.append(batch_images)
    filtered_images = torch.stack(filtered_images)
    return filtered_images


@torch.no_grad()
def main():
    transformers.set_seed(42)
    args = parse_args()
    config = GenieConfig.from_pretrained(args.genie_config)
    val_dataset = RawTokenDataset(
        args.val_data_dir,
        window_size=args.window_size,
        config=config,
        stride=args.stride,
        filter_overlaps=True,
        split="val",
    )
    args.latent_h = args.latent_w = val_dataset.metadata["s"]

    decode_latents = decode_latents_wrapper()
    lpips_alex = lpips.LPIPS(
        net="alex"
    )  # Calculate LPIPS w/ AlexNet, which is the fastest model out of their options

    if args.max_examples is not None:
        val_dataset.valid_start_inds = val_dataset.valid_start_inds[: args.max_examples]

    dataloader = DataLoader(
        val_dataset, collate_fn=default_data_collator, batch_size=args.batch_size
    )

    evaluator = GenieEvaluator(args, decode_latents)
    metrics = defaultdict(AvgMetric)

    if args.save_outputs_dir is not None:
        outputs_to_save = defaultdict(list)

    for batch in tqdm(dataloader):
        batch_size = batch["input_ids"].size(0)
        reshaped_input_ids = rearrange(
            batch["input_ids"],
            "b (t h w) -> b t h w",
            t=args.window_size,
            h=args.latent_h,
            w=args.latent_w,
        )

        start_time = time.time()
        samples, factored_logits = evaluator.predict_zframe_logits(
            batch["input_ids"], batch["actions"], batch["texts"]
        )
        # print(batch["actions"].size())
        frames_per_batch = (args.window_size - 1) * batch["input_ids"].size(0)
        metrics["gen_time"].update(
            (time.time() - start_time) / frames_per_batch, batch_size
        )

        loss = compute_loss(batch["labels"], factored_logits)

        acc = (reshaped_input_ids[:, 1:].to("cuda") == samples).float().mean().item()

        metrics["loss"].update(loss, batch_size)
        metrics["acc"].update(acc, batch_size)

        start_time = time.time()
        pred_frames = evaluator.predict_next_frames(samples)
        metrics["dec_time"].update(
            (time.time() - start_time) / frames_per_batch, batch_size
        )

        decoded_gtruth = decode_tokens(reshaped_input_ids, decode_latents)
        metrics["pred_lpips"].update_list(
            compute_lpips(decoded_gtruth[:, 1:], pred_frames, lpips_alex)
        )

        # 后处理
        # pred_frames = mean_filtering(pred_frames)
        # pred_frames = gaussian_filtering(pred_frames)
        # metrics["pred_lpips"].update_list(compute_lpips(decoded_gtruth[:, 1:], pred_frames, lpips_alex))

        psnr = [
            compute_psnr(
                decoded_gtruth[i, 1:].cpu().numpy(), pred_frames[i, :].cpu().numpy()
            )
            for i in range(batch_size)
        ]
        metrics["psnr"].update_list(psnr)

        # print({key: f"{val.mean():.4f}" for key, val in metrics.items()})
        # print(decoded_gtruth[:,0].size())
        # print(pred_frames.size()) #16 15 3 256 256

        # delta_PSNR
        samples2, factored_logits2 = evaluator.predict_zframe_logits(
            batch["input_ids"], batch["random_actions"], batch["random_texts"]
        )
        pred_frames2 = evaluator.predict_next_frames(samples2)
        # 后处理
        # pred_frames2 = mean_filtering(pred_frames2)
        psnr_random = [
            compute_psnr(
                decoded_gtruth[i, 1:].cpu().numpy(), pred_frames2[i, :].cpu().numpy()
            )
            for i in range(batch_size)
        ]
        delta_psnr = list(np.array(psnr) - np.array(psnr_random))
        metrics["delta_psnr"].update_list(delta_psnr)

        print({key: f"{val.mean():.4f}" for key, val in metrics.items()})

        if args.save_outputs_dir is not None:
            outputs_to_save["pred_frames"].append(pred_frames)
            outputs_to_save["pred_logits"].append(factored_logits)
            outputs_to_save["gtruth_frames"].append(decoded_gtruth)
            outputs_to_save["gtruth_tokens"].append(reshaped_input_ids)

        # break
    if args.save_outputs_dir is not None:
        os.makedirs(args.save_outputs_dir, exist_ok=True)
        save_outputs_dir = Path(args.save_outputs_dir)
        torch.save(
            torch.cat(outputs_to_save["pred_frames"], dim=0).cpu(),
            save_outputs_dir / "pred_frames.pt",
        )
        torch.save(
            torch.cat(outputs_to_save["pred_logits"], dim=0).cpu(),
            save_outputs_dir / "pred_logits.pt",
        )
        torch.save(
            torch.cat(outputs_to_save["gtruth_frames"], dim=0).cpu(),
            save_outputs_dir / "gtruth_frames.pt",
        )
        torch.save(
            torch.cat(outputs_to_save["gtruth_tokens"], dim=0).cpu(),
            save_outputs_dir / "gtruth_tokens.pt",
        )


if __name__ == "__main__":
    main()
