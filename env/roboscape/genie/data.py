import json
import math
import os
import random
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset as TorchDataset
from scipy.spatial.transform import Rotation as R
from env.roboscape.genie.genie.factorization_utils import (
    factorize_token_ids,
    unfactorize_token_ids,
)
from env.roboscape.genie.genie.config import GenieConfig
from env.roboscape.genie.genie.st_mask_git import cosine_schedule


class RawTokenDataset(TorchDataset):
    """Loads raw uint32 tokens as memmap-backed array"""

    def __init__(
        self,
        data_dir,
        window_size,
        config: GenieConfig,
        stride=1,
        filter_interrupts=True,
        filter_overlaps=False,
        split="train",
        rollout=False,
        use_action=None,
        use_text=None,
        hybrid_sample=True,
        noise_ratio=None,
        noise_dim=None,
        use_bin_action=True,
        use_target_action=True,
        norm_7_dim=True,
    ):
        """
        Args:
            data_dir: directory with the same format as `data/train_v0` and `data/val_v0`.
                Notably, has `video.bin` and `metadata.json`
            window_size: number of frames per "video" sequence
            stride: frame skip
            filter_interrupts: Under 3% of training frame sequences are the concatenation of two different clips.
                If filter_interrupts is True, will filter out these sequences using the segment ids.
            filter_overlaps: If False (default), one frame will appear in multiple examples;
                e.g. frame 0 might appear as the first frame in example 0 and also the second frame in example 15.
                If True, will filter out examples so that each frame appears at most once in the dataset.
        """
        root_dir = Path(data_dir)
        data_dir = root_dir / split
        if not os.path.exists(data_dir / "metadata.json"):
            data_dir = root_dir
        with open(data_dir / "metadata.json") as f:
            self.metadata = json.load(f)

        shape = (
            self.metadata["num_images"],
            self.metadata["h"],
            self.metadata["w"],
        )
        (
            video_tokens_path,
            segment_ids_path,
            action_tokens_path,
            text_tokens_path,
            done_path,
        ) = [
            data_dir / f"{name}.bin"
            for name in [
                "video",
                "segment_ids",
                "target_pose" if use_target_action else "action",
                "text",
                "done",
            ]
        ]
        token_dtype = np.dtype(self.metadata.get("token_dtype", "uint32"))
        self.data = np.memmap(
            video_tokens_path, dtype=token_dtype, mode="r", shape=shape
        )
        # model config
        self.use_action = config.use_action if use_action is None else use_action
        self.use_text = config.use_text if use_text is None else use_text
        self.use_done = config.use_done
        self.hybrid_sample = hybrid_sample
        # noise and bin config
        self.noise_ratio = noise_ratio
        self.noise_dim = noise_dim
        self.use_bin_action = use_bin_action
        # agibot
        if os.path.exists(action_tokens_path):
            self.actions = np.memmap(
                action_tokens_path, dtype=np.float32, mode="r"
            ).reshape(self.metadata["num_images"], -1)
            if self.actions.shape[1] == 8 and norm_7_dim:
                quat = self.actions[:, 3:7]  # shape: (..., 4)
                # 转成欧拉角 (roll, pitch, yaw)，默认是 'xyz' 顺序
                euler = np.stack(
                    [R.from_quat(x).as_euler("xyz") for x in quat]
                )  # shape: (..., 3)
                # 然后你可以替换原来的四元数部分
                self.actions = np.concatenate(
                    [
                        self.actions[:, :3],  # (..., 3)
                        euler,  # (..., 3)
                        self.actions[:, -1].reshape(-1, 1),
                    ],
                    axis=-1,
                )  # 最终 shape: (..., 7)
            self.num_bins_per_dim = 256
            self.fit_bins(self.actions)
            # 步骤1：计算每个维度的最大值和最小值（用于归一化/反归一化）
            self.dim_mins = self.actions.min(axis=0)  # (7,)：每个维度的最小值
            self.dim_maxs = self.actions.max(axis=0)  # (7,)：每个维度的最大值
            # 避免分母为0（若某维度所有值相同，max-min=0，直接用原始尺度）
            self.dim_ranges = np.where(
                self.dim_maxs - self.dim_mins < 1e-12,
                1.0,
                self.dim_maxs - self.dim_mins,
            )
        else:
            self.actions = None
        if os.path.exists(text_tokens_path):
            self.texts = np.memmap(
                text_tokens_path, dtype=np.float32, mode="r"
            ).reshape(self.metadata["num_images"], -1)
        else:
            self.texts = None

        if os.path.exists(segment_ids_path):
            self.segment_ids = np.memmap(
                segment_ids_path, dtype=np.int32, mode="r"
            ).reshape(self.metadata["num_images"], -1)
        else:
            self.segment_ids = None

        if os.path.exists(done_path):
            self.done = np.memmap(done_path, dtype=np.float32, mode="r").reshape(
                self.metadata["num_images"], -1
            )
        else:
            self.done = None

        self.window_size, self.stride = window_size, stride  # 16 15
        # Number of frames between the first and last frames of a video sequence (excluding one endpoint frame)
        self.video_len = (self.window_size - 1) * self.stride  # 225 2Hz
        print("video length:", self.video_len)
        self.valid_start_inds = []
        self.cliped_segment_ids = []
        if not self.segment_ids is None:
            for start_ind in tqdm(range(len(self.data) - self.video_len)):
                if not (
                    filter_interrupts
                    and self.segment_ids[start_ind]
                    != self.segment_ids[start_ind + self.video_len]
                ):
                    self.valid_start_inds.append(start_ind)
                    self.cliped_segment_ids.append(self.segment_ids[start_ind])
            print("valid_start_inds num", len(self.valid_start_inds))

        # print(stride)
        if filter_overlaps:
            # Instead of using a sliding window, use each frame at most once
            filtered_start_inds = []
            for start_ind in self.valid_start_inds:
                overlapping_start_inds = {
                    start_ind - i * self.stride for i in range(1, self.window_size)
                }
                # all sequences from `overlapping_start_inds` will also contain `start_ind`,
                # so exclude sequence starting from `start_ind` if any of `overlapping_start_inds` is already being used
                for existing_start_ind in filtered_start_inds[
                    -self.window_size * self.stride :
                ]:
                    # Bound could be improved
                    if existing_start_ind in overlapping_start_inds:
                        break
                else:
                    filtered_start_inds.append(start_ind)

            self.valid_start_inds = filtered_start_inds
        self.rollout = rollout
        if rollout:
            self.vid_lens = {}
            filtered_start_inds = []
            for start_ind in self.valid_start_inds[:: self.stride]:
                seg_id = int(self.segment_ids[start_ind])
                if not int(seg_id) in self.vid_lens:
                    frames = int(np.sum(self.segment_ids == seg_id))
                    self.vid_lens[int(seg_id / 2)] = frames
                    filtered_start_inds.append(start_ind)
            self.valid_start_inds = filtered_start_inds

    def __len__(self):
        return len(self.valid_start_inds)

    def __getitem__(self, idx):
        """
        Returns a flattened sequence of tokens representing `self.window_size` frames,
        spaced `self.stride` apart.
        """
        if self.use_action and self.use_text and self.hybrid_sample:
            strategy = random.random()
            if strategy <= 0.3:
                strategy = "action"
            elif strategy <= 0.6:
                strategy = "text"
            else:
                strategy = "both"
        elif self.use_action:
            strategy = "action"
        elif self.use_text:
            strategy = "text"
        else:
            strategy = "both"
        video_len = self.video_len if not self.rollout else self.vid_lens[idx]
        start_ind = self.valid_start_inds[idx]
        x = torch.from_numpy(
            (self.data[start_ind : start_ind + video_len + 1 : self.stride]).astype(
                np.int64
            )
        )  # (n,16,16)
        x = x.flatten()  # n*16*16
        if self.actions is not None:
            raw_action = self.actions[
                start_ind : start_ind + video_len + 1 : self.stride
            ]
            if self.noise_ratio is not None and self.noise_dim is not None:
                new_action = self.get_noised_action(raw_action)
                raw_action = new_action
            if self.use_bin_action:
                actions = self.get_bin_centers(raw_action)
            else:
                actions = raw_action
            actions = torch.from_numpy((actions).astype(np.float32))  # (n,28)
            actions = actions.flatten()
            random_indices = np.random.randint(0, len(self.actions) - video_len)
            if self.use_bin_action:
                random_actions = self.get_bin_centers(
                    self.actions[
                        random_indices : random_indices + video_len + 1 : self.stride
                    ]
                )
            else:
                random_actions = self.actions[
                    random_indices : random_indices + video_len + 1 : self.stride
                ]
            random_actions = torch.from_numpy(
                (random_actions).astype(np.float32)
            )  # (n,28)
            random_actions = random_actions.flatten()
        else:
            actions = None

        if strategy == "text":
            actions = torch.from_numpy(np.zeros(actions.shape).astype(np.float32))
            random_actions = torch.from_numpy(
                np.zeros(random_actions.shape).astype(np.float32)
            )
        attention_mask = torch.ones_like(x)
        if self.texts is not None:
            texts = torch.from_numpy(
                (
                    self.texts[start_ind : start_ind + video_len + 1 : self.stride]
                ).astype(np.float32)
            )  # (n,28)
            texts[1:] = 0
            texts = texts.flatten()
            random_indices = np.random.randint(0, len(self.texts) - video_len)
            random_texts = torch.from_numpy(
                (
                    self.texts[
                        random_indices : random_indices + video_len + 1 : self.stride
                    ]
                ).astype(np.float32)
            )  # (n,28)
            random_texts = random_texts.flatten()
        else:
            texts = None
        if strategy == "action":
            texts = torch.from_numpy(np.zeros(texts.shape).astype(np.float32))
            random_texts = torch.from_numpy(
                np.zeros(random_texts.shape).astype(np.float32)
            )

        if self.done is not None:
            done = torch.from_numpy(
                (self.done[start_ind : start_ind + video_len + 1 : self.stride]).astype(
                    np.float32
                )
            )  # (n,1)
            done = done.flatten()
        else:
            done = None

        return {
            "input_ids": x,
            "labels": x,
            "attention_mask": attention_mask,
            "actions": actions,
            "texts": texts,
            "random_actions": random_actions,
            "random_texts": random_texts,
            "done": done,
        }

    def get_noised_action(self, raw_action, noise_dim=None, noise_ratio=None):
        noise_dim = self.noise_dim if noise_dim is None else noise_dim
        noise_ratio = self.noise_ratio if noise_ratio is None else noise_ratio
        # TODO: action dim
        ori = raw_action
        norm_ori = (ori - self.dim_mins) / self.dim_ranges  # (7,)：归一化后的ori
        # 生成与归一化ori同形状的标准正态噪声
        eps_norm = np.random.randn(*norm_ori.shape)
        # 计算归一化噪声的L2范数（避免方向影响，只保留强度）
        n_eps_norm = np.linalg.norm(eps_norm, ord=2)
        # 计算归一化ori的L2范数（噪声强度与归一化ori成比例）
        n_ori_norm = np.linalg.norm(norm_ori, ord=2)
        # 防止噪声范数为0（极端情况）
        if n_eps_norm < 1e-12:
            eps_norm = np.random.randn(*norm_ori.shape)
            n_eps_norm = np.linalg.norm(eps_norm, ord=2)
        # 调整噪声强度：基于归一化后的ori尺度，乘以噪声比例
        eps_norm_scaled = eps_norm * (noise_ratio * n_ori_norm / n_eps_norm)
        # 步骤5：将加噪后的归一化值反归一化回原始数据范围
        # 反归一化公式：denorm_noise = norm_ori + eps_norm_scaled) * 维度范围 + 维度最小值
        norm_noise = norm_ori + eps_norm_scaled  # (7,)：归一化加噪后的值
        denorm_noise = (
            norm_noise * self.dim_ranges + self.dim_mins
        )  # (7,)：反归一化回原始尺度
        if noise_dim == -1:
            noise_dim = list(range(denorm_noise.shape[1]))
        elif noise_dim == 123:
            noise_dim = [0, 1, 2]
        elif noise_dim == 456:
            noise_dim = [3, 4, 5]
        return_action = ori.copy()
        return_action[:, noise_dim] = denorm_noise[:, noise_dim]
        return return_action

    def fit_bins(self, action_np):
        """
        重写：基于10%-90%区间划分bin，超出部分作为单独类别
        每个维度的bin结构：
        - bin0：所有 < q10 的样本
        - bin1 ~ binN：[q10, q90] 区间内的等距划分（N = num_bins_per_dim）
        - binN+1：所有 > q90 的样本
        """
        if len(action_np.shape) != 2:
            raise ValueError("输入action_np必须是(B, dim)形状的二维数组！")

        B, dim = action_np.shape
        self.dim_bins = []
        self.dim_bin_centers = []
        self.bin_counts = []
        self.sample_bin_indices = np.zeros((B, dim), dtype=int)

        for d in range(dim):
            dim_data = action_np[:, d]  # 第d维的所有样本

            # 步骤1：计算10%分位数（q10）和90%分位数（q90）
            q10 = np.percentile(dim_data, 10)  # 10%分位数（90%数据大于等于此值）
            q90 = np.percentile(dim_data, 90)  # 90%分位数（90%数据小于等于此值）
            # 处理q10 == q90的极端情况（如数据高度集中）
            if np.isclose(q10, q90):
                q10 = np.min(dim_data)
                q90 = np.max(dim_data)

            # 步骤2：生成bins边界
            # - 左侧边界：使用数据最小值（确保覆盖所有 < q10 的样本）
            # - 中间区间：[q10, q90] 内划分 num_bins_per_dim 个等距bin
            # - 右侧边界：使用数据最大值（确保覆盖所有 > q90 的样本）
            left_bound = np.min(dim_data)
            right_bound = np.max(dim_data)
            # 中间区间的bins（q10到q90）
            middle_bins = np.linspace(q10, q90, self.num_bins_per_dim + 1)
            # 完整bins：[left_bound] + middle_bins + [right_bound]
            # 注意：left_bound 可能 < q10，right_bound 可能 > q90
            bins = np.concatenate([[left_bound], middle_bins, [right_bound]])

            # 步骤3：计算每个bin的中心值
            bin_centers = []
            # bin0：< q10 的中心（left_bound 到 q10 的中点）
            bin0_center = (left_bound + q10) / 2.0
            bin_centers.append(bin0_center)
            # bin1 ~ binN：中间区间的bin中心
            middle_centers = (middle_bins[:-1] + middle_bins[1:]) / 2.0
            bin_centers.extend(middle_centers)
            # binN+1：> q90 的中心（q90 到 right_bound 的中点）
            binN1_center = (q90 + right_bound) / 2.0
            bin_centers.append(binN1_center)
            bin_centers = np.array(bin_centers)

            # 步骤4：统计每个bin的样本数量
            counts, _ = np.histogram(dim_data, bins=bins)

            # 步骤5：计算每个样本在该维度的bin索引
            sample_indices = np.digitize(dim_data, bins) - 1  # 转换为0开始的索引
            # 由于bins已包含min和max，索引不会超出范围，无需额外clip
            self.sample_bin_indices[:, d] = sample_indices

            # 保存当前维度的结果
            self.dim_bins.append(bins)
            self.dim_bin_centers.append(bin_centers)
            self.bin_counts.append(counts)

        # 转换为numpy数组方便索引
        self.dim_bins = np.array(self.dim_bins)
        self.dim_bin_centers = np.array(self.dim_bin_centers)
        self.bin_counts = np.array(self.bin_counts)  # 形状：[dim, num_bins_per_dim + 2]

    def get_bin_centers(self, action_np) -> np.ndarray:
        """
        为输入的(B, dim) tensor的每个元素，找到其所属bin的中心值

        参数:
            action_tensor: 形状为(B, dim)的tensor

        返回:
            center_values: 形状为(B, dim)的numpy数组，每个元素是对应位置的bin中心值
        """
        if self.dim_bin_centers is None:
            raise ValueError("请先调用fit_bins方法生成bin中心！")

        B, dim = action_np.shape
        center_values = np.zeros((B, dim), dtype=np.float32)  # 存储结果

        for d in range(dim):
            # 第d维的bins和bin中心
            bins = self.dim_bins[d]
            bin_centers = self.dim_bin_centers[d]

            # 步骤1：找到每个元素在第d维的bin索引
            dim_data = action_np[:, d]
            indices = np.digitize(dim_data, bins) - 1  # 转换为0开始的索引
            indices = np.clip(indices, 0, self.num_bins_per_dim - 1)  # 确保索引有效

            # 步骤2：根据索引获取对应的bin中心值
            center_values[:, d] = bin_centers[indices]

        return center_values


def get_maskgit_collator(config: GenieConfig):
    mask_token_id = config.image_vocab_size
    # h = w = math.isqrt(config.S)
    h = 2 * math.isqrt(config.S)
    w = math.isqrt(config.S)
    action_dim = config.action_dim
    text_dim = config.text_dim

    def collate_fn(features) -> dict[str, torch.Tensor]:
        # during training, map (z_0, z_1', z_2') -> (null, z_1, z_2)
        # (z_0, z_1') -> (null, z_1) is the diffusion operator on z_1' -> z_1
        input_ids = torch.stack([ex["input_ids"] for ex in features])
        actions = torch.stack([ex["actions"] for ex in features])
        texts = torch.stack([ex["texts"] for ex in features])
        done = torch.stack([ex["done"] for ex in features])
        device = input_ids.device

        x_THW = rearrange(
            input_ids, "b (t h w) -> b t h w", b=len(features), t=config.T, h=h, w=w
        )
        x_THWC = factorize_token_ids(
            x_THW, config.num_factored_vocabs, config.factored_vocab_size
        )
        labels = x_THW.clone()
        actions = rearrange(
            actions, "b (t d) -> b t d", b=len(features), t=config.T, d=action_dim
        )
        texts = rearrange(
            texts, "b (t d) -> b t d", b=len(features), t=config.T, d=text_dim
        )
        done = rearrange(done, "b (t d) -> b t d", b=len(features), t=config.T, d=1)
        # As done in Copilot-4D paper, add random noise sampled with a random rate between 0% and `config.max_corrupt_rate`
        r = torch.rand(x_THWC.size(), device=device)
        u01 = torch.rand((), device=device)
        random_patches_mask = r < config.max_corrupt_rate * u01
        random_values = torch.randint(
            low=0,
            high=config.factored_vocab_size,
            size=x_THWC.size(),
            dtype=torch.long,
            device=device,
        )
        x_THWC[random_patches_mask] = random_values[random_patches_mask]

        if random.random() < config.non_mlm_ratio:  # Closer to autoregressive inference
            # Leave frames [0, first_masked_frame) unmasked.
            first_masked_frame = random.randint(config.num_prompt_frames, config.T - 1)
            x_THWC_view = x_THWC[:, first_masked_frame:]

            # Arbitrary numbers here, but corrupting later frames more
            # since we likely have compounding errors.
            correct_rate = random.uniform(0.25, 1.0)
            for i in range(x_THWC_view.size(1)):
                correct_rate *= random.uniform(0.9, 1.0)
                r = torch.rand(
                    (len(features), h, w, config.num_factored_vocabs), device=device
                )
                random_patches_mask = r > correct_rate
                x_THWC_view[:, i][random_patches_mask] = random_values[
                    :, first_masked_frame + i
                ][random_patches_mask]
        else:  # Typical MLM masking
            first_masked_frame = 1

        mask = torch.zeros(1)
        c = 0
        while mask.max() == 0:  # We could get unlucky and mask no tokens?
            # per-minibatch, per-frame masking probability (could try variable masking rate from MUSE)
            mask_prob_T = cosine_schedule(
                torch.rand(len(features), config.T - first_masked_frame, 1, 1)
            )

            r = torch.rand_like(x_THW[:, first_masked_frame:], dtype=torch.float)
            mask = r < mask_prob_T
            c += 1

        if c > 1:
            print(f"Generated mask {c} > 1 times.")

        x_THW = unfactorize_token_ids(
            x_THWC, config.num_factored_vocabs, config.factored_vocab_size
        )
        x_THW[:, first_masked_frame:][mask] = mask_token_id

        return {
            "input_ids": rearrange(x_THW, "b t h w -> b (t h w)"),
            "labels": rearrange(labels, "b t h w -> b (t h w)"),
            "actions": rearrange(actions, "b t d -> b (t d)"),
            "texts": rearrange(texts, "b t d -> b (t d)"),
            "done": rearrange(done, "b t d -> b (t d)"),
        }

    return collate_fn
