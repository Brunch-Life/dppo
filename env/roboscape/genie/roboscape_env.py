import gym
from gym import spaces
from idna import decode
import numpy as np
from env.roboscape.genie.genie.st_mask_git import GenieConfig, STMaskGIT
from collections import deque
import torch
from env.roboscape.genie.magvit2.models.lfqgan import VQModel
from env.roboscape.genie.magvit2.config import VQConfig
import os
import random
import cv2
from torchvision import transforms
import gc
import torchvision.transforms.v2.functional as transforms_f
from einops import rearrange
import json
from env.roboscape.genie.data import RawTokenDataset
import lpips
from env.roboscape.genie.eval_utils import compute_lpips, decode_tokens
from env.roboscape.genie.visualize import decode_latents_wrapper
from PIL import Image
import torch.nn.functional as F
import imageio
from itertools import cycle, islice


class RoboScapeEnv(gym.Env):
    def __init__(
        self,
        task,
        wmconfig,
        goaler_config,
        wmckpt,
        goaler_ckpt,
        batch_size,
        total_steps,
        scene_id,
        data_dir,
        window_size=16,
        num_prompt_frames=8,
        latent_side_len=16,
        stride=4,
        force_action=False,
        vis_path=None,
        video_tokenizer_ckpt="/ML-vePFS/tangyinzhou/RoboScape-R/dppo/env/roboscape/genie/magvit2.ckpt",
        force_env_id=None,
    ):
        super(RoboScapeEnv, self).__init__()
        # 定义action空间，这里的action是绝对数值
        self.action_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.window_size = window_size
        self.num_prompt_frames = num_prompt_frames
        self.latent_side_len = latent_side_len
        self.task = task
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.config = GenieConfig.from_pretrained(wmconfig)
        self.goaler_config = GenieConfig.from_pretrained(goaler_config)
        self.config.use_mup = False  # Note: changing this may affect pre-trained model due to attn scaling
        self.config.T = self.window_size
        self.config.S = self.latent_side_len**2
        self.stride = stride
        self.force_action = force_action
        self.dataset = RawTokenDataset(
            data_dir,
            config=self.goaler_config,
            window_size=window_size,
            stride=stride,
            split="val",
            use_action=True,
            use_text=False,
            hybrid_sample=False,
            use_target_action=True,
        )
        # 始化世界模型
        self.world_model = STMaskGIT.from_pretrained(wmckpt).to("cuda")
        self.world_model.eval()

        self.goaler_model = STMaskGIT.from_pretrained(goaler_ckpt).to("cuda")
        self.goaler_model.eval()

        # 定义observation, action, reward的buffer
        self.observation_buffer = deque(maxlen=num_prompt_frames)
        self.action_buffer = deque(maxlen=num_prompt_frames)
        self.reward_buffer = deque(maxlen=num_prompt_frames)
        self.target_observations = []
        self.force_env_id = force_env_id

        # 定义视频tokenizer
        self.video_tokenizer = self.load_video_tokenizer(video_tokenizer_ckpt)
        # self.resize_transform = transforms.Resize((256, 256))
        self.resize_transform = transforms.Resize((224, 224))
        self.lpips_net = lpips.LPIPS(net="alex")
        self.vis_path = vis_path
        os.makedirs(self.vis_path, exist_ok=True)

    def load_video_tokenizer(self, video_tokenizer_ckpt):
        tokenizer = VQModel(
            VQConfig(),
            ckpt_path=video_tokenizer_ckpt,
        ).cuda()
        tokenizer.eval()
        return tokenizer

    def save_image(self, file_name, image):
        imageio.imwrite(file_name, image)

    def _generate_init_frame(self, mode=None):
        """
        从已有的数据集中随机选batchsize个首帧，并将其填充到num_prompts序列 (256x256x3 image).
        """
        # 处理observation
        # 计算数据集中的总片段数量（通过segment_ids去重统计）
        # segment_num = len(set([x for x in self.dataset.segment_ids.reshape(-1)]))

        # 初始化存储列表
        init_frames = []
        init_actions = []
        init_texts = []
        target_observations = []
        valid_start_inds = []

        # for index, seg_id in enumerate(self.dataset.cliped_segment_ids):
        #     if int(seg_id) >= len(valid_start_inds):
        #         valid_start_inds.append(index)

        # 选择有效的起始索引（从数据集中筛选）
        choose_idx = []  # 临时存储当前片段的有效索引
        current_id = 0  # 当前处理的片段ID
        ptr = 0  # 指针，用于遍历数据集

        # 遍历数据集的有效起始索引，筛选出每个片段的有效起始点
        while ptr < len(self.dataset.valid_start_inds):
            # 如果当前指针指向的索引属于当前片段ID
            if (
                self.dataset.segment_ids[self.dataset.valid_start_inds[ptr]]
                == current_id
            ):
                choose_idx.append(ptr)
                ptr += self.dataset.video_len
            # 如果当前指针指向的索引属于更大的片段ID（说明当前片段已处理完）
            elif (
                self.dataset.segment_ids[self.dataset.valid_start_inds[ptr]]
                > current_id
            ):
                # 调整指针位置，找到当前片段的最后一个有效索引
                ptr -= self.dataset.video_len
                while (
                    self.dataset.segment_ids[self.dataset.valid_start_inds[ptr]]
                    == current_id
                ):
                    ptr += self.dataset.stride
                ptr -= self.dataset.stride
                choose_idx.append(ptr)  # 记录当前片段的最后一个有效索引

                # 移动指针到下一个片段
                while (
                    self.dataset.segment_ids[self.dataset.valid_start_inds[ptr]]
                    == current_id
                ):
                    ptr += 1
                # 更新当前片段ID为新片段ID
                current_id = self.dataset.segment_ids[
                    self.dataset.valid_start_inds[ptr]
                ]
                valid_start_inds.append(choose_idx)  # 保存当前片段的有效索引列表
                choose_idx = []  # 重置选择列表
            else:
                ptr += 1

        idxs = (
            random.sample(
                valid_start_inds,
                self.batch_size,
            )
            if self.force_env_id is None
            else list(
                islice(
                    cycle([valid_start_inds[x] for x in self.force_env_id]),
                    self.batch_size,
                )
            )
        )
        # idxs = valid_start_inds[list(range(self.batch_size))]

        # change init scene
        # for i, idx in enumerate(idxs):
        #     if random.uniform(0, 1) > 0.5:
        #         if idx + 100 < len(valid_start_inds):
        #             idxs[i] = idx + 100
        #         else:
        #             idxs[i] = idx - 100

        gt_actions = []
        gt_observations = []
        for example in idxs:
            for i, example_ind in enumerate(example):
                example_THW = (
                    self.dataset[example_ind]["input_ids"]
                    .reshape(self.window_size, -1, 16)
                    .to("cuda")
                )
                example_action = (
                    self.dataset[example_ind]["actions"]
                    .reshape(self.window_size, 7)
                    .to("cuda")
                )
                example_text = (
                    self.dataset[example_ind]["texts"]
                    .reshape(self.window_size, 768)
                    .to("cuda")
                )
                if i == 0:
                    init_frames.append(example_THW)  # 编码后的首帧
                    # 处理action
                    init_actions.append(example_action)
                    init_texts.append(example_text)
                    gt_action = example_action
                    gt_obs = example_THW
                elif example[i] - example[i - 1] == self.dataset.video_len:
                    gt_action = torch.concat([gt_action, example_action], dim=0)
                    gt_obs = torch.concat([gt_obs, example_THW], dim=0)
                else:
                    gap = int((example[i] - example[i - 1]) / self.dataset.stride)
                    gt_action = torch.concat([gt_action, example_action[gap:]], dim=0)
                    gt_obs = torch.concat([gt_obs, example_THW[gap:]], dim=0)
                    gt_actions.append(gt_action)
                    gt_observations.append(gt_obs)

        max_len = max(t.size(0) for t in gt_actions)
        padded = [
            F.pad(t, (0, 0, 0, max_len - t.size(0)), value=-1) for t in gt_actions
        ]
        self.gt_actions = torch.stack(padded, dim=0)
        padded = [
            F.pad(
                t,
                (0, 0, 0, 0, 0, max_len - t.size(0)),
                value=self.world_model.mask_token_id,
            )
            for t in gt_observations
        ]
        self.gt_observations = torch.stack(padded, dim=0)

        init_frames = torch.stack(init_frames).permute(1, 0, 2, 3)  # T B H W
        init_actions = torch.stack(init_actions).permute(1, 0, 2)  # T B C
        init_texts = torch.stack(init_texts).permute(1, 0, 2)  # T B C

        self.observation_buffer = deque(
            init_frames[: self.num_prompt_frames], maxlen=self.num_prompt_frames
        )
        self.action_buffer = deque(
            init_actions[: self.num_prompt_frames], maxlen=self.num_prompt_frames
        )
        self.text_buffer = deque(
            init_texts[: self.num_prompt_frames], maxlen=self.num_prompt_frames
        )

        # init goal image for each env
        force_goal = True
        for idx in range(len(idxs)):
            target_observations.append(
                gt_observations[idx][-1]
                if force_goal
                else self.get_target_observations(idx, mode)
            )
        self.target_observations = torch.stack(target_observations).unsqueeze(1)
        self.target_observations = decode_tokens(
            self.target_observations.cpu(), decode_latents_wrapper()
        )  # B 1 C (2)H W
        # save target observations to vis_path
        for env_id, image in enumerate(self.target_observations.permute(0, 1, 3, 4, 2)):
            save_path = f"{self.vis_path}/env_{env_id}"
            os.makedirs(save_path, exist_ok=True)
            image = image[0].cpu().numpy()
            filename = f"{save_path}/target.png"
            self.save_image(filename, image)

        # get init frame for start time
        init_frames = decode_tokens(
            init_frames.cpu(), decode_latents_wrapper()
        )  # T B C (2)H W
        init_frame = init_frames[self.num_prompt_frames]  # B C (2)H W
        ####Test for now###
        for env_id, image in enumerate(init_frame.permute(0, 2, 3, 1)):
            save_path = f"{self.vis_path}/env_{env_id}"
            os.makedirs(save_path, exist_ok=True)
            image = image.cpu().numpy()
            filename = f"{save_path}/test_init.png"
            self.save_image(filename, image)
        output = torch.transpose(init_frame, 0, 1)
        return init_frame  # B, C, (2)H, W tensor

    def get_target_observations(self, idx, mode=None):
        # generate target observation with init frame and text instruction for a single env
        # max_frames = 200
        max_frames = 20
        prompt_len = len(self.observation_buffer)
        for slide_id, start_frame in enumerate(
            range(0, max_frames, self.window_size - 1)
        ):
            end_frame = min(start_frame + self.window_size, max_frames)
            if end_frame - start_frame < self.window_size:
                break
            # 截取当前窗口的输入
            if start_frame == 0:
                current_THW = (
                    torch.stack(list(self.observation_buffer))
                    .permute(1, 0, 2, 3)[idx]
                    .unsqueeze(0)
                )
                current_THW = torch.concat(
                    [
                        current_THW,
                        torch.zeros(
                            (
                                current_THW.size()[0],
                                self.window_size - current_THW.size()[1],
                                current_THW.size()[2],
                                current_THW.size()[3],
                            ),
                            dtype=torch.int64,
                        ).to("cuda"),
                    ],
                    dim=1,
                )
                prompt_THW = current_THW.clone()
                prompt_THW[:, prompt_len:] = self.goaler_model.mask_token_id
            else:
                prompt_THW = current_THW.clone()
                prompt_THW[:, :prompt_len] = last_frames[:, -prompt_len:]
                prompt_THW[:, prompt_len:] = self.goaler_model.mask_token_id

            current_action = (
                torch.stack(list(self.action_buffer)).permute(1, 0, 2)[idx].unsqueeze(0)
            )
            current_action = torch.concat(
                [
                    current_action,
                    torch.zeros(
                        (
                            current_action.size()[0],
                            self.window_size - current_action.size()[1],
                            current_action.size()[2],
                        )
                    ).to("cuda"),
                ],
                dim=1,
            )
            current_action = torch.zeros(current_action.size()).to("cuda")
            prompt_action = current_action.clone()

            current_text = (
                torch.stack(list(self.text_buffer)).permute(1, 0, 2)[idx].unsqueeze(0)
            )
            padding = current_text[:, [0]].expand(-1, self.window_size - prompt_len, -1)
            current_text = torch.cat([current_text, padding], dim=1)
            prompt_text = current_text.clone()
            # # 只取首帧作为提示
            # save observation for prompt frames
            image_save_dir = f"{self.vis_path}/{mode}/env_{idx}/init/slide_{slide_id}"
            os.makedirs(image_save_dir, exist_ok=True)
            for timestep in range(self.num_prompt_frames):
                obs = self._vis(
                    np.expand_dims(prompt_THW[:, timestep].cpu().numpy(), axis=1)
                )
                image_save_path = f"{image_save_dir}/prompt_{timestep}.png"
                obs[0][0].save(image_save_path, format="PNG")
            samples = []
            # generate pred frames
            for timestep in range(prompt_len, self.window_size):
                if (
                    end_frame - start_frame < self.window_size
                    and timestep >= end_frame - start_frame
                ):
                    break

                samples_HW, _, done_pred = self.goaler_model.maskgit_generate(
                    prompt_THW,
                    prompt_action,
                    prompt_text,
                    out_t=timestep,
                    maskgit_steps=2,
                    temperature=0,
                )

                samples.append(samples_HW)
                obs = self._vis(np.expand_dims(samples_HW.cpu().numpy(), axis=1))
                # save generated frame
                for image_id, image in enumerate(obs):
                    image_save_path = f"{image_save_dir}/{timestep}.png"
                    image[0].save(image_save_path, format="PNG")
                if done_pred[:, timestep, :].item() >= 0.9 or (
                    end_frame == max_frames and timestep == self.window_size
                ):
                    return samples_HW.clone()[0]
                prompt_THW[:, timestep] = samples_HW
            last_frames = prompt_THW.clone()
        return samples_HW.clone()[0]

    def _vis(self, all_video_data):
        def rescale_magvit_output(magvit_output):
            """
            [-1, 1] -> [0, 255]

            Important: clip to [0, 255]
            """
            rescaled_output = (magvit_output.detach().cpu() + 1) * 127.5
            clipped_output = torch.clamp(rescaled_output, 0, 255).to(dtype=torch.uint8)
            return clipped_output

        decoded_imgs = []
        for video_data in all_video_data:
            env_decoded_imgs = []
            batch = torch.from_numpy(video_data.astype(np.int64))
            # tokenizer = self.video_tokenizer.to(device="cuda", dtype=torch.bfloat16)
            tokenizer = self.video_tokenizer.to(device="cuda")
            if tokenizer.use_ema:
                with tokenizer.ema_scope():
                    quant = tokenizer.quantize.get_codebook_entry(
                        rearrange(batch, "b h w -> b (h w)"),
                        bhwc=batch.shape + (tokenizer.quantize.codebook_dim,),
                    ).flip(1)
                    env_decoded_imgs.append(
                        (
                            (
                                rescale_magvit_output(
                                    tokenizer.decode(quant.to(device="cuda"))
                                )
                            )
                        )
                    )
            decoded_imgs.append(
                [transforms_f.to_pil_image(img) for img in torch.cat(env_decoded_imgs)]
            )
        return decoded_imgs

    def reset(self):
        """
        Reset the environment to the initial state.
        Returns:
            observation (numpy array): The initial frame.  [B, H, W, C]
                ["sensor_data"]["3rd_view_camera"]["rgb"] and ["sensor_data"]["hand_camera"]["rgb"] is not None
                "agent", "extra", "sensor_param", "sensor_data"
            info (dict): Additional information.
        """
        # TODO: Modify data type and shape
        # get init frame for each env
        current_observation_token = self._generate_init_frame()  # N C (2)H W tensor
        # save init frame for each env
        H = current_observation_token.shape[2] // 2

        return_obs = {
            "agent": None,
            "extra": None,
            "sensor_param": None,
            "sensor_data": {
                "3rd_view_camera": {
                    "rgb": (current_observation_token[:, :, :H, :] / 255)
                    .to(torch.float32)
                    .permute(0, 2, 3, 1)
                },
                "hand_camera": {
                    "rgb": (current_observation_token[:, :, H:, :] / 255)
                    .to(torch.float32)
                    .permute(0, 2, 3, 1)
                },
            },
        }
        return return_obs, {}

    def step(self, action, act_step=None):
        """
        Take a step in the environment.
        Args:
            action (ndarray float32): The action vector.
        Returns:
            observation (torch.float32): The next frame.        [B, H, W, C]
                ["sensor_data"]["3rd_view_camera"]["rgb"] and ["sensor_data"]["hand_camera"]["rgb"] is not None
                "agent", "extra", "sensor_param", "sensor_data"
            reward (torch.float32): The reward for the action.  [B,]
            terminated (torch.bool): Done for now               [B,]
            truncated (torch.bool): Done for now                [B,]
            info (dict): Additional information.                None
        """
        force_action = False
        if force_action and act_step is not None:
            action = (
                self.gt_actions[:, act_step + self.num_prompt_frames, :].cpu().numpy()
            )

        # TODO: Modify below
        # step for each env
        next_obs, reward, done_pred = self._generate_next_frame(action)
        self.observation_buffer.append(next_obs)
        # TODO: Modify above

        # for image_id, image in enumerate(self.current_observation):
        #     image_save_dir = f"{self.vis_path}/{mode}/env_{image_id}"
        #     os.makedirs(image_save_dir, exist_ok=True)
        #     image[0].save(
        #         f"{image_save_dir}/step_{step+self.num_prompt_frames}.png", format="PNG"
        #     )

        # TODO: Modify below
        self.reward_buffer.append(reward.reshape(-1))
        # TODO: Modify above

        # Return the new state, reward, done flag, and additional info
        pred_frames = decode_tokens(
            next_obs.unsqueeze(1).cpu(), decode_latents_wrapper()
        )
        pred_frames = torch.transpose(pred_frames, 0, 1)  # T B C H W
        H = pred_frames.shape[3] // 2
        obs = {
            "agent": None,
            "extra": None,
            "sensor_param": None,
            "sensor_data": {
                "3rd_view_camera": {
                    "rgb": (pred_frames[-1, :, :, :H, :] / 255)
                    .to(torch.float32)
                    .permute(0, 2, 3, 1)
                },
                "hand_camera": {
                    "rgb": (pred_frames[-1, :, :, H:, :] / 255)
                    .to(torch.float32)
                    .permute(0, 2, 3, 1)
                },
            },
        }  # [B, H, W, C]
        reward = torch.from_numpy(reward).to(torch.float32)
        done = (done_pred[:, -1, :].reshape(-1) > 0.9).to(torch.bool)
        terminated = done
        truncated = done
        info = {}
        ###Test for now###
        env_num = pred_frames.shape[1]
        for env_id in range(env_num):
            image = pred_frames[0][env_id]  # H W C
            save_dir = f"{self.vis_path}/env_{env_id}"
            os.makedirs(save_dir, exist_ok=True)
            imageio.imwrite(
                f"{save_dir}/test.png",
                image.permute(1, 2, 0).cpu().numpy(),
            )
        return obs, reward, terminated, truncated, info

    def _generate_next_frame(self, action):
        """
        基于当前时刻的observation和action生成下一时刻的observation和reward
        """
        action = torch.from_numpy(action).to("cuda").to(torch.float32)
        prompt_THW = torch.stack(list(self.observation_buffer))
        prompt_THW = rearrange(prompt_THW, "T B H W -> B T H W")
        prompt_THW = torch.concatenate(
            (
                prompt_THW,
                torch.full(
                    (
                        prompt_THW.shape[0],
                        self.window_size - self.num_prompt_frames,
                        prompt_THW.shape[2],
                        prompt_THW.shape[3],
                    ),
                    self.world_model.mask_token_id,
                ).to("cuda"),
            ),
            dim=1,
        )
        prompt_action = torch.stack(list(self.action_buffer))
        prompt_action = rearrange(prompt_action, "T B D -> B T D")
        prompt_action = torch.concatenate(
            (
                prompt_action,
                action.unsqueeze(1),
                torch.zeros(
                    self.batch_size,
                    self.window_size - self.num_prompt_frames - 1,
                    prompt_action.shape[-1],
                ).to("cuda"),
            ),
            dim=1,
        )
        prompt_text = torch.stack(list(self.text_buffer))
        prompt_text = rearrange(prompt_text, "T B D -> B T D")
        prompt_text = torch.concatenate(
            (
                prompt_text,
                torch.zeros(
                    self.batch_size,
                    self.window_size - self.num_prompt_frames,
                    prompt_text.shape[-1],
                ).to("cuda"),
            ),
            dim=1,
        )
        samples_HW, _, done_pred = self.world_model.maskgit_generate(
            prompt_THW,
            prompt_action,
            prompt_text,
            out_t=prompt_THW.shape[1] - 1,
            maskgit_steps=2,
            temperature=0,
        )
        done = done_pred[:, self.num_prompt_frames]
        reward = self.cal_reward(samples_HW.unsqueeze(1), done)
        self.action_buffer.append(action.to(self.action_buffer[0].dtype).to("cuda"))
        gt_action = [x for x in self.gt_actions.permute(1, 0, 2)][15]
        # 计算gt_action和action的距离
        action_distance = torch.norm(gt_action - action, dim=1)
        return samples_HW, reward, done_pred

    def cal_reward(self, samples_HW, done):
        decoded_pred = decode_tokens(samples_HW.cpu(), decode_latents_wrapper())
        dense_reward = compute_lpips(
            decoded_pred, self.target_observations, self.lpips_net
        )
        sparse_reward = [10 if x >= 0.9 else 0 for x in done[:, 0]]
        return np.array(dense_reward).reshape(-1) + np.array(sparse_reward).reshape(-1)

    def _is_goal_reached(self, reward_buffer):
        """
        Check if the goal is reached based on the current frame.
        """
        if len(reward_buffer) < 8:
            return False
        std_dev = np.std(np.array(reward_buffer))
        return std_dev <= 0.1

    def _is_truncated(self, reward_buffer):
        """
        Check if the the observation is too far away from success.
        """
        if len(reward_buffer) < 8:
            return False
        return all(
            reward_buffer[i] > reward_buffer[i + 1]
            for i in range(len(reward_buffer) - 1)
        )

    def close(self):
        """
        Close the environment.
        """
        pass  # No resources to clean up i

    def render(self, mode="rgb_array"):
        """
        Receives:
        mode choices: ["rgb_array", "image"]
        Returns:
        rgb_array: ndarray of shape [B C H W] for float32
        or
        image: PIL Image objects
        """
        if mode == "rgb_array":
            obs = torch.stack(list(self.observation_buffer), dim=1)
            obs = decode_tokens(obs.cpu(), decode_latents_wrapper())
            obs = obs.permute(1, 0, 3, 4, 2)
        else:
            obs = None
        return obs
