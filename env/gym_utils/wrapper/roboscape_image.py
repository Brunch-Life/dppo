"""
author: Yinuo
date: 2025-09-10

Environment wrapper for Maniskill environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio
import torch

# import os
# import sys
from datetime import datetime


class RoboScapeImageWrapper(gym.Env):
    def __init__(
        self,
        env,
        shape_meta: dict,
        normalization_path=None,
        low_dim_keys=[],
        image_keys=[
            "3rd_view_camera",
            "hand_camera",
        ],
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="3rd_view_camera",
        cfg=None,
    ):
        self.env = env
        self.num_envs = cfg.env.n_envs
        self.init_state = init_state
        self.has_reset_before = False
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.cfg = cfg
        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            self.NORMALIZE_PARAMS = np.load(normalization_path)

            self.pose_gripper_mean = np.concatenate(
                [
                    self.NORMALIZE_PARAMS[key]["mean"]
                    for key in ["pose", "gripper_width"]
                ]
            )
            self.pose_gripper_scale = np.concatenate(
                [
                    self.NORMALIZE_PARAMS[key]["scale"]
                    for key in ["pose", "gripper_width"]
                ]
            )
            self.proprio_gripper_mean = np.concatenate(
                [
                    self.NORMALIZE_PARAMS[key]["mean"]
                    for key in ["proprio_state", "gripper_width"]
                ]
            )
            self.proprio_gripper_scale = np.concatenate(
                [
                    self.NORMALIZE_PARAMS[key]["scale"]
                    for key in ["proprio_state", "gripper_width"]
                ]
            )

        # setup spaces
        low = np.full((self.num_envs, 7), fill_value=-1.0, dtype=np.float32)
        high = np.full((self.num_envs, 7), fill_value=1.0, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float32,
        )
        self.low_dim_keys = low_dim_keys
        self.image_keys = image_keys
        self.obs_keys = low_dim_keys + image_keys
        observation_space = spaces.Dict()
        for key, value in shape_meta["obs"].items():
            shape = (self.num_envs,) + value["shape"]
            if key.endswith("rgb"):
                min_value, max_value = 0, 1
            elif key.endswith("state"):
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32,
            )
            observation_space[key] = this_space
        self.observation_space = observation_space

    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1] (B,obs_dim)
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = np.asarray(action)  # (B,action_dim)
        action = (
            action * self.pose_gripper_scale[None, :] + self.pose_gripper_mean[None, :]
        )
        return action

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                image = raw_obs["sensor_data"][key]["rgb"]  # [B, H, W, C]
                image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
                image = image.float()
                if obs["rgb"] is None:
                    obs["rgb"] = image
                else:
                    obs["rgb"] = torch.cat([obs["rgb"], image], dim=1)  # (B, C, H, W)
            else:
                if obs["state"] is None:
                    obs["state"] = raw_obs[key].float()
                else:
                    obs["state"] = torch.cat(
                        [obs["state"], raw_obs[key].float()], dim=1
                    )
        # if self.normalize:
        #     obs["state"] = self.normalize_obs(obs["state"])
        # obs["rgb"] *= 255  # [0, 1] -> [0, 255], in float64
        obs["rgb"] = obs["rgb"].float()

        return obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed"""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs, info = self.env.reset_to(
                {"states": self.init_state}
            )  # raw_obs: (B, obs_dim)
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs, info = self.env.reset()  # raw_obs: (B, obs_dim)
        else:
            # random reset
            raw_obs, info = self.env.reset()  # raw_obs: (B, obs_dim)
        return self.get_observation(raw_obs)

    def step(self, action, timestep=None):
        if self.normalize:
            action = self.unnormalize_action(action)  # (B,action_dim)
        raw_obs, reward, terminated, truncated, info = self.env.step(
            action, timestep
        )  # raw_obs: (B, obs_dim)
        obs = self.get_observation(raw_obs)  # obs: (B, obs_dim)
        ###Test for now###
        env_num = obs["rgb"].shape[0]
        for env_id in range(env_num):
            image = obs["rgb"][env_id].permute(1, 2, 0)  # H W C
            save_dir = f"{self.cfg.env.vis_path}/env_{env_id}"
            import os
            import imageio

            cam_3rd = image[:, :, :3]
            cam_wrist = image[:, :, 3:]
            image = (torch.concat([cam_3rd, cam_wrist], dim=0) * 255).to(torch.uint8)
            os.makedirs(save_dir, exist_ok=True)
            imageio.imwrite(
                f"{save_dir}/test.png",
                image.cpu().numpy(),
            )
        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render()


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    cfg = OmegaConf.load(
        "/iag_ad_01/ad/tangyinzhou/tyz/WM4RL_DPPO/dppo/cfg/roboscape/finetune/ft_ppo_diffusion_mlp_img.yaml"
    )
    shape_meta = cfg["shape_meta"]

    import matplotlib.pyplot as plt

    # from mani_skill.envs.sapien_env import BaseEnv
    # from mani_skill.envs.tasks.tabletop import TabletopPickPlaceEnv
    from roboscape.genie.roboscape_env import RoboScapeEnv

    wrappers = cfg.env.wrappers
    obs_modality_dict = {
        "low_dim": (
            wrappers.maniskill_image.low_dim_keys
            if "maniskill_image" in wrappers
            else wrappers.maniskill_lowdim.low_dim_keys
        ),
        "rgb": (
            wrappers.maniskill_image.image_keys
            if "maniskill_image" in wrappers
            else None
        ),
    }
    if obs_modality_dict["rgb"] is None:
        obs_modality_dict.pop("rgb")
    # ObsUtils.initialize_obs_modality_mapping_from_dict(obs_modality_dict)

    if "TabletopPickPlaceEnv" in cfg.env_name:
        env_kwargs = dict(
            num_envs=cfg.env.n_envs,
            obs_mode="rgb+segmentation",
            control_mode="pd_ee_delta_pose",
            sim_backend="gpu",
            sim_config={
                "sim_freq": 1000,
                "control_freq": 25,
            },
            max_episode_steps=300,
            sensor_configs={"shader_pack": "default"},
            is_table_green=False,
            render_mode="rgb_array",
        )
        if cfg.env.robot_uids is not None:
            env_kwargs["robot_uids"] = tuple(cfg.env.robot_uids.split(","))
        if cfg.env_name == "TabletopPickPlaceEnv-v1":
            env_kwargs["object_name"] = cfg.env.object_name
            env_kwargs["container_name"] = cfg.env.container_name
        elif cfg.env_name == "TabletopPickEnv-v1":
            env_kwargs["object_name"] = cfg.env.object_name
        env: BaseEnv = gym.make(
            cfg.env_name,
            **env_kwargs,
        )
    else:
        env_kwargs = dict(
            task=cfg.env.task,
            wmconfig=cfg.env.wmconfig,
            goaler_config=cfg.env.goaler_config,
            wmckpt=cfg.env.wmckpt,
            goaler_ckpt=cfg.env.goaler_ckpt,
            batch_size=cfg.env.n_envs,
            total_steps=cfg.env.total_steps,
            scene_id=cfg.env.scene_id,
            data_dir=cfg.env.data_dir,
            vis_path=cfg.env.vis_path
            + f"/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            force_action=cfg.env.force_action,
            num_prompt_frames=cfg.env.num_prompt_frames,
        )
        if cfg.env_name == "RoboScape":
            env = RoboScapeEnv(**env_kwargs)

    wrapper = ManiskillImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["3rd_view_camera", "hand_camera"],
        cfg=cfg,
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    action = wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    print(obs.keys())
    print(obs["rgb"].shape)
    image_3rd = obs["rgb"][0, :3].permute(1, 2, 0).cpu().numpy()
    image_hand = obs["rgb"][0, 3:].permute(1, 2, 0).cpu().numpy()
    image_3rd = (image_3rd * 255).astype(np.uint8)
    image_hand = (image_hand * 255).astype(np.uint8)
    imageio.imwrite("test_obs_3rd.png", image_3rd)
    imageio.imwrite("test_obs_hand.png", image_hand)
    img = wrapper.render()
    img = img[0].cpu().numpy()
    wrapper.close()
    plt.imshow(img)
    plt.savefig("test.png")
