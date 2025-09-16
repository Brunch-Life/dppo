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

from mani_skill.utils.structs.pose import Pose
from mani_skill.evaluation.policies.diffusion_policy.dp_modules.utils.math import get_pose_from_rot_pos_batch
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles
from torchvision import transforms



class ManiskillImageWrapper(gym.Env):
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
        self.record_video = False
        self.clamp_obs = clamp_obs
        self.cfg = cfg
        
        self.debug_cnt = 0
        # set up normalization
        self.normalize = normalization_path is not None
        if self.normalize:
            self.NORMALIZE_PARAMS = np.load(normalization_path, allow_pickle=True)

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
        
        # 480x640 -> 224x224
        from torch import nn
        ratio = 0.95
        original_size = (480, 640)
        self.transformations = [
            # transforms.RandomCrop(size=[int(original_size[0] * ratio), int(original_size[1] * ratio)]),
            transforms.Resize((224, 224), antialias=True),
        ]
        self.transformations = nn.Sequential(*self.transformations)

        self.record_third = []
        self.record_wrist = []

        self.pos_at_obs = np.zeros((self.num_envs, 4, 4))
        self.pos_at_obs_new = np.zeros((self.num_envs, 4, 4))


    def concat_images(images_third, images_wrist):
        images = []
        for i in range(len(images_third)):
            images.append(np.concatenate([images_third[i], images_wrist[i]], axis=1))
        return images


    def normalize_obs(self, obs):
        obs = 2 * (
            (obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5
        )  # -> [-1, 1] (B,obs_dim)
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = np.asarray(action)
        action = action * self.pose_gripper_scale[None, :] + self.pose_gripper_mean[None, :]
        return action

    def action_transform(self, action):
        assert len(action.shape) == 2 and action.shape[1] == 10
        (B,action_dim) = action.shape

        mat_6 = action[:,3:9].reshape(action.shape[0],3,2) # [B ,3, 2]
        mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0]) # [B, 3]
        mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1]) # [B, 3]
        z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1]) # [B, 3]
        z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
        mat = np.concatenate([mat_6, z_vec], axis=2) # [B, 3, 3]
        pos = action[:, :3] # [B, 3]
        gripper_width = action[:, -1, np.newaxis] # [B, 1]


        init_to_desired_pose = self.pos_at_obs @ get_pose_from_rot_pos_batch(mat, pos) # for delta_action in base frame 

        pose_action = np.concatenate(
            [
                init_to_desired_pose[:, :3, 3],
                matrix_to_euler_angles(torch.from_numpy(init_to_desired_pose[:, :3, :3]),"XYZ").numpy(),
                gripper_width
            ],
            axis=1) # [B, 7]

        return pose_action

    def get_observation(self, raw_obs):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        
        image_list = []
        for key in self.obs_keys:
            image_list.append(raw_obs['sensor_data'][key]['rgb'].to(torch.uint8).cpu().numpy())
        
        image_list = np.asarray(image_list)  # (M, B, H, W, C)
        M, B, H, W, C = image_list.shape

        image_data = torch.from_numpy(image_list)  # (M, B, H, W, C)
        image_data = image_data.permute(1, 0, 4, 2, 3)  # (B, M, C, H, W)
        image_data = image_data.reshape(B * M, C, H, W)
        try:
            for transform in self.transformations:
                image_data = transform(image_data)
                # print(front.shape, goal.shape)
        except Exception as e:
            print(e)

        image_data = image_data.view(B, M, C, 224, 224)

        # imageio.imwrite("test_obs_3rd.png", image_data[0][0].permute(1, 2, 0).cpu().numpy())
        # imageio.imwrite("test_obs_wrist.png", image_data[0][1].permute(1, 2, 0).cpu().numpy())
        image_data = image_data.float() / 255.0  # [0,1]
        for key in self.obs_keys:
            if key in self.image_keys:
                pass
                # image = raw_obs["sensor_data"][key]["rgb"]  # [B, H, W, C]
                # image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
                # image = image.float()
                # image = self.transformations(image)
                # if obs["rgb"] is None:
                #     obs["rgb"] = image
                # else:
                #     obs["rgb"] = torch.cat([obs["rgb"], image], dim=1)  # (B, C, H, W)
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
        obs["rgb"] = torch.tensor(image_data).float().to(raw_obs['sensor_data'][self.image_keys[0]]['rgb'].device) # obs["rgb"].float()
        if obs['state'] is None:
            obs['state'] = torch.zeros(obs['rgb'].shape[0], 10).to(obs['rgb'].device)
        
        
        # imageio.imwrite("test_obs_3rd.png", obs["rgb"][0, :3].permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        # imageio.imwrite("test_obs_hand.png", obs["rgb"][0, 3:].permute(1, 2, 0).cpu().numpy().astype(np.uint8)) 

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

        if self.record_video:
            self.save_video(options["video_path"])


        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Call reset
        new_seed = options.get(
            "seed", None
        )  # used to set all environments to specified seeds
        env_reset_options = {
            "reconfigure": True,
            "episode_id": torch.arange(self.num_envs)+self.debug_cnt*100,
        }
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset(seed=0, options=env_reset_options)
                self.has_reset_before = True

            # always reset to the same state to be compatible with gym
            raw_obs, info = self.env.reset_to(
                {"states": self.init_state}
            )  # raw_obs: (B, obs_dim)
        elif new_seed is not None:
            self.seed(seed=new_seed)
            raw_obs, info = self.env.reset(seed=new_seed, options=env_reset_options)  # raw_obs: (B, obs_dim)
        else:
            # random reset
            raw_obs, info = self.env.reset(seed=new_seed, options=env_reset_options)  # raw_obs: (B, obs_dim)

        self.debug_cnt += 1

        pose:Pose = self.env.agent.ee_pose_at_robot_base
        self.pos_at_obs_new = pose.to_transformation_matrix().cpu().numpy()

        return self.get_observation(raw_obs)

    def step(self, action):
        (B,action_dim) = action.shape
        # print("raw_action:", action)
        if self.normalize:
            # print("IN?")
            action = self.unnormalize_action(action)
            # print("unnormalize action:", action)

        # print("unprocess action?:", action[0])
        action = self.action_transform(action)
        # print("action_transform:", action)

        # print("action?:", action[0])

        # action_dict = np.load("./debug/action.npy", allow_pickle=True).item()
        # action = action_dict["total_action_list"][self.debug_cnt]
        # self.debug_cnt += 1
        # action = torch.tensor(action).to(torch.float32).to(self.env.device).repeat(self.num_envs, 1)
        
        raw_obs, reward, terminated, truncated, info = self.env.step(
            action
        )  # raw_obs: (B, obs_dim)
        # print("raw_obs:", raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].shape, raw_obs["sensor_data"]["hand_camera"]["rgb"].shape)
        # exit(0)
        obs = self.get_observation(raw_obs)  # obs: (B, obs_dim)

        # render if specified
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img[0].cpu().numpy())

        # if self.save_video:
        #     record_third = obs["rgb"][0, :3].permute(1, 2, 0).cpu().numpy()
        #     record_wrist = obs["rgb"][0, 3:].permute(1, 2, 0).cpu().numpy()
        #     self.record_third.append(record_third)
        #     self.record_wrist.append(record_wrist)

        pose:Pose = self.env.agent.ee_pose_at_robot_base
        self.pos_at_obs_new = pose.to_transformation_matrix().cpu().numpy()

        return obs, reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        return self.env.render()


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    cfg = OmegaConf.load("cfg/maniskill/finetune/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]

    import matplotlib.pyplot as plt
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.tasks.tabletop import TabletopPickPlaceEnv

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

    wrapper = ManiskillImageWrapper(
        env=env,
        shape_meta=shape_meta,
        image_keys=["3rd_view_camera", "hand_camera"],
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
