"""
author: Yinuo
date: 2025-09-10

Environment wrapper for Maniskill environments with image observations.

Also return done=False since we do not terminate episode early.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_image_wrapper.py

"""

from transforms3d.euler import mat2euler
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import imageio
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from mani_skill.evaluation.policies.diffusion_policy.dp_modules.utils.math import (
    get_pose_from_rot_pos_batch,
)
from mani_skill.utils.geometry.rotation_conversions import matrix_to_euler_angles

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
        render_hw=(224, 224),
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

    def get_observation(self, raw_obs, output_dim=224):
        obs = {"rgb": None, "state": None}  # stack rgb if multiple cameras
        for key in self.obs_keys:
            if key in self.image_keys:
                image = raw_obs["sensor_data"][key]["rgb"]  # [B, H, W, C]
                image = image.permute(0, 3, 1, 2)  # [B, C, H, W]
                image = image.float()
                if not image.shape[-1] == output_dim:
                    image = F.interpolate(
                        input=image,
                        size=(output_dim, output_dim),  # 目标 (H, W)
                        mode="bilinear",  # 可选：'bilinear'（双线性）、'bicubic'（双三次，效果更好但稍慢）
                        align_corners=False,
                    )
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

    def action_transform(self, action):
        assert len(action.shape) == 2 and action.shape[1] == 10
        (B, action_dim) = action.shape

        def mat_6_to_mat(mat_6):
            assert mat_6.shape[-1] == 2
            assert mat_6.shape[-2] == 3

            mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0])  # [B, 3]
            mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1])  # [B, 3]
            z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1])  # [B, 3]
            z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
            mat = np.concatenate([mat_6, z_vec], axis=2)  # [B, 3, 3]

            return mat

        action_pos, action_mat_6, gripper_width = (
            action[:, :3],
            action[:, 3:9],
            action[:, 9:],
        )

        mat = mat_6_to_mat(action_mat_6.reshape(-1, 3, 2))

        action_euler = []
        for i in range(mat.shape[0]):
            action_euler.append(mat2euler(mat[i]))

        action_euler = np.stack(action_euler)

        action_7d = np.concatenate([action_pos, action_euler, gripper_width], axis=-1)

        return action_7d

    def action_transform_abs(self, action):
        assert len(action.shape) == 2 and action.shape[1] == 10
        (B, action_dim) = action.shape

        mat_6 = action[:, 3:9].reshape(action.shape[0], 3, 2)  # [B ,3, 2]
        mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0])  # [B, 3]
        mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1])  # [B, 3]
        z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1])  # [B, 3]
        z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
        mat = np.concatenate([mat_6, z_vec], axis=2)  # [B, 3, 3]
        pos = action[:, :3]  # [B, 3]
        gripper_width = action[:, -1, np.newaxis]  # [B, 1]

        init_to_desired_pose = get_pose_from_rot_pos_batch(
            mat, pos
        )  # for delta_action in base frame

        pose_action = np.concatenate(
            [
                init_to_desired_pose[:, :3, 3],
                matrix_to_euler_angles(
                    torch.from_numpy(init_to_desired_pose[:, :3, :3]), "XYZ"
                ).numpy(),
                gripper_width,
            ],
            axis=1,
        )  # [B, 7]

        return pose_action

    # def action_transform(self, action):
    #     assert len(action.shape) == 2 and action.shape[1] == 10
    #     (B, action_dim) = action.shape

    #     mat_6 = action[:, 3:9].reshape(action.shape[0], 3, 2)  # [B ,3, 2]
    #     mat_6[:, :, 0] = mat_6[:, :, 0] / np.linalg.norm(mat_6[:, :, 0])  # [B, 3]
    #     mat_6[:, :, 1] = mat_6[:, :, 1] / np.linalg.norm(mat_6[:, :, 1])  # [B, 3]
    #     z_vec = np.cross(mat_6[:, :, 0], mat_6[:, :, 1])  # [B, 3]
    #     z_vec = z_vec[:, :, np.newaxis]  # (B, 3, 1)
    #     mat = np.concatenate([mat_6, z_vec], axis=2)  # [B, 3, 3]
    #     pos = action[:, :3]  # [B, 3]
    #     gripper_width = action[:, -1, np.newaxis]  # [B, 1]

    #     # [b, 7] to [b, 4, 4] + [b, 10] --> [b, 7]
    #     def seven_dof_pose_with_euler_to_matrix(seven_dof_pose, euler_order="xyz"):
    #         """
    #         将包含旋转角(欧拉角)和gripper状态的7维姿态批量转换为4x4变换矩阵

    #         参数:
    #             seven_dof_pose: 形状为(B, 7)的数组，格式为[x, y, z, roll, pitch, yaw, gripper]
    #                             其中前3个是位置，中间3个是欧拉角，最后1个是gripper状态
    #             euler_order: 欧拉角旋转顺序，如'xyz', 'zyx'等，根据实际情况调整

    #         返回:
    #             形状为(B, 4, 4)的批量4x4变换矩阵
    #         """
    #         # 获取批次大小
    #         batch_size = seven_dof_pose.shape[0]

    #         # 提取位置分量 (B, 3)
    #         positions = seven_dof_pose[:, :3]

    #         # 提取欧拉角分量 (B, 3)
    #         eulers = seven_dof_pose[:, 3:6]

    #         # 创建批量4x4单位矩阵 (B, 4, 4)
    #         transform_matrices = np.eye(4, dtype=np.float32)[np.newaxis, ...].repeat(
    #             batch_size, axis=0
    #         )

    #         # 设置平移分量 (B, 3) -> (B, 3, 1) 并赋值
    #         transform_matrices[:, :3, 3] = positions

    #         # 将欧拉角批量转换为旋转矩阵 (B, 3, 3)
    #         rotations = R.from_euler(euler_order, eulers, degrees=False)
    #         rotation_matrices = rotations.as_matrix()  # 形状为 (B, 3, 3)

    #         # 设置旋转分量
    #         transform_matrices[:, :3, :3] = rotation_matrices

    #         return transform_matrices

    #     ##### Modified below #####
    #     # pose: Pose = self.env.agent.ee_pose_at_robot_base
    #     # self.pose_at_obs = pose.to_transformation_matrix().cpu().numpy()  # [b, 4, 4]
    #     ##### Modified above #####
    #     pose = self.env.action_buffer[-1].cpu().numpy()  # (B, 7)
    #     matrix = seven_dof_pose_with_euler_to_matrix(pose, euler_order="xyz")
    #     init_to_desired_pose = matrix @ get_pose_from_rot_pos_batch(
    #         mat, pos
    #     )  # for delta_action in base frame

    #     pose_action = np.concatenate(
    #         [
    #             init_to_desired_pose[:, :3, 3],
    #             matrix_to_euler_angles(
    #                 torch.from_numpy(init_to_desired_pose[:, :3, :3]), "XYZ"
    #             ).numpy(),
    #             gripper_width,
    #         ],
    #         axis=1,
    #     )  # [B, 7]

    #     return pose_action

    def step(self, action, timestep=None):
        if self.normalize:
            action = self.unnormalize_action(action)  # (B,action_dim)
        if self.cfg.abs_action:
            action = self.action_transform_abs(action)
        else:
            action = self.action_transform(action)
        raw_obs, reward, terminated, truncated, info = self.env.step(
            action, timestep
        )  # raw_obs: (B, obs_dim)
        obs = self.get_observation(raw_obs)  # obs: (B, obs_dim)
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
