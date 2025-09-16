"""
Multi-step wrapper. Allow executing multiple environmnt steps. Returns stacked observation and optionally stacked previous action.

Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/gym_util/multistep_wrapper.py

TODO: allow cond_steps != img_cond_steps (should be implemented in training scripts, not here)
"""

import gymnasium as gym
from typing import Optional
from gymnasium import spaces
import numpy as np
from collections import defaultdict, deque
import torch

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x, axis=0), n, axis=0)


def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype,
    )


def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f"Unsupported space type {type(space)}")


def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return x[-n:]


def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result


def aggregate(data, method="max"):
    # Convert list to tensor if needed
    if isinstance(data, list):
        data = torch.stack(data)
    
    if method == "max":
        # equivalent to any
        return torch.max(data, dim=0).values # (B,)
    elif method == "min":
        # equivalent to all
        return torch.min(data, dim=0).values # (B,)
    elif method == "mean":
        return torch.mean(data, dim=0) # (B,)
    elif method == "sum":
        return torch.sum(data, dim=0)
    else:
        raise NotImplementedError()


def stack_last_n_obs(all_obs, n_steps):
    """Apply padding"""
    if all_obs[-1] is None: # for no obs_state
        return None
    assert len(all_obs) > 0
    assert len(all_obs) >= n_steps

    start_idx = - n_steps
    stack_obs = torch.stack(all_obs[start_idx:]).transpose(0, 1)  # (B, n_steps, C, H, W)
    assert stack_obs.shape[0] == all_obs[-1].shape[0]
    assert stack_obs.shape[1] == n_steps
    assert stack_obs.shape[2:] == all_obs[-1].shape[1:]
    result = stack_obs
    return result


class MultiStep(gym.Wrapper):

    def __init__(
        self,
        env,
        n_obs_steps=1,
        n_action_steps=1,
        max_episode_steps=None,
        reward_agg_method="sum",  # never use other types
        prev_action=True,
        reset_within_step=True,
        pass_full_observations=False,
        verbose=False,
        **kwargs,
    ):
        super().__init__(env)
        self._single_action_space = env.action_space
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.prev_action = prev_action
        self.reset_within_step = reset_within_step
        self.pass_full_observations = pass_full_observations
        self.verbose = verbose

    def reset(
        self,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: dict = {},
        **kwargs,
    ):
        """Resets the environment."""
        obs = self.env.reset(options=options) # obs: (B, obs_dim)
        # for key in obs.keys():
        #     obs[key] = obs[key].unsqueeze(1) # (B, 1, obs_dim)
        self.obs = deque([obs], maxlen=max(self.n_obs_steps + 1, self.n_action_steps))
        if self.prev_action:
            self.action = deque(
                [self._single_action_space.sample()], maxlen=self.n_obs_steps
            )
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda: deque(maxlen=self.n_obs_steps + 1))
        obs = self._get_obs(self.n_obs_steps) # obs: (B, n_obs_steps, obs_dim)

        return obs

    def step(self, action):
        """
        actions: (B, n_action_steps,) + action_shape
        """
        if action.ndim == 1:  # in case action_steps = 1
            action = action[None]
        self.env.pos_at_obs = self.env.pos_at_obs_new
        for act_step, act in enumerate(action.transpose(1, 0, 2)):
            # done does not differentiate terminal and truncation
            observation, reward, terminated, truncated, info = self.env.step(act) # act: (B, action_dim)
            terminated = truncated.clone()

            reward = torch.where(info["success"], 1, 0)

            self.obs.append(observation) # self.obs: list: n_steps of (B, obs_dim)
            self.action.append(act) # self.action: list: n_steps of (B, action_dim)
            self.reward.append(reward) # self.reward: list: n_steps of (B,)
            
            done = torch.logical_or(truncated, terminated) # done: (B,)
            self.done.append(done) # self.done: list: n_steps of (B,)
            self._add_info(info)
        observation = self._get_obs(self.n_obs_steps) # observation: (B, n_obs_steps, obs_dim)
        reward = aggregate(self.reward, self.reward_agg_method) # reward: (B,)
        done = aggregate(self.done, "max") # done: (B,)
        info = dict_take_last_n(self.info, self.n_obs_steps)
        if self.pass_full_observations:
            info["full_obs"] = self._get_obs(act_step + 1) # full_obs: (B, n_obs_steps, obs_dim)

        # In mujoco case, done can happen within the loop above
        if self.reset_within_step and torch.any(truncated).item():
            assert torch.all(truncated), "The envs should be truncated at the same time"
            env_ind = torch.where(truncated)[0] # env_ind: eg. [0, 1, 2]
            # need to save old observation in the case of truncation only, for bootstrapping
            if torch.any(truncated):
                if isinstance(observation, dict):
                    for key in observation.keys():
                        info["final_obs"] = observation[key][torch.where(truncated)] # final_obs: (B, n_obs_steps, obs_dim)
                else:
                    info["final_obs"] = observation[torch.where(truncated)] # final_obs: (B, n_obs_steps, obs_dim)

            # reset
            options = {}
            options["env_idx"] = env_ind # env_ind: eg. [0, 1, 2]

            observation = ( # observation: (B, n_obs_steps, obs_dim)
                self.reset(options=options)
            ) 
            self.verbose and print(f"Reset env{env_ind} within wrapper.") # print(f"Reset env{env_ind} within wrapper.")

        # reset reward and done for next step
        self.reward = list()
        self.done = list()
        return observation, reward, terminated, truncated, info # observation: (B, n_obs_steps, obs_dim), reward: (B,), terminated: (B,), truncated: (B,), info: dict

    def _get_obs(self, n_steps=1):
        assert len(self.obs) > 0
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs([obs[key] for obs in self.obs], n_steps)
            return result # result: (B, n_steps, obs_dim)
        else:
            raise RuntimeError("Unsupported space type")

    def get_prev_action(self, n_steps=None):
        if n_steps is None:
            n_steps = self.n_obs_steps - 1  # exclude current step
        assert len(self.action) > 0
        return stack_last_n_obs(self.action, n_steps) # (B, n_steps, action_dim)

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value) # self.info: (B, n_steps, info_dim)

    def render(self, **kwargs):
        """Not the best design"""
        return self.env.render(**kwargs) 


if __name__ == "__main__":
    import os
    from omegaconf import OmegaConf
    import json

    import imageio

    cfg = OmegaConf.load("cfg/maniskill/finetune/ft_ppo_diffusion_mlp_img.yaml")
    shape_meta = cfg["shape_meta"]


    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.envs.tasks.tabletop import TabletopPickPlaceEnv

    from env.gym_utils.wrapper.maniskill_image import ManiskillImageWrapper



    import matplotlib.pyplot as plt


    wrappers = cfg.env.wrappers

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
        is_table_green = False,
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

    wrapper = MultiStep(
        env=ManiskillImageWrapper(
            env=env,
            shape_meta=shape_meta,
            image_keys=["3rd_view_camera", "hand_camera"],
            cfg=cfg,
        ),
        n_obs_steps=10,
        n_action_steps=10,
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    print(obs.keys())

    action = wrapper.action_space.sample()
    obs, reward, terminated, truncated, info = wrapper.step(action)
    img = wrapper.render()
    image=img[0].cpu().numpy()
    imageio.imwrite("test.png", image)
    wrapper.close()

