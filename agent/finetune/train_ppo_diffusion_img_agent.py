"""
DPPO fine-tuning for pixel observations.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
import math
import imageio

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.common.modules import RandomShiftsAug


class TrainPPOImgDiffusionAgent(TrainPPODiffusionAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Image randomization
        self.augment = cfg.train.augment
        if self.augment:
            self.aug = RandomShiftsAug(pad=4)

        # Set obs dim -  we will save the different obs in batch in a dict
        shape_meta = cfg.shape_meta
        self.obs_dims = {k: shape_meta.obs[k]["shape"] for k in shape_meta.obs}

        # Gradient accumulation to deal with large GPU RAM usage
        self.grad_accumulate = cfg.train.grad_accumulate

        self.debug_step = 0

    def run(self):

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = {}
            if self.itr % self.render_freq == 0 and self.render_video:
                # for env_ind in range(self.n_render):
                options_venv["video_path"] = os.path.join(
                    self.render_dir, f"itr-{self.itr}_trial-0.mp4"
                )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # # eval
            # self.model.eval()
            # data_dict = np.load("./debug/data.npy", allow_pickle=True).item()

            # print("image_data:", data_dict["image_data"].reshape(1, 1, 6, 224, 224))

            # # print("shape check:", data_dict["image_data"].shape, data_dict["qpos_data"].shape, data_dict["pred_actions"].shape)
            # image_data = data_dict["image_data"].reshape(1, 1, 6, 224, 224)
            # qpos_data = data_dict["qpos_data"].reshape(1, 1, 10)
            # # print("shape check:", image_data.shape, qpos_data.shape, data_dict["pred_actions"].shape)
            # cond = {
            #     "rgb": torch.from_numpy(image_data).float().to(self.device),
            #     "state": torch.from_numpy(qpos_data).float().to(self.device),
            # }

            # samples = self.model(
            #     cond=cond,
            #     deterministic=eval_mode,
            #     return_chain=True,
            # )

            # print("predict_actions:", data_dict["pred_actions"])
            # print("model actions:", samples.trajectories.cpu().numpy())

            # # data_dict_2 = np.load("./debug/data2.npy", allow_pickle=True).item()
            # # print("check:", data_dict_2["naction"].shape, data_dict_2["obs_cond"].shape, data_dict_2["k"], data_dict_2["noise_pred"].shape)

            # # print("cond_true:", data_dict_2["obs_cond"])

            # # noise_pred = self.model.actor(torch.tensor(data_dict_2["naction"]).float().to(self.device),
            # #                               torch.tensor([0]).float().to(self.device),
            # #                               cond=cond)
            # # print("noise_pred:", noise_pred.cpu().numpy())
            # # print("true noise:", data_dict_2["noise_pred"])
            # exit(0)

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                k: np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, *self.obs_dims[k])
                )
                for k in self.obs_dims
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():

                    # device = prev_obs_venv["rgb"].device

                    # data_dict = np.load("./debug/data3.npy", allow_pickle=True).item()
                    # image_list = data_dict["image_data"]
                    # # print("shape:", image_list.shape)
                    # raw_obs = {
                    #     "sensor_data": {
                    #         "3rd_view_camera": {
                    #             "rgb": torch.tensor(image_list[0]).to(device),
                    #         },
                    #         "hand_camera": {
                    #             "rgb": torch.tensor(image_list[1]).to(device),
                    #         },
                    #     }
                    # }
                    # # raw_obs['sensor_data']["3rd_view_camera"]['rgb'] = image_list[0]
                    # # raw_obs['sensor_data']["hand_camera"]['rgb'] = image_list[1]
                    # prev_obs_venv = self.venv.env.get_observation(raw_obs)

                    # save_dir = "./debug/img/"
                    # os.makedirs(save_dir, exist_ok=True)
                    # for i in range(prev_obs_venv["rgb"].shape[2]):
                    #     image = prev_obs_venv["rgb"][0, 0, i].cpu().numpy()
                    #     print("image:", image.shape)
                    #     image = image.transpose(1, 2, 0)
                    #     image = (image * 255).astype(np.uint8)
                    #     import cv2
                    #     cv2.imwrite(f"{save_dir}/img_{step}_{i}.png", image)
                    
                    prev_obs_venv["rgb"] = prev_obs_venv["rgb"].reshape(-1, 1, 6, 224, 224)


                    # debug_obs_dir = "/ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting/debug/obs"


                    # self.debug_step = min(self.debug_step, 14)

                    # print("debug_step:", self.debug_step)
                    
                    # file_name = f"test_image_{self.debug_step}.pt"

                    # image_debug = torch.load(os.path.join(debug_obs_dir, file_name)).to(prev_obs_venv["rgb"].device)

                    # image_save = image_debug.cpu().numpy()
                    # image_save = image_save.transpose(0, 2, 3, 1)
                    # image_save = image_save*255
                    # image_save = image_save.astype(np.uint8)
                    # for i in range(len(image_save)):
                    #     imageio.imwrite(f"debug/obs/test_image_{self.debug_step}_{i}.png", image_save[i])

                    # # image_debug = image_debug.reshape(-1, 1, 6, 224, 224)
                    # image_debug = einops.rearrange(image_debug, "n c h w -> 1 1 (n c) h w")
                    # prev_obs_venv["rgb"] = image_debug
                    prev_obs_venv["state"] = (
                        torch.zeros(
                            (
                                prev_obs_venv["rgb"].shape[0],
                                prev_obs_venv["rgb"].shape[1],
                                10,
                            )
                        )
                    ).float().to(prev_obs_venv["rgb"].device)


                    self.debug_step += 1
                     
                    cond = {
                        key: prev_obs_venv[key]
                        .float()
                        .to(self.device)
                        for key in self.obs_dims
                    }
                    # batch each type of obs and put into dict
                    # cond["rgb"] = cond["rgb"].float() / 255.0

                    # data_dict = np.load("./debug/data3.npy", allow_pickle=True).item()

                    # print("check obs:", data_dict["obs_image"].shape, cond["rgb"].shape)
                    # exit(0)

                    # print("=============== %d ===============" % self.debug_step)
                    
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = (
                        samples.trajectories.cpu().numpy()
                    )  # n_env x horizon x act
                    
                    # print("output_venv:", output_venv[0], output_venv.shape)

                    # exit(0)

                    # action_check = np.load("./debug/data3.npy", allow_pickle=True).item()
                    # print("????????:", action_check["raw_actions"])
                    # # print("output_venv:", output_venv)
                    # print("???:", self.venv.env.unnormalize_action(output_venv))
                    # print("delta:", self.venv.env.unnormalize_action(output_venv) - action_check["raw_actions"])
                    # exit(0)

                    # print("output_venv:", output_venv)
                    # print("???:", self.venv.env.action_transform(self.venv.env.unnormalize_action(output_venv[0, 0:4])))
                    # exit(0)

                    chains_venv = (
                        samples.chains.cpu().numpy()
                    )  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # env here
                # Apply multi-step action
                # MultiStep wrapper expects (n_action_steps, n_env, action_dim)
                # We have (n_env, n_action_steps, action_dim), so transpose to get correct shape
                # action_venv = action_venv.transpose(1, 0, 2)  # (n_action_steps, n_env, action_dim)
                # print("action_venv:", torch.from_numpy(action_venv[0, :3,:]))
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )

                done_venv = terminated_venv | truncated_venv
                for k in obs_trajs:
                    # Transpose from (n_cond_step, n_envs, ...) to (n_envs, n_cond_step, ...)
                    # For 5D tensor: (1, 32, 6, 224, 224) -> (32, 1, 6, 224, 224)
                    # axes = [1, 0] + list(range(2, prev_obs_venv[k].ndim))
                    # obs_trajs[k][step] = prev_obs_venv[k].permute(axes).cpu().numpy()
                    obs_trajs[k][step] = prev_obs_venv[k].cpu().numpy()
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv.cpu().numpy()
                terminated_trajs[step] = terminated_venv.cpu().numpy()
                firsts_trajs[step + 1] = done_venv.cpu().numpy()

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    # apply image randomization
                    obs_trajs["rgb"] = (
                        torch.from_numpy(obs_trajs["rgb"]).float().to(self.device)
                    )
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )
                    if self.augment:
                        rgb = einops.rearrange(
                            obs_trajs["rgb"],
                            "s e t c h w -> (s e t) c h w",
                        )
                        rgb = self.aug(rgb)
                        obs_trajs["rgb"] = einops.rearrange(
                            rgb,
                            "(s e t) c h w -> s e t c h w",
                            s=self.n_steps,
                            e=self.n_envs,
                        )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    for k in obs_trajs:
                        obs_k = einops.rearrange(
                            obs_trajs[k],
                            "s e ... -> (s e) ...",
                        )
                        obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                        for i, obs_t in enumerate(obs_ts_k):
                            obs_ts[i][k] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = (
                            self.model.critic(obs, no_augment=True)
                            .cpu()
                            .numpy()
                            .flatten()
                        )
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        key: torch.from_numpy(obs_venv[key]).float().to(self.device)
                        for key in self.obs_dims
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts, no_augment=True)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

                # k for environment step
                obs_k = {
                    k: einops.rearrange(
                        obs_trajs[k],
                        "s e ... -> (s e) ...",
                    )
                    for k in obs_trajs
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device)
                    .float()
                    .reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):

                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = torch.unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {k: obs_k[k][batch_inds_b] for k in obs_k}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        loss.backward()
                        if (batch + 1) % self.grad_accumulate == 0:
                            if self.itr >= self.n_critic_warmup_itr:
                                if self.max_grad_norm is not None:
                                    torch.nn.utils.clip_grad_norm_(
                                        self.model.actor_ft.parameters(),
                                        self.max_grad_norm,
                                    )
                                self.actor_optimizer.step()
                                if (
                                    self.learn_eta
                                    and batch % self.eta_update_interval == 0
                                ):
                                    self.eta_optimizer.step()
                            self.critic_optimizer.step()
                            self.actor_optimizer.zero_grad()
                            self.critic_optimizer.zero_grad()
                            if self.learn_eta:
                                self.eta_optimizer.zero_grad()
                            log.info(f"run grad update at batch {batch}")
                            log.info(
                                f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                            )

                            # Stop gradient update if KL difference reaches target
                            if (
                                self.target_kl is not None
                                and approx_kl > self.target_kl
                                and self.itr >= self.n_critic_warmup_itr
                            ):
                                flag_break = True
                                break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=False,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0][
                                    "lr"
                                ],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
