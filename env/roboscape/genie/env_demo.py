from wm_env.wm_env_new import WorldModelEnv
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generates samples (as tokens) from GENIE model. "
        "Optionally visualizes these tokens as GIFs or comics."
    )
    parser.add_argument(
        "--wmconfig",
        type=str,
        default="/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie/genie/configs/magvit_n32_h8_d512_action_done.json",
    )
    parser.add_argument(
        "--goaler_config",
        type=str,
        default="/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie/genie/configs/magvit_n32_h8_d512_text_done.json",
    )
    parser.add_argument(
        "--wmckpt",
        type=str,
        default="/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/action_only_finetune/step_450000",
    )
    parser.add_argument(
        "--goaler_ckpt",
        type=str,
        default="/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/text_only_finetune/step_160000",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/merged",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick",
    )
    parser.add_argument(
        "--scene_id",
        type=str,
        default="20250714_100315",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--total_steps",
        type=int,
        default=2,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = WorldModelEnv(
        args.task,
        args.wmconfig,
        args.goaler_config,
        args.wmckpt,
        args.goaler_ckpt,
        args.batch_size,
        args.total_steps,
        args.scene_id,
        args.data_dir,
    )
    init_obs = env.reset()
    for x in range(10):
        action = np.random.uniform(-1, 1, size=(args.batch_size, 8))
        new_obs, rewards, dones, _ = env.step(action)


if __name__ == "__main__":
    main()
