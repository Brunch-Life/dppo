import subprocess

cmd = [
    "/ML-vePFS/zhangxin/envs/mininconda/envs/dppo_roboscape/bin/python",
    "env/roboscape/genie/genie/generate.py",
    "--checkpoint_dir",
    "/manifold-obs/tangyinzhou/encoded_maniskill/action_target_bins_resumed/step_140000",
    "--output_dir",
    "/manifold-obs/tangyinzhou/encoded_maniskill/text_action_finetune_vis",
    "--stride",
    "4",
    "--val_data_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta_multi_view_small/merged",
    # "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta/merged",
    "--genie_config",
    "/ML-vePFS/tangyinzhou/RoboScape-R/dppo/env/roboscape/genie/genie/configs/magvit_n32_h8_d512_action_done.json",
    "--action_noise",
    "0.01",
    "--noise_dim",
    "-1",
    # "--force_end",
]

try:
    # 启动并等待外部Python程序结束
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/ML-vePFS/tangyinzhou/RoboScape-R/dppo/",
    )
    print("External script output:", result.stdout)
    if result.returncode != 0:
        print("External script error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error running external script:", e.stderr)
