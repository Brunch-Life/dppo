import subprocess
import os

cmd = [
    "/iag_ad_01/ad/tangyinzhou/anaconda3/envs/worldmodel_tyz/bin/python",
    "train.py",
    "--genie_config",
    "genie/configs/magvit_n32_h8_d512_action_done.json",
    "--output_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/action_finetune_resumed",
    "--max_eval_steps",
    "10",
    "--learning_rate",
    "1e-5",
    "--num_train_epochs",
    "20",
    "--window_size",
    "16",
    "--stride",
    "4",
    "--per_device_train_batch_size",
    "1",
    "--per_device_eval_batch_size",
    "1",
    "--train_data_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta_multi_view_small/merged",
    "--val_data_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta_multi_view_small/merged",
    "--not_use_bin_action",
]

try:
    # 设置环境变量，让程序使用4号GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "1"

    # 启动并等待外部Python程序结束
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie",
        env=env,  # 使用自定义的环境变量
    )
    print("External script output:", result.stdout)
    if result.returncode != 0:
        print("External script error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error running external script:", e.stderr)
