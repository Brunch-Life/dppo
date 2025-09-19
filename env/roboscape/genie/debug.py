import subprocess
import os

cmd = [
    "/ML-vePFS/zhangxin/envs/mininconda/envs/dppo_roboscape/bin/python",
    "env/roboscape/genie/train.py",
    "--genie_config",
    "/ML-vePFS/tangyinzhou/RoboScape-R/dppo/env/roboscape/genie/genie/configs/magvit_n32_h8_d512_action_done.json",
    "--output_dir",
    "/manifold-obs/tangyinzhou/encoded_maniskill/action_target_bins",
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
    "/manifold-obs/tangyinzhou/encoded_maniskill_delta_multi_view_full/merged",
    "--val_data_dir",
    "/manifold-obs/tangyinzhou/encoded_maniskill_delta_multi_view_full/merged",
    "--use_target_action",
    "--use_multi_view",
    "--use_bin_action",
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
        cwd="/ML-vePFS/tangyinzhou/RoboScape-R/dppo",
        env=env,  # 使用自定义的环境变量
    )
    print("External script output:", result.stdout)
    if result.returncode != 0:
        print("External script error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error running external script:", e.stderr)
