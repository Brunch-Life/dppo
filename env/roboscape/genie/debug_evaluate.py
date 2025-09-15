import subprocess

cmd = [
    "/iag_ad_01/ad/tangyinzhou/anaconda3/envs/worldmodel_tyz/bin/python",
    "genie/evaluate.py",
    "--checkpoint_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/action_only_finetune/step_460000",
    "--save_outputs_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/text_action_finetune_vis",
    "--stride",
    "4",
    "--val_data_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill/merged",
    "--genie_config",
    "/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie/genie/configs/magvit_n32_h8_d512_action_done.json",
]

try:
    # 启动并等待外部Python程序结束
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie",
    )
    print("External script output:", result.stdout)
    if result.returncode != 0:
        print("External script error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error running external script:", e.stderr)
