import subprocess

cmd = [
    "/iag_ad_01/ad/tangyinzhou/anaconda3/envs/worldmodel_tyz/bin/python",
    "visualize_encoded.py",
    "--token_dir",
    "/tangyinzhou-tos-volc-engine/tyz/encoded_maniskill_delta_multi_view/merged/val",
    "--genie_config",
    "/iag_ad_01/ad/tangyinzhou/tyz/observation-genie/genie/genie/configs/magvit_n32_h8_d512_action_done.json",
    "--disable_comic",
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
