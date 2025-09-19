import subprocess

cmd = [
    "/ML-vePFS/zhangxin/envs/mininconda/envs/dppo_roboscape/bin/python",
    "script/run.py",
    f"--config-name=ft_ppo_diffusion_mlp_img",
    f"--config-dir=/ML-vePFS/tangyinzhou/RoboScape-R/dppo/cfg/roboscape/finetune",
]

try:
    # 启动并等待外部Python程序结束
    result = subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd="/ML-vePFS/tangyinzhou/RoboScape-R/dppo",
    )
    print("External script output:", result.stdout)
    if result.returncode != 0:
        print("External script error:", result.stderr)
except subprocess.CalledProcessError as e:
    print("Error running external script:", e.stderr)
