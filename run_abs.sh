
export http_proxy="http://100.68.170.237:3128"
export https_proxy="http://100.68.170.237:3128"


CUDA_VISIBLE_DEVICES=1 python script/run.py --config-name=ft_ppo_diffusion_mlp_img_abs \
    --config-dir=cfg/maniskill/finetune/ \
    --
