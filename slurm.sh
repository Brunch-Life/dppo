export http_proxy="http://100.68.170.237:3128"
export https_proxy="http://100.68.170.237:3128"


# system config
# cp "/etc/apt/sources.list" "/etc/apt/sources.list.bak"
# cp "/iag_ad_01/ad/tangyinzhou/bingwen/server_doc/sources.list" "/etc/apt/sources.list"
apt-get update
apt-get install libvulkan1 vulkan-tools libglvnd-dev -y

# project config
# ln -s /iag_ad_01/ad/tangyinzhou/bingwen/Documents ~/Documents
cd /ML-vePFS/tangyinzhou/yinuo/dppo

SOURCE_FILE_PATH="/ML-vePFS/tangyinzhou/bingwen/ManiSkill/docker/nvidia_icd.json"
DEST_FILE_PATH="/usr/share/vulkan/icd.d/nvidia_icd.json"

mkdir -p "/usr/share/vulkan/icd.d/"
if [ ! -f "$DEST_FILE_PATH" ]; then
    echo "$DEST_FILE_PATH not exist, copying from $SOURCE_FILE_PATH"
    cp "$SOURCE_FILE_PATH" "$DEST_FILE_PATH"
else
    echo "nvidia_icd.json $DEST_FILE_PATH exist, skip copy."
fi


cd /ML-vePFS/tangyinzhou/yinuo/dppo

/ML-vePFS/zhangxin/envs/mininconda/envs/dppo/bin/python script/run.py \
--config-name=ft_ppo_diffusion_mlp_img \
--config-dir=cfg/maniskill/finetune/