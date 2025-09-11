from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import math
import einops
from torchvision import models as vision_models


@dataclass
class ResNetEncoderConfig:
    """ResNet编码器配置参数"""

    pretrained: bool = False
    input_coord_conv: bool = False
    # 可以根据需要添加更多ResNet相关配置参数


class ResNetEncoder(nn.Module):
    """ResNet图像编码器，保持与ViTEncoder兼容的接口"""

    def __init__(
        self,
        obs_shape: List[int],
        cfg: ResNetEncoderConfig,
        num_channel=3,
        img_h=96,
        img_w=96,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.cfg = cfg
        self.num_channel = num_channel
        self.img_h = img_h
        self.img_w = img_w

        # 初始化ResNet18
        self.resnet = self._build_resnet()

        # 计算特征相关维度信息
        self.num_features = 512  # ResNet18最后一层的输出通道数
        output_shape = self.output_shape(obs_shape)
        self.num_patches = output_shape[1] * output_shape[2]  # 空间维度的乘积
        self.patch_repr_dim = self.num_features
        self.repr_dim = self.num_features * self.num_patches  # 展平后的总维度

    def _build_resnet(self):
        """构建ResNet18网络结构"""
        # 加载ResNet18，可选择预训练权重
        net = vision_models.resnet18(pretrained=self.cfg.pretrained)

        # 处理输入通道数和坐标卷积
        if self.cfg.input_coord_conv:
            # 使用坐标卷积替换第一个卷积层
            net.conv1 = CoordConv2d(
                self.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        elif self.num_channel != 3:
            # 调整第一个卷积层以适应不同的输入通道数
            net.conv1 = nn.Conv2d(
                self.num_channel, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        # 移除最后的全连接层和平均池化层，保留卷积部分
        return nn.Sequential(*(list(net.children())[:-2]))

    def forward(self, obs, flatten=False) -> torch.Tensor:
        """
        前向传播函数

        Args:
            obs: 输入图像，形状为 [batch, channels, height, width]
            flatten: 是否将输出展平为 [batch, num_features * num_patches]

        Returns:
            编码后的特征
        """
        # 图像归一化 (与ViT保持一致的预处理)
        obs = obs / 255.0 - 0.5

        # 通过ResNet提取特征
        feats = self.resnet(obs)  # 形状: [batch, 512, h, w]

        # 转换为 [batch, num_patches, patch_repr_dim] 格式，与ViT输出格式对齐
        feats = einops.rearrange(feats, "b c h w -> b (h w) c")

        # 如果需要展平为一维特征
        if flatten:
            feats = feats.flatten(1, 2)

        return feats

    def output_shape(self, input_shape):
        """
        计算输出特征的形状

        Args:
            input_shape: 输入形状，格式为 [channels, height, width]

        Returns:
            输出特征形状，格式为 [num_features, out_h, out_w]
        """
        assert len(input_shape) == 3, "输入形状应为 [channels, height, width]"
        # ResNet18的总步长为32
        out_h = int(math.ceil(input_shape[1] / 32.0))
        out_w = int(math.ceil(input_shape[2] / 32.0))
        return [self.num_features, out_h, out_w]


class CoordConv2d(nn.Conv2d):
    """坐标卷积层，在输入特征中添加坐标信息"""

    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels + 2, out_channels, *args, **kwargs)
        self.in_channels = in_channels

    def forward(self, x):
        """
        添加x和y坐标通道到输入中

        Args:
            x: 输入张量，形状为 [batch, channels, height, width]

        Returns:
            添加坐标信息后的特征图
        """
        batch_size, _, height, width = x.size()

        # 创建坐标通道
        x_coord = torch.linspace(-1, 1, width, device=x.device)
        y_coord = torch.linspace(-1, 1, height, device=x.device)
        x_coord = x_coord.repeat(batch_size, height, 1).unsqueeze(1)  # [B, 1, H, W]
        y_coord = (
            y_coord.repeat(batch_size, width, 1).transpose(1, 2).unsqueeze(1)
        )  # [B, 1, H, W]

        # 拼接原始特征和坐标特征
        x_with_coord = torch.cat([x, x_coord, y_coord], dim=1)

        return super().forward(x_with_coord)


# 测试代码
if __name__ == "__main__":
    # 测试ResNet编码器
    obs_shape = [3, 224, 224]  # [通道数, 高度, 宽度]
    cfg = ResNetEncoderConfig(pretrained=False)

    enc = ResNetEncoder(
        obs_shape,
        cfg,
        num_channel=obs_shape[0],
        img_h=obs_shape[1],
        img_w=obs_shape[2],
    )

    print(enc)
    checkpoint = torch.load(
        "/ML-vePFS/tangyinzhou/yinuo/dp_train_zhiting/ckpts/20250904_131139/policy_step_200000_seed_0.ckpt",
        map_location=torch.device("cpu"),
    )
    ckpt_state_dict = checkpoint["nets"]  # 从 'nets' 中提取完整状态字典
    prefix_to_remove = "policy.backbones.0.nets."
    model_state_dict = enc.state_dict()
    new_state_dict = {}

    # 收集所有需要加载的参数键
    for ckpt_key in ckpt_state_dict.keys():
        if ckpt_key.startswith(prefix_to_remove):
            # 裁剪前缀并构建模型键
            model_key = ckpt_key[len(prefix_to_remove) :]
            model_key = f"resnet.{model_key}"

            # 检查模型是否有这个键，确保形状匹配
            if model_key in model_state_dict:
                if ckpt_state_dict[ckpt_key].shape == model_state_dict[model_key].shape:
                    new_state_dict[model_key] = ckpt_state_dict[ckpt_key]
                else:
                    print(f"形状不匹配，跳过: {model_key}")
            else:
                print(f"模型中不存在该键，跳过: {model_key}")

    # 4. 加载参数
    enc.load_state_dict(new_state_dict, strict=False)
    print(f"成功加载 {len(new_state_dict)}/{len(model_state_dict)} 个参数")
    x = torch.rand(1, *obs_shape) * 255  # 生成随机图像输入
    print("输出形状 (未展平):", enc(x, flatten=False).size())
    print("输出形状 (展平):", enc(x, flatten=True).size())
    print("特征维度:", enc.repr_dim)
