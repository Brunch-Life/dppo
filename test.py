# test_compatibility.py
import torch
from accelerate import Accelerator
import xformers
import flash_attn

# 验证PyTorch版本
print("Torch版本:", torch.__version__)  # 应输出2.4.0

# 验证accelerate与torch兼容性
accelerator = Accelerator()
print("Accelerator初始化成功")

# 1. 先确认PyTorch CUDA可用（xformers依赖此环境）
if not torch.cuda.is_available():
    raise RuntimeError("PyTorch CUDA不可用，xformers无法使用CUDA")

# 2. 确认xformers安装成功（打印版本）
print(f"Xformers可用（版本: {xformers.__version__}，CUDA支持依赖PyTorch正常）")


# 验证flash-attn
q = torch.randn(2, 4, 8, 64, device="cuda")
k = torch.randn(2, 4, 16, 64, device="cuda")
v = torch.randn(2, 4, 16, 64, device="cuda")
out = flash_attn.flash_attn_qkvpacked_func(q, k, v)
print("Flash-Attn测试通过，输出形状:", out.shape)