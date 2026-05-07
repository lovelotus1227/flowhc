import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
print(f"PyTorch内置CUDA版本: {torch.version.cuda}")
import mmcv
print(f"MMCV版本: {mmcv.__version__}")
# 验证CUDA是否适配
print(f"MMCV CUDA是否可用: {mmcv.is_cuda_available()}")