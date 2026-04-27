import torch
import intel_extension_for_pytorch as ipex

print(f"PyTorch Version: {torch.__version__}")
print(f"IPEX Version: {ipex.__version__}")
print(f"Intel GPU Available: {torch.xpu.is_available()}")