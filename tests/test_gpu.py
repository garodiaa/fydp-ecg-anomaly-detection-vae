import torch
print(f"PyTorch version: {torch.__version__}")
is_cuda_available = torch.cuda.is_available()
print(f"Is CUDA available? {is_cuda_available}")
if is_cuda_available:
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")