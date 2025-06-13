import os
import ollama
import torch

def run_test():
    print("CUDA: " + str(torch.cuda.is_available()));
    print("TORCH VERSION: " + str(torch.__version__))  # PyTorch version
    print("TORCH CUDA VERSION: " + str(torch.version.cuda))  # CUDA version PyTorch was compiled with
    print("CUDA_VISIBLE_DEVICES: " + os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"));
    print("CUDA DEVICE: " + str(torch.cuda.current_device()))  # Should show the current device (GPU)
    print("CUDA DEVICE NAME: " + str(torch.cuda.get_device_name(0)))  # Should return the name of your GPU