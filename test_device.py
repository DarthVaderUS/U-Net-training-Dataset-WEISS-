import torch
import torchvision
import torchaudio

def print_separator(title):
    print("\n" + "="*10 + f" {title} " + "="*10)

# 基本信息
print_separator("Environment Info")
print(f"PyTorch Version       : {torch.__version__}")
print(f"TorchVision Version   : {torchvision.__version__}")
print(f"TorchAudio Version    : {torchaudio.__version__}")
print(f"CUDA Available        : {torch.cuda.is_available()}")
print(f"cuDNN Enabled         : {torch.backends.cudnn.enabled}")

# CUDA 设备信息
if torch.cuda.is_available():
    print_separator("CUDA Devices Info")
    print(f"GPU Count             : {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i} Name        : {torch.cuda.get_device_name(i)}")
        print(f"Device {i} Capability  : {torch.cuda.get_device_capability(i)}")
        print(f"Device {i} Memory Allocated : {torch.cuda.memory_allocated(i)/1024**2:.2f} MB")
        print(f"Device {i} Memory Cached    : {torch.cuda.memory_reserved(i)/1024**2:.2f} MB")

# 简单张量计算测试
print_separator("Tensor Computation Test")

try:
    x_cpu = torch.rand(1000, 1000)
    y_cpu = torch.mm(x_cpu, x_cpu)
    print("CPU tensor multiplication successful.")
except Exception as e:
    print(" CPU tensor multiplication failed:", e)

if torch.cuda.is_available():
    try:
        device = torch.device("cuda")
        x_gpu = torch.rand(1000, 1000).to(device)
        y_gpu = torch.mm(x_gpu, x_gpu)
        print(" GPU tensor multiplication successful.")
    except Exception as e:
        print(" GPU tensor multiplication failed:", e)

# 简单模型迁移测试
print_separator("Model GPU Test")
model = torch.nn.Conv2d(3, 16, 3, padding=1)

try:
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = torch.rand(1, 3, 224, 224).to("cuda")
    else:
        dummy_input = torch.rand(1, 3, 224, 224)
    
    output = model(dummy_input)
    print(f" Model ran on {'GPU' if torch.cuda.is_available() else 'CPU'} successfully.")
except Exception as e:
    print(" Model execution failed:", e)
