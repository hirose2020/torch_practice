import torch

print(f"version of torch: \033[31m{torch.__version__}\033[0m")
print(f"cuda available: \033[31m{torch.cuda.is_available()}\033[0m")
print(f"cuda device count: \033[31m{torch.cuda.device_count()}\033[0m")
print(f"cuda current device: \033[31m{torch.cuda.current_device()}\033[0m")
print(f"device name: \033[31m{torch.cuda.get_device_name()}\033[0m")
print(f"device capability: \033[31m{torch.cuda.get_device_capability()}\033[0m")
