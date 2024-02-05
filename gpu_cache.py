import torch
import GPUtil

device = torch.device("cuda")

print("init")
p = torch.cuda.get_device_properties(device=device).total_memory
print(p)

r = torch.cuda.memory_reserved(device=device)
print(r)
l = torch.cuda.memory_allocated(device=device)
print(l)

a = torch.rand(1, 1000, 1000, 1000).to("cuda")

print("\ntensor instance")
r = torch.cuda.memory_reserved(device=device)
print(r)
l = torch.cuda.memory_allocated(device=device)
print(l)

del a

print("\ndel tensor")
r = torch.cuda.memory_reserved(device=device)
print(r)
l = torch.cuda.memory_allocated(device=device)
print(l)

torch.cuda.empty_cache()

print("\nempty cache")
r = torch.cuda.memory_reserved(device=device)
print(r)
l = torch.cuda.memory_allocated(device=device)
print(l)
