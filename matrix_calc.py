#!/usr/bin/env python
#
# CPUとGPUのtorchテンソル計算速度を比較

import torch
import time
import numpy as np
import GPUtil

# テンソルの二乗の計算をn回繰り返し、かかった時間の平均時間を返す
def timeit(p: int, n: int, cuda: str) -> float:
    t_ave = 0
    for i in range(n):
        t0 = time.time()
        x = torch.rand(p, p)
        if cuda == "cuda":
            x = x.to("cuda")
            x = torch.mm(x, x)
        elif cuda == "cpu":
            x = torch.mm(x, x)
        elif cuda == "numpy":
            nx = x.numpy()
            nx = np.dot(nx, nx)

        dt = time.time() - t0
        print(f"device = {cuda}, p = {p}, n = {n}, dt = {dt}")
        t_ave = t_ave + dt
    del x # tensorが占めているGPUのメモリを開放
    torch.cuda.empty_cache() # キャッシュも消す
    return t_ave / n

import pandas as pd
df = pd.DataFrame(columns=["size", "cuda", "cpu", "numpy", "ratio"])
size_set = [1000, 3000, 10000]

t_cuda, t_cpu, t_numpy = [], [], []
t_ratio = []
n = 10
for size_p in size_set:
    t = timeit(size_p, n, "cuda")
    t_cuda.append(t)
    t = timeit(size_p, n, "cpu")
    t_cpu.append(t)
    t = timeit(size_p, n, "numpy")
    t_numpy.append(t)
    t_ratio.append(t_cpu[-1]/t_cuda[-1])

df["size"] = size_set
df["cuda"] = t_cuda
df["cpu"] = t_cpu
df["numpy"] = t_numpy
df["ratio"] = t_ratio
print(df)
