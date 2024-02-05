#!/usr/bin/env python

import torch
import time

N = 100
#M = 20000
M = 10000
t_sum = 0.0
for i in range(N):
    t0 = time.time()
    x = torch.rand(M, M).to("cuda")
    torch.mm(x, x)
    dt = time.time() - t0

    del x
    torch.cuda.empty_cache()
    print(f"{i:4d}:  {dt:6.4f}")
    t_sum += dt

print(t_sum / N)
