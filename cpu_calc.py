import numpy as np
import time

N=1000
t0 = time.time()

x = np.random.rand(N, N)
x = np.dot(x, x)

dt = time.time() - t0

print(dt)

