import torch
import numpy as np

def log_lh(p):
    return (2 * torch.log(p) + 3 * torch.log(1-p))

num_epoch = 40
lr = 0.01

p = torch.tensor(0.1, dtype=torch.float32, requires_grad=True)

logs = np.zeros((0, 3))
for epoch in range(num_epoch):
    loss = -log_lh(p)
    loss.backward()
    with torch.no_grad():
        p -= lr * p.grad
        p.grad.zero_()
    log = np.array([epoch, p.item(), loss.item()]).reshape(1, -1)
    logs = np.vstack([logs, log])

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"]
fig, ax = plt.subplots(1, 2)
ax[0].plot(logs[:, 0], logs[:, 1])
ax[0].set_title("p")
ax[1].plot(logs[:, 0], logs[:, 2])
ax[1].set_title("loss")
plt.tight_layout()
plt.show()


