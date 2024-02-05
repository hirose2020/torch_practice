#
# https://tatsukioike.com/pytorchnn/0064/
# PYTORCHでニューラルネットワーク#4】単回帰分析

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
torch.manual_seed(1)

x_train = torch.normal(5, 1, size=(10,))
t_train = 3*x_train + 2 + torch.randn(10)
x_train = x_train.unsqueeze(1).float()
t_train = t_train.unsqueeze(1).float()

plt.scatter(x_train,t_train)

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

#list(model.parameters())

for epoch in range(1, 5 + 1):
        y = model(x_train)
        loss_train = criterion(y, t_train)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        print(f"Epoch {epoch}, loss_train {loss_train:.4f}")

x = torch.arange(3,7).unsqueeze(1).float()
y = model(x)
plt.plot(x, y.detach())
plt.scatter(x_train, t_train)


