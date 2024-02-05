#
# 現場で使えるPytorch開発入門
# p.46 Chap 3.1
# 多層パーセプトロン（MPL)
# 手書き文字の認識

# FeedForward型MLP
import torch
from torch import nn

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

# データの用意、学習
from torch import optim
from sklearn.datasets import load_digits
digits = load_digits()
x = digits.data
y = digits.target
x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

losses = []

"""
y = y.to("cuda:0")
x = x.to("cuda:0")
net.to("cuda:0")
"""

for epoc in range(1000):
    optimizer.zero_grad() # 前回のbarkwardで計算された勾配を削除
    y_pred = net(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

#print(list(net.parameters()))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

## ミニバッチ処理
print("--- mini batch ---\n\n")
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(x, y)
loader = DataLoader(ds, batch_size=64, shuffle=True)

net = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

losses = []
for epoc in range(30):
    running_loss = 0.0
    for xx, yy in loader:
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    losses.append(running_loss)

#print(list(net.parameters()))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()
