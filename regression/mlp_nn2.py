#
# 現場で使えるPytorch開発入門
# p.52 Chap 3.3
# 多層パーセプトロン（MPL)
# 手書き文字の認識
# Dropoutによる正則化

# FeedForward型MLP
import torch
from torch import nn

k = 100
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Linear(k, 10)
)

# データの用意、分割
from torch import optim
from sklearn.datasets import load_digits
digits = load_digits()
x = digits.data
y = digits.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.int64)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

# ミニバッチ処理の用意
from torch.utils.data import TensorDataset, DataLoader
ds = TensorDataset(x_train, y_train)
loader = DataLoader(ds, batch_size=64, shuffle=True)

# 学習
train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    y_pred = net(x_test)
    test_loss = loss_fn(y_pred, y_test)
    test_losses.append(test_loss.item())

import matplotlib.pyplot as plt
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()

# 過学習が起こり、testのlossが上昇してしまう。
# Dropoutで過学習を抑える

# 確率0.5でランダムに変数の次元を捨てる
k = 100
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(k, 10)
)

# trainとevalでDropoutの挙動を切り替える
optimizer = optim.Adam(net.parameters())

train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    net.train() # 学習モードへ
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    net.eval() # 評価モードへ
    y_pred = net(x_test)
    test_loss = loss_fn(y_pred, y_test)
    test_losses.append(test_loss.item())

import matplotlib.pyplot as plt
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()

# Batch Normalizationによる学習の加速
k = 100
net = nn.Sequential(
    nn.Linear(64, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, k),
    nn.ReLU(),
    nn.BatchNorm1d(k),
    nn.Linear(k, 10)
)

# trainとevalでDropoutの挙動を切り替える
optimizer = optim.Adam(net.parameters())

train_losses = []
test_losses = []
for epoch in range(100):
    running_loss = 0.0
    net.train() # 学習モードへ
    for i, (xx, yy) in enumerate(loader):
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / i)
    net.eval() # 評価モードへ
    y_pred = net(x_test)
    test_loss = loss_fn(y_pred, y_test)
    test_losses.append(test_loss.item())

import matplotlib.pyplot as plt
plt.plot(train_losses, label="train")
plt.plot(test_losses, label="test")
plt.legend()
plt.show()



