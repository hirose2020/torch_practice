#
# PyTorchの使い方｜線形回帰から基本
# https://dreamer-uma.com/beginner-how-to-use-pytorch/

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# data
data = np.random.randn(200, 1)
label = 3*data + np.random.randn(200, 1)*0.5

x_train = data[:150, :]
y_train = label[:150, :]
x_test = data[:50, :]
y_test = label[:50, :]

x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()
"""
fig, ax = plt.subplots()
ax.scatter(x_train, y_train, alpha=0.8, label="train data")
ax.scatter(x_test, y_test, alpha=0.8, label="test data")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.legend()
plt.show()
"""
# PyTorchの自動微分を用いた実装
def model(x):
    return w * x + b

def criterion(output, y):
    loss = ((output - y)**2).mean()
    return loss

# パラメータ設定
w = torch.tensor(0.0, requires_grad=True).float()
b = torch.tensor(0.0, requires_grad=True).float()
lr = 0.01
num_epoch = 1000

# 学習
train_loss_list = []
for epoch in range(num_epoch):
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    with torch.no_grad():
        w -= lr*w.grad
        b -= lr*b.grad
        w.grad.zero_()
        b.grad.zero_()
    if (epoch%5 == 0):
        train_loss_list.append(loss)
        print(f"Epoch: {epoch} loss: {loss:.5f}")

# loss表示
# torchからnumpyへ変換
t = [buf.detach().numpy() for buf in train_loss_list]
fig, ax = plt.subplots(dpi=200)
epoch_list = np.arange(0, 1000, 5)
ax.plot(epoch_list, t)
ax.set_title(f'Result : w = {w:.4f}, b = {b:.4f}', fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()

### optimizerによる書き方
# 学習モデルを定義
def model(x):
    return w*x + b
# パラメータの初期値
w = torch.tensor(0.0, requires_grad=True).float()
b = torch.tensor(0.0, requires_grad=True).float()
# 損失関数を定義
criterion = nn.MSELoss()
# 最適化手法を指定
optimizer = optim.SGD([w, b], lr=0.01)
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 勾配初期化
    optimizer.zero_grad()
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算
    loss.backward()
    # パラメータ更新
    optimizer.step()
    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss.detach().item())
        print(f'【EPOCH {epoch}】 loss : {loss:.5f}')

### nn.Moduleを使用したモデル設計
# 学習モデルを定義
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1, 1, bias=True)
        # 初期値の設定
        nn.init.constant_(self.fc.weight, 0.0)
        nn.init.constant_(self.fc.bias, 0.0)
    def forward(self, x):
        x = self.fc(x)
        return x

# インスタンス生成
model = Net()
# 損失関数を定義
criterion = nn.MSELoss()
# 最適化手法を決定
optimizer = optim.SGD(model.parameters(), lr=0.01)
# エポック数
num_epoch = 1000

train_loss_list = []
for epoch in range(num_epoch):
    # 訓練モードに変更
    model.train()
    # 勾配初期化
    optimizer.zero_grad()
    # 予測
    output = model(x_train)
    # 損失関数を計算
    loss = criterion(output, y_train)
    # 勾配を計算
    loss.backward()
    # パラメータ更新
    optimizer.step()
    # lossを記録
    if (epoch%5==0):
        train_loss_list.append(loss.detach().item())
        print(f'【EPOCH {epoch}】 loss : {loss.detach().item():.5f}')

print(f"Result : w = {model.fc.weight.item():.4f}, b = {model.fc.bias.item():.4f}")
