# PyTorchで線形回帰
# https://dreamer-uma.com/beginner-how-to-use-pytorch/

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# 標準正規分布に従う入力
data = np.random.randn(200, 1)
# y=3xにノイズを追加した出力
label = 3*data + np.random.randn(200, 1)*0.5

x_train = data[:150, :]
y_train = label[:150, :]
x_test = data[:50, :]
y_test = label[:50, :]

# tesnsor化 : float32に変更
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).float()
x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

# nn.Moduleを使用したモデル設計
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
num_epoch = 10000

# 学習
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


# 誤差関数とパラメータの推定値を表示        
fig, ax = plt.subplots()
epoch_list = np.arange(0, 10000, 5)
ax.plot(epoch_list, train_loss_list)
# w, bの確認の仕方が異なることに注意
ax.set_title(f'Result : w = {model.fc.weight.item():.4f}, b = {model.fc.bias.item():.4f}', fontsize=15)
ax.set_ylabel('train loss', fontsize=20)
plt.show()

print('Result : w = {:.4f}, {:.4f}'.format(model.fc.weight.item(), model.fc.bias.item()))

