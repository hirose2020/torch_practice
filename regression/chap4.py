# 4章 予測関数定義
# from https://github.com/makaishi2/pytorch_book_info/blob/main/notebooks/ch04_model_dev.ipynb

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

import torch
import torch.nn as nn
from torchviz import make_dot

# レイヤー関数定義

# 最初の線形関数 784 入力数 128 出力数
l1 = nn.Linear(784, 128)
# 2番目の線形関数 #128 入力数  10 出力数
l2 = nn.Linear(128, 10)
# 活性化関数
relu = nn.ReLU(inplace=True)

# 入力テンソルから出力テンソルを計算
# ダミー入力データを作成
inputs = torch.randn(100, 784)
# 中間テンソル1の計算
m1 = l1(inputs)
# 中間テンソル2の計算
m2 = relu(m1)
# 出力テンソルの計算
outputs = l2(m2)
# 入力テンソルと出力テンソルのshape確認
print('入力テンソル', inputs.shape)
print('出力テンソル', outputs.shape)

# nn.Sequentialを使って、全体を合成関数として定義
net2 = nn.Sequential(
    l1,
    relu,
    l2
)
outputs2 = net2(inputs)

# 入力テンソルと出力テンソルのshape確認
print('入力テンソル', inputs.shape)
print('出力テンソル', outputs2.shape)

# 訓練データ、検証データの計算
np.random.seed(123)
x = np.random.randn(100,1)

# yの値はx^2に乱数の要素を1/10程度付加した
y = x**2 + np.random.randn(100,1) * 0.1

# データを50件ずつに分け、それぞれ訓練用、検証用とする
x_train = x[:50,:]
x_test = x[50:,:]
y_train = y[:50,:]
y_test = y[50:,:]

# 散布図表示
plt.scatter(x_train, y_train, c='b', label='訓練データ')
plt.scatter(x_test, y_test, c='k', marker='x', label='検証データ')
plt.legend()
plt.show()

# 入力変数x と正解値 ytのTesor化
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

import torch.optim as optim

# モデルの定義
class Net(nn.Module):
    def __init__(self):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()
        # 出力層の定義
        self.l1 = nn.Linear(1, 1)   
    # 予測関数の定義
    def forward(self, x):
        x1 = self.l1(x) # 線形回帰
        return x1

# 学習率
lr = 0.01
# インスタンス生成　(パラメータ値初期化)
net = Net()
# 最適化アルゴリズム: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)
# 損失関数： 最小二乗誤差
criterion = nn.MSELoss()
# 繰り返し回数
num_epochs = 10000
# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

# 繰り返し計算メインループ
for epoch in range(num_epochs):
    # 勾配値初期化
    optimizer.zero_grad()
    # 予測計算
    outputs = net(inputs)
    # 誤差計算
    loss = criterion(outputs, labels)
    #勾配計算
    loss.backward()
    # 勾配降下法の適用
    optimizer.step()
    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# 結果のグラフ化
labels_pred = net(inputs_test)
plt.title('隠れ層なし　活性化関数なし')
plt.scatter(inputs_test[:,0].data, labels_pred[:,0].data, c='b', label='予測値')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='正解値')
plt.legend()
plt.show()

# 疑似ディープラーニングの場合
# モデルの定義
class Net2(nn.Module):
    def __init__(self):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()
        # 出力層の定義
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)
    # 予測関数の定義
    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        return x3

# 学習率
lr = 0.01
# インスタンス生成　(パラメータ値初期化)
net2 = Net2()
# 最適化アルゴリズム: 勾配降下法
optimizer = optim.SGD(net2.parameters(), lr=lr)
# 損失関数： 最小二乗誤差
criterion = nn.MSELoss()
# 繰り返し回数
num_epochs = 10000
# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

# 繰り返し計算メインループ
for epoch in range(num_epochs):
    # 勾配値初期化
    optimizer.zero_grad()
    # 予測計算
    outputs = net2(inputs)
    # 誤差計算
    loss = criterion(outputs, labels)
    #勾配計算
    loss.backward()
    # 勾配降下法の適用
    optimizer.step()
    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')

# 結果のグラフ化
labels_pred2 = net2(inputs_test)

plt.title('隠れ層２層　活性化関数なし')
plt.scatter(inputs_test[:,0].data, labels_pred2[:,0].data, c='b', label='予測値')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='正解値')
plt.legend()
plt.show()

# ディープラーニング(活性化関数あり)の場合
# モデルの定義

class Net3(nn.Module):
    def __init__(self):
        #  親クラスnn.Modulesの初期化呼び出し
        super().__init__()
        # 出力層の定義
        self.l1 = nn.Linear(1, 10)
        self.l2 = nn.Linear(10, 10)
        self.l3 = nn.Linear(10,1)
        self.relu = nn.ReLU(inplace=True)
    # 予測関数の定義
    def forward(self, x):
        x1 = self.relu(self.l1(x))
        x2 = self.relu(self.l2(x1))
        x3 = self.l3(x2)
        return x3

# 学習率
lr = 0.01
# インスタンス生成　(パラメータ値初期化)
net3 = Net3()
# 最適化アルゴリズム: 勾配降下法
optimizer = optim.SGD(net3.parameters(), lr=lr)
# 損失関数： 最小二乗誤差
criterion = nn.MSELoss()
# 繰り返し回数
num_epochs = 10000
# 評価結果記録用 (損失関数値のみ記録)
history = np.zeros((0,2))

# 繰り返し計算メインループ
for epoch in range(num_epochs):
    # 勾配値初期化
    optimizer.zero_grad()
    # 予測計算
    outputs = net3(inputs)
    # 誤差計算
    loss = criterion(outputs, labels)
    #勾配計算
    loss.backward()
    # 勾配降下法の適用
    optimizer.step()
    # 100回ごとに途中経過を記録する
    if ( epoch % 100 == 0):
        history = np.vstack((history, np.array([epoch, loss.item()])))
        print(f'Epoch {epoch} loss: {loss.item():.5f}')
# 結果の可視化
labels_pred3 = net3(inputs_test)

plt.title('隠れ層２層　活性化関数あり')
plt.scatter(inputs_test[:,0].data, labels_pred3[:,0].data, c='b', label='予測値')
plt.scatter(inputs_test[:,0].data, labels_test[:,0].data, c='k', marker='x',label='正解値')
plt.legend()
plt.show()

