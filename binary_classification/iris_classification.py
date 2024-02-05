#
# irisデータ：３種類のアヤメの花に対して、花弁とがく片の長さと幅の測定値
# （２種類分だけを利用）
#
# 入力テンソル: [2]
# 予測モデル: 線形関数nn.Linear wight[1, 2], bias[1], シグモイド関数
# 出力テンソル: [1]
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#### データの準備
iris = load_iris()

x_data = iris.data[:100, :2] # Setosa, Versicolourのみのデータに絞る
y_data = iris.target[:100]

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30, random_state=123)

#### モデル定義 : ２入力１出力のロジスティクス回帰
n_input = x_train.shape[1] # 入力次元数
n_output = 1 # 出力次元数

class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.l1 = nn.Linear(n_input, n_output)
        self.sigmoid = nn.Sigmoid()

        self.l1.weight.data.fill_(1.0) # 初期値を1にする
        self.l1.bias.data.fill_(1.0)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.sigmoid(x1)
        return x2

net = Net(n_input, n_output)
print(net)
print(summary(net))

#### 最適化アルゴリズム・損失関数の定義
lr = 0.01
criterion = nn.BCELoss() # ２値分類用交差エントロピー関数
optimizer = optim.SGD(net.parameters(), lr=lr) # 勾配降下法

#### データのtensor化
inputs = torch.tensor(x_train).float()
labels = torch.tensor(y_train).float()
inputs_test = torch.tensor(x_test).float()
labels_test = torch.tensor(y_test).float()

labels1 = labels.view((-1, 1)) # N行1列の行列へ変更
labels1_test = labels_test.view((-1, 1)) # N行1列の行列へ変更

#### 学習
num_epochs = 10000
history = np.zeros((0, 5))

for epoch in range(num_epochs):
    # 訓練
    optimizer.zero_grad() # 勾配値初期化
    outputs = net(inputs) # 予測計算
    loss = criterion(outputs, labels1) # 損失計算
    loss.backward() # 勾配計算
    optimizer.step() # パラメータ修正
    train_loss = loss.item() # 損失の保存
    predicted = torch.where(outputs < 0.5, 0, 1) # 予測ラベル計算
    train_acc = (predicted == labels1).sum() / len(y_train) # 精度計算
    # 予測
    outputs_test = net(inputs_test)
    loss_test = criterion(outputs_test, labels1_test)
    val_loss = loss_test.item()
    predicted_test = torch.where(outputs_test < 0.5, 0, 1)
    val_acc = (predicted_test == labels1_test).sum() / len(y_test)
    
    if ( epoch % 10 == 0):
        print (f'Epoch [{epoch}/{num_epochs}], loss: {train_loss:.5f} acc: {train_acc:.5f} val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}')
        item = np.array([epoch, train_loss, train_acc, val_loss, val_acc])
        history = np.vstack((history, item))

#### 学習曲線
plt.plot(history[:, 0], history[:, 1], "b", label="training")
plt.plot(history[:, 0], history[:, 3], "k", label="varidation")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.title("loss curve")
plt.legend()
plt.show()

#### 決定境界表示
# パラメータの取得

bias = net.l1.bias.data.numpy()
weight = net.l1.weight.data.numpy()
print(f'BIAS = {bias}, WEIGHT = {weight}')

# 決定境界描画用 x1の値から x2の値を計算する
def decision(x):
    return(-(bias + weight[0, 0] * x)/ weight[0, 1])

# 散布図のx1の最小値と最大値
xl = np.array([x_test[:, 0].min(), x_test[:, 0].max()])
yl = decision(xl)

# 結果確認
#print(f'xl = {xl}  yl = {yl}')

x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.scatter(x_t0[:, 0], x_t0[:, 1], marker="x", c="b", s=50, label="class 0")
plt.scatter(x_t1[:, 0], x_t1[:, 1], marker="x", c="k", s=50, label="class 1")
plt.plot(xl, yl, c='b')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()
plt.show()
