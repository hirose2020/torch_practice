#
# 現場で使えるPytorch開発入門
# p.59 Chap 2.4.2
# irisデータのロジスティックス回帰

# データセットの準備
import torch
from sklearn.datasets import load_iris
iris = load_iris()
x = iris.data[:100]
y = iris.target[:100]

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# nn, optimモジュールで線形回帰
from torch import nn, optim

net = nn.Linear(4, 1)
optimizer = optim.SGD(net.parameters(), lr=0.25)
loss_fn = nn.BCEWithLogitsLoss() # クロスエントロピー

losses = []

for epoc in range(100):
    optimizer.zero_grad() # 前回のbarkwardで計算された勾配を削除
    y_pred = net(x)
    # y_predは(n,1)のshapeなので、(n,)に直す必要がある
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

print(list(net.parameters()))

import matplotlib.pyplot as plt
plt.plot(losses)
plt.show()

# モデルの作成
h = net(x) # 線型結合の結果
# sigmoid関数を作用させた結果はy=1の確率を表す
prob = nn.functional.sigmoid(h)
# 確率0.5以上のものをクラス１と予想
y_pred = prob >= 0.5

