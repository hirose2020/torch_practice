# pytorchで線型回帰
#
# https://rightcode.co.jp/blog/information-technology/pytorch-automatic-differential-linear-regression

import numpy as np
import matplotlib.pyplot as plt
import torch

x = torch.tensor(5.)
w = torch.tensor(2., requires_grad=True) # 微分の対象とする
b = torch.tensor(1., requires_grad=True)

print('x =', x)
print('w =', w)
print('b =', b)
 
# 教師データ 
N = 200
x = np.random.rand(N)*30-15
y = 2*x + np.random.randn(N)*5

x = x.astype(np.float32)
y = y.astype(np.float32)

# データを描画
plt.scatter(x, y, marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# torchに変換
x = torch.from_numpy(x)
y = torch.from_numpy(y)

def model(x):
    return w*x + b

# 損失関数　平均２乗平均
def mse(p, y):
    return ((p-y)**2).mean()

# 学習率
lr = 1.0e-4
 
# 変数を初期化します
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
 
losses = []
for epoch in range(3000):
    # 線形モデルによる値の予測
    p = model(x)
    
    # 損失値と自動微分
    loss = mse(p, y)
    loss.backward()
    
    # グラディエントを使って変数`w`と`b`の値を更新する。
    with torch.no_grad():
        w -= w.grad * lr
        b -= b.grad * lr
        w.grad.zero_()
        b.grad.zero_()
 
    # グラフ描画用
    losses.append(loss.item())

def draw_linear_regression(x, y, p):
    # PyTorchのTensorからNumpyに変換
    x = x.numpy()
    y = y.numpy()
    p = p.detach().numpy()

    plt.scatter(x, y, marker='.')
    plt.scatter(x, p, marker='o')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
draw_linear_regression(x, y, p)