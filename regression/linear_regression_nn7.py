#
# 現場で使えるPytorch開発入門
# p.33 Chap 2.3

# データの生成
# y = 1 + 2*x1 + 3*x2
import torch
w_true = torch.Tensor([1, 2, 3]) # 真の係数
# 切片を回帰係数に含めるため最初の次元に1を追加
x = torch.cat([torch.ones(100, 1), torch.randn(100, 2)], 1)
# 真の係数とxとの内積を行列とベクトルの積でまとめて計算
y = torch.mv(x, w_true) + torch.randn(100) * 0.5 
w = torch.randn(3, requires_grad=True)

gamma = 0.1 # 学習率

# nn, optimモジュールで線形回帰
from torch import nn, optim

net = nn.Linear(in_features=3, out_features=1, bias=False)
optimizer = optim.SGD(net.parameters(), lr=0.1)
loss_fn = nn.MSELoss()

losses = []

for epoc in range(100):
    optimizer.zero_grad() # 前回のbarkwardで計算された勾配を削除
    y_pred = net(x)
    # y_predは(n,1)のshapeなので、(n,)に直す必要がある
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

for l in losses:
    print(l)

print(list(net.parameters()))
