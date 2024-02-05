# PyTorchのネットワークモデルを使って線形回帰
# https://watlab-blog.com/2021/03/29/pytorch-linear-regression/

import torch
import numpy as np
from matplotlib import pyplot as plt

def linear_regression(dimension, iteration, lr, x, y):
    net = torch.nn.Linear(in_features=dimension, out_features=1, bias=False)  # ネットワークに線形結合モデルを設定
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)                      # 最適化にSGDを設定
    E = torch.nn.MSELoss()                                                    # 損失関数にMSEを設定

    # 学習ループ
    losses = []
    for i in range(iteration):
        optimizer.zero_grad()                                                 # 勾配情報を0に初期化
        y_pred = net(x)                                                       # 予測
        loss = E(y_pred.reshape(y.shape), y)                                  # 損失を計算(shapeを揃える)
        loss.backward()                                                       # 勾配の計算
        optimizer.step()                                                      # 勾配の更新
        losses.append(loss.item())                                            # 損失値の蓄積
        print(list(net.parameters()))

    # 回帰係数を取得して回帰直線を作成
    w0 = net.weight.data.numpy()[0, 0]
    w1 = net.weight.data.numpy()[0, 1]
    x_new = np.linspace(np.min(x.T[1].data.numpy()), np.max(x.T[1].data.numpy()), len(x))
    y_curve = w0 + w1 * x_new

    # グラフ描画
    plot(x.T[1], y, x_new, y_curve, losses)
    return net, losses

def plot(x, y, x_new, y_pred, losses):
    # ここからグラフ描画-------------------------------------------------
    # フォントの種類とサイズを設定する。
    plt.rcParams['font.size'] = 14
#    plt.rcParams['font.family'] = 'Times New Roman'

    # 目盛を内側にする。
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    # グラフの上下左右に目盛線を付ける。
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(121)
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax2 = fig.add_subplot(122)
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')

    # 軸のラベルを設定する。
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('E')

    # スケール設定
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 30)
    ax2.set_xlim(0, 1000)
    ax2.set_ylim(0.1, 100)
    ax2.set_yscale('log')

    # データプロット
    ax1.scatter(x, y, label='dataset')
    ax1.plot(x_new, y_pred, color='red', label='PyTorch result')
    ax2.plot(np.arange(0, len(losses), 1), losses)
    ax2.text(600, 30, 'Loss=' + str(round(losses[len(losses)-1], 2)), fontsize=16)
    ax2.text(600, 50, 'Iteration=' + str(round(len(losses), 1)), fontsize=16)

    # グラフを表示する。
    ax1.legend()
    fig.tight_layout()
    plt.show()
    plt.close()
    # -------------------------------------------------------------------

# サンプルデータ
x = np.random.uniform(0, 10, 100)                   # x軸をランダムで作成
y = np.random.uniform(0.2, 1.9, 100) + x + 10       # yを分散した線形データとして作成
x = torch.from_numpy(x.astype(np.float32)).float()  # xをテンソルに変換
y = torch.from_numpy(y.astype(np.float32)).float()  # yをテンソルに変換
X = torch.stack([torch.ones(100), x], 1)            # xに切片用の定数1配列を結合

# 線形回帰を実行
net, losses = linear_regression(dimension=2, iteration=1000, lr=0.01, x=X, y=y)
