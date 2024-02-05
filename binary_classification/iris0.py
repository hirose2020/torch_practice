#!/usr/bin/env python3
#coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

# アヤメデータの半分を訓練、もう半分をテストデータに分割
iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(
    iris.data, iris.target, test_size=0.5)

# PyTorchのテンソルへ変換
xtrain = torch.from_numpy(xtrain).type('torch.FloatTensor')
ytrain = torch.from_numpy(ytrain).type('torch.LongTensor')
xtest = torch.from_numpy(xtest).type('torch.FloatTensor')
ytest = torch.from_numpy(ytest).type('torch.LongTensor')

# バイアスのユニット以外に６つのユニットを置く
# 中間層は３値分類なので３つのユニットになる
class MyIris(nn.Module):
    def __init__(self):
        super(MyIris, self).__init__()
        self.l1 = nn.Linear(4, 6)
        self.l2 = nn.Linear(6, 3)
    # forwardには順方向の計算
    def forward(self, x):
        h1 = torch.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

model = MyIris()
# 最適化アルゴリズムの設定: SGD 最適化勾配降下法
optimizer = optim.SGD(model.parameters(), lr=0.1)
# 損失関数: 識別の問題の場合は交差エントロピーを使う
criterion = nn.CrossEntropyLoss()

# 学習 = 再急降下法
model.train()
for i in range(1000): # 1000エポック繰り返す
    output = model(xtrain) # 順方向の計算
    loss = criterion(output, ytrain)
    print(i, loss.item()) # 誤差が減ることを確認
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step() # optimizerに設定されたパラメータが更新

# モデルの保存    
torch.save(model.state_dict(), 'myiris.model')
# model.load_state_dict(torch.load('myiris.model'))

model.eval()
with torch.no_grad():
    output1 = model(xtest)
    ans = torch.argmax(output1, 1)
    print(((ytest == ans).sum().float() / len(ans)).item()) # 正解率
    





        
