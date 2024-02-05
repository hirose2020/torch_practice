#
# Pytorchによる自然言語処理プログラミング
# Chap.1, p20-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# prepration of datasets
iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.5)

xtrain = torch.from_numpy(xtrain).type("torch.FloatTensor")
xtest = torch.from_numpy(xtest).type("torch.FloatTensor")
ytrain = torch.from_numpy(ytrain).type("torch.LongTensor")
ytest = torch.from_numpy(ytest).type("torch.LongTensor")
xtrain = xtrain.to(device)
xtest = xtest.to(device)
ytrain = ytrain.to(device)
ytest = ytest.to(device)

# definition of model
class MyIris(nn.Module):
    def __init__(self):
        super(MyIris, self).__init__()
        self.l1 = nn.Linear(4, 6)
        self.l2 = nn.Linear(6, 3)
    def forward(self, x):
        h1 = torch.sigmoid(self.l1(x))
        h2 = self.l2(h1)
        return h2

model = MyIris().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

# training
model.train()
for i in range(1000):
    output = model(xtrain)
    loss = criterion(output, ytrain)
    print(i, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# save of model
torch.save(model.state_dict(), "myiris.model")
# loda of model
# model.load_state_dict(torch.load("myiris.model")

# varidation
model.eval()
with  torch.no_grad():
    output1 = model(xtest)
    ans = torch.argmax(output1, 1)
    print(((ytest == ans).sum().float() / len(ans)).item())
