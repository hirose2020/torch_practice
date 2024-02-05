# torch.nnを使って、複雑な関数の回帰をデモ
# １次元データだけど、複雑なデータ
# GPUで計算

# データの用意
import numpy as np
N = 2000
x = np.linspace(-10, 10, N)
y = np.zeros((N), dtype=np.float32)

for i in range(N):
    if x[i] < -1:
        y[i] = 1
    elif (x[i] >= -1 and x[i] < 1):
        y[i] = 3 * x[i] + 4
    elif (x[i] >= 1 and x[i] < 3):
        y[i] = 7
    elif (x[i] >= 3 and x[i] < 4):
        y[i] = -6 * x[i] + 25
    elif x[i] >= 4:
        y[i] = x[i] - 3

# ノイズを載せる
import random
for i in range(N):
    y[i] = y[i] + random.uniform(-0.5, 0.5)

# 学習データ、テストデータの用意
# テンソル化
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

xy = np.zeros((N, 2))
for i in range(N):
    xy[i][0] = x[i]
    xy[i][1] = y[i]

rng = np.random.default_rng(1)
rng.shuffle(xy)
threshold = int(np.round(N * 0.8))
xy_train = xy[:threshold]
xy_test = xy[threshold:]

x_train = torch.tensor(xy_train[:, 0:1], dtype=torch.float32).to(device)
y_train = torch.tensor(xy_train[:, 1], dtype=torch.float32).view(-1, 1).to(device)
x_test = torch.tensor(xy_test[:, 0:1], dtype=torch.float32).to(device)
y_test = torch.tensor(xy_test[:, 1], dtype=torch.float32).view(-1, 1).to(device)

# NNモデル定義
nn1 = 32
nn2 = 16
nn3 = 8
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(1, nn1)
        self.fc2 = nn.Linear(nn1, nn2)
        self.fc3 = nn.Linear(nn2, nn3)
        self.fc4 = nn.Linear(nn3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # linear activation
        return x

model = RegressionModel().to(device)

# 損失関数とオプティマイザ
lr = 0.005
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr)

# データローダーの準備
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

# モデルの学習
epochs = 1000
loss_history = []
for epoch in range(epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device) # GPUへ移す
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# 学習過程でエポックごとの損失関数値をプロット
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(loss_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('loss')
ax.set_xlim(0, 1000)
#ax.set_ylim(-0.01, 0.5)
plt.show()

# 予測
model.eval()
with torch.no_grad():
    y_pred = model(x_test)

y_test = y_test.to("cpu")
y_pred = y_pred.to("cpu")

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(y_test, y_pred, alpha=0.5)
ax.set_xlabel('y_data')
ax.set_ylabel('y_pred')
#ax.set_xlim(-1.1,1.3)
#ax.set_ylim(-1.1,1.3)
plt.show()


x_test = x_test.to("cpu")
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.scatter(x_test, y_pred, alpha=0.5)
ax.set_xlabel('x_data')
ax.set_ylabel('y_pred')
plt.show()

print("model parameters: ")
print(model.state_dict())

