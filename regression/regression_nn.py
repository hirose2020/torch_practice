import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
#import seaborn as sns
import matplotlib.pyplot as plt

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# データファイルの読み込み
df = pd.read_csv("mra-02.csv")

# ペアプロットを描く
#sns.pairplot(df.drop('no', axis=1))
#plt.show()

# データの準備
xy_data = df.loc[:, ['no', 'x1', 'x2', 'x3', 'y']].values

# シャッフルして、トレーニング80%, テスト20%にデータ分割
rng = np.random.default_rng(1)
rng.shuffle(xy_data)
threshold = int(np.round(500 * 0.8))
xy_train = xy_data[:threshold]
xy_test = xy_data[threshold:]

# データをテンソルに変換し、GPUに移動
x_train = torch.tensor(xy_train[:, 1:4], dtype=torch.float32).to(device)
y_train = torch.tensor(xy_train[:, 4], dtype=torch.float32).view(-1, 1).to(device)
x_test = torch.tensor(xy_test[:, 1:4], dtype=torch.float32).to(device)
y_test = torch.tensor(xy_test[:, 4], dtype=torch.float32).view(-1, 1).to(device)

# モデル定義
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # linear activation
        return x

# モデルをGPUに移動
model = RegressionModel().to(device)

# 損失関数とオプティマイザ
lr = 0.001
criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=lr)

# データローダーの準備
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

# モデルの訓練
epochs = 3000
loss_history = []
for epoch in range(epochs):
    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 予測
model.eval()
with torch.no_grad():
    y_pred = model(x_test)  # x_test はすでにGPUに移動している

