# MNISTを使った数字認識
# https://github.com/makaishi2/pytorch_book_info/blob/main/notebooks/ch08_dl.ipynb
# p278-
# inputs(torch.Size[784]) -> net1(隠れ層のノード数=128, relu) -> outputs(torch.Size[10])

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary
from torchviz import make_dot

# GPUの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MNISTデータの取得
import torchvision.datasets as datasets
data_root = "./data"
train_set0 = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
)

# テンソル化
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(), # データのテンソル化
    transforms.Normalize(0.5, 0.5), # データの正規化
    transforms.Lambda(lambda x: x.view(-1)), # １階テンソルに変換
])
train_set = datasets.MNIST(
    root=data_root,
    train=True,
    download=True,
    transform=transform
)
test_set = datasets.MNIST(
    root=data_root,
    train=False,
    download=True,
    transform=transform
)

# ミニバッチ用データ生成
from torch.utils.data import DataLoader
batch_size = 500
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
)
test_loader = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
)

# モデルの定義
# 入力784, 隠れ層１つ、出力10
class Net(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super().__init__()
        # 隠れ層の定義
        self.l1 = nn.Linear(n_input, n_hidden)
        # 出力層の定義
        self.l2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.l1(x)
        x2 = self.relu(x1)
        x3 = self.l2(x2)
        return x3

# モデル変数の生成
torch.manual_seed(123)
torch.cuda.manual_seed(123)

n_input = 784
n_hidden = 128
n_output = 10
net = Net(n_input, n_output, n_hidden).to(device)
lr = 0.01 # 学習率
optimizer = torch.optim.SGD(net.parameters(), lr=lr) # アルゴリズム: 勾配降下法
criterion = nn.CrossEntropyLoss() # 損失関数： 交差エントロピー関数

# モデル内パラメータの確認
#for params in net.named_parameters():
#    print(params)
#print(net)    

# 予測計算 GPUへ転送
for images, labels in train_loader:
    break
inputs = images.to(device)
labels = labels.to(device)
outputs = net(inputs)
#print(outputs)

# 損失計算
loss = criterion(outputs, labels)
print(loss.item())
#g = make_dot(loss, params=dict(net.named_parameters()))
#g.format = "png"
#g.render("./NeuralNet")
"""
# 勾配計算
loss.backward()
w = net.to("cpu")
print(w.l1.weight.grad.numpy())
print(w.l1.bias.grad.numpy())
print(w.l2.weight.grad.numpy())
print(w.l2.bias.grad.numpy())

# パラメータ修正
optimizer.step()
print(w.l1.weight.grad.numpy())
print(w.l1.bias.grad.numpy())
"""

# 乱数の固定化
torch.manual_seed(123)
torch.cuda.manual_seed(123)
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms = True

# 学習率
lr = 0.01
# モデルインスタンス生成
net = Net(n_input, n_output, n_hidden).to(device)
# 損失関数： 交差エントロピー関数
criterion = nn.CrossEntropyLoss()
# 最適化関数: 勾配降下法
optimizer = optim.SGD(net.parameters(), lr=lr)
# 繰り返し回数
num_epochs = 100
# 評価結果記録用
history = np.zeros((0,5))

# tqdmライブラリのインポート
from tqdm import tqdm
# 繰り返し計算メインループ
for epoch in range(num_epochs):
    # 1エポックあたりの正解数(精度計算用)
    n_train_acc, n_val_acc = 0, 0
    # 1エポックあたりの累積損失(平均化前)
    train_loss, val_loss = 0, 0
    # 1エポックあたりのデータ累積件数
    n_train, n_test = 0, 0
    # 訓練フェーズ
    for inputs, labels in tqdm(train_loader):
        # 1バッチあたりのデータ件数
        train_batch_size = len(labels)
        # 1エポックあたりのデータ累積件数
        n_train += train_batch_size
        # GPUヘ転送
        inputs = inputs.to(device)
        labels = labels.to(device)
        #勾配の初期化
        optimizer.zero_grad()
        # 予測計算
        outputs = net(inputs)
        # 損失計算
        loss = criterion(outputs, labels)
        # 勾配計算
        loss.backward()
        # パラメータ修正
        optimizer.step()
        # 予測ラベル導出
        predicted = torch.max(outputs, 1)[1]
        # 平均前の損失と正解数の計算
        # lossは平均計算が行われているので平均前の損失に戻して加算
        train_loss += loss.item() * train_batch_size 
        n_train_acc += (predicted == labels).sum().item() 

    #予測フェーズ
    for inputs_test, labels_test in test_loader:
        # 1バッチあたりのデータ件数
        test_batch_size = len(labels_test)
        # 1エポックあたりのデータ累積件数
        n_test += test_batch_size

        inputs_test = inputs_test.to(device)
        labels_test = labels_test.to(device)
            
        # 予測計算
        outputs_test = net(inputs_test)
        # 損失計算
        loss_test = criterion(outputs_test, labels_test)
        #予測ラベル導出
        predicted_test = torch.max(outputs_test, 1)[1]
        #  平均前の損失と正解数の計算
        # lossは平均計算が行われているので平均前の損失に戻して加算
        val_loss +=  loss_test.item() * test_batch_size
        n_val_acc +=  (predicted_test == labels_test).sum().item()

    # 精度計算
    train_acc = n_train_acc / n_train
    val_acc = n_val_acc / n_test
    # 損失計算
    ave_train_loss = train_loss / n_train
    ave_val_loss = val_loss / n_test
    # 結果表示
    print (f'Epoch [{epoch+1}/{num_epochs}], loss: {ave_train_loss:.5f} acc: {train_acc:.5f} val_loss: {ave_val_loss:.5f}, val_acc: {val_acc:.5f}')
    # 記録
    item = np.array([epoch+1 , ave_train_loss, train_acc, ave_val_loss, val_acc])
    history = np.vstack((history, item))

#損失と精度の確認
print(f'初期状態: 損失: {history[0,3]:.5f} 精度: {history[0,4]:.5f}' )
print(f'最終状態: 損失: {history[-1,3]:.5f} 精度: {history[-1,4]:.5f}' )

# 学習曲線の表示 (精度)
plt.rcParams['figure.figsize'] = (9,8)
plt.plot(history[:,0], history[:,2], 'b', label='訓練')
plt.plot(history[:,0], history[:,4], 'k', label='検証')
plt.xlabel('繰り返し回数')
plt.ylabel('精度')
plt.title('学習曲線(精度)')
plt.legend()
plt.show()

# 最初の50件でイメージを「正解値:予測値」と表示
plt.figure(figsize=(10, 8))
for i in range(50):
  ax = plt.subplot(5, 10, i + 1)
    
  # numpyに変換
  image = images[i]
  label = labels[i]
  pred = predicted[i]
  if (pred == label):
    c = 'k'
  else:
    c = 'b'
    
  # imgの範囲を[0, 1]に戻す
  image2 = (image + 1)/ 2
    
  # イメージ表示
  plt.imshow(image2.reshape(28, 28),cmap='gray_r')
  ax.set_title(f'{label}:{pred}', c=c)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()


