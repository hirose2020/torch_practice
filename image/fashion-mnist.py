#
# 現場で使えるPyTorch開発入門
# Chap4.2 p.66
# CNNによる画像分類 Fashion-MNIST

# データの用意
import torch
from torch import nn, optim
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import tqdm

fashion_mnist_train = FashionMNIST("./FashionMNIST", train=True, download=True,
    transform=transforms.ToTensor())
fashion_mnist_test  = FashionMNIST("./FashionMNIST", train=False, download=True,
    transform=transforms.ToTensor())

batch_size = 128
train_loader = DataLoader(fashion_mnist_train, batch_size=batch_size,
    shuffle=True)
test_loader  = DataLoader(fashion_mnist_test, batch_size=batch_size,
    shuffle=False)

# CNNの構築
class FlattenLayer(nn.Module):
    def forward(self, x):
        sizes = x.size()
        return x.view(sizes[0], -1)

conv_net = nn.Sequential(
    nn.Conv2d(1, 32, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(32),
    nn.Dropout2d(0.25),
    nn.Conv2d(32, 64, 5),
    nn.MaxPool2d(2),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.Dropout2d(0.25),
    FlattenLayer()
)

# 確認
test_input = torch.ones(1, 1, 28, 28)
conv_output_size = conv_net(test_input).size()[-1]
print(conv_output_size)

# 2層のMLP
mlp = nn.Sequential(
    nn.Linear(conv_output_size, 200),
    nn.ReLU(),
    nn.BatchNorm1d(200),
    nn.Dropout(0.25),
    nn.Linear(200, 10)
)

net = nn.Sequential(
    conv_net,
    mlp
)

# 評価のヘルパー関数
def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)

    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    acc = (ys == ypreds).float().sum() / len(ys)
    return acc.item()

# 訓練のヘルパー関数
def train_net(net, train_loader, test_loader, optimizer_cls=optim.Adam,
          loss_fn=nn.CrossEntropyLoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)
        train_acc.append(n_acc / n)
        val_acc.append(eval_net(net, test_loader, device))
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)

net.to("cuda:0")
train_net(net, train_loader, test_loader, n_iter=20, device="cuda:0")

