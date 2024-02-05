#
# 現場で使えるPyTorch開発入門
# Chap 4.4 p.83
# CNN回帰モデルによる画像の高解像度化
# 顔画像Labeld Faces in the Wild(LFW)データセット

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import tqdm

# Dataset, Dataloaderの用意
class DownSizedPairImageFolder(ImageFolder):
    def __init__(self, root, transform=None, large_size=128, small_size=32, **kwds):
        super().__init__(root, transform=transform, **kwds)
        self.large_resizer = transforms.Resize(large_size)
        self.small_resizer = transforms.Resize(small_size)

    def __getitem__(self, index):
        path, _ = self.imgs[index]
        img = self.loader(path)

        large_img = self.large_resizer(img)
        small_img = self.small_resizer(img)

        if self.transform is not None:
            large_img = self.transform(large_img)
            small_img = self.transform(small_img)

        return small_img, large_img

train_data = DownSizedPairImageFolder("./lfw-deepfunneled/train/",
    transform=transforms.ToTensor())
test_data =  DownSizedPairImageFolder("./lfw-deepfunneled/test/",
    transform=transforms.ToTensor())

batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

# モデルの作成
net = nn.Sequential(
    nn.Conv2d(3, 256, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.Conv2d(256, 512, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(512),
    nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(256),
    nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(128),
    nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
    nn.ReLU(),
    nn.BatchNorm2d(64),
    nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)
)

# PSNR
import math
def psnr(mse, max_v=1.0):
    return 10 * math.log10(max_v**2 / mse)

# 評価関数
def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            y_pred = net(x)
        ys.append(y) # ミニバッチごとの予測結果を１つにまとめる
        ypreds.append(y_pred)
    ys = torch.cat(ys)
    ypreds = torch.cat(ypreds)
    score = nn.functional.mse_loss(ypreds, ys).item() # 予測精度計算
    return score

# 訓練関数
def train_net(net, train_loader, test_loader,
         optimizer_cls=optim.Adam, loss_fn=nn.MSELoss(), n_iter=10, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        score = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx = xx.to(device)
            yy = yy.to(device)
            y_pred = net(xx)
            loss = loss_fn(y_pred, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
        train_losses.append(running_loss / len(train_loader))
        val_acc.append(eval_net(net, test_loader, device=device))
        print(epoch, train_losses[-1], psnr(train_losses[-1]), psnr(val_acc[-1]), flush=True)

# 訓練実行
net.to("cuda:0")
train_net(net, train_loader, test_loader, device="cuda:0")

# 画像を拡大して比較
from torchvision.utils import save_image

random_test_loader = DataLoader(test_data, batch_size=4, shuffle=True)
it = iter(random_test_loader)
x, y = next(it)

# Bilinearで拡大
bl_recon = torch.nn.functional.upsample(x, 128, mode="bilinear", align_corners=True)

# CNNで拡大
yp = net(x.to("cuda:0")).to("cpu")

# ファイルに書き出し
save_image(torch.cat([y, bl_recon, yp], 0), "cnn_upscale.jpg", nrow=4)

