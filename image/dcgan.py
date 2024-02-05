#
# 現場で使えるPyTorch開発入門
# Chap 4.5, p.93
# DCGANによる画像生成
# Oxford 102(花のデータセット)を使う
# https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import tqdm

# preparation of datasets
img_data = ImageFolder("./oxford-102/",
    transform=transforms.Compose([
        transforms.Resize(80),
        transforms.CenterCrop(64),
        transforms.ToTensor()
]))

batch_size = 64
img_loader = DataLoader(img_data, batch_size=batch_size, shuffle=True)

# 潜在特徴ベクトルzを100次元へ
nz = 100
ngf = 32

class GNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False), #1
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), #2
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), #3
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), #4
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False), #5
            nn.Tanh()
        )

    def forward(self, x):
        out = self.main(x)
        return out

# 識別モデルの作成
ndf = 32

class DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False), #1
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), #2
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), #3
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), #4
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False) #5
        )

    def forward(self, x):
        out = self.main(x)
        return out.squeeze()

# 訓練関数
d = DNet().to("cuda:0")
g = GNet().to("cuda:0")

opt_d = optim.Adam(d.parameters(), lr=0.002, betas=(0.5, 0.999))
opt_g = optim.Adam(g.parameters(), lr=0.002, betas=(0.5, 0.999))

ones = torch.ones(batch_size).to("cuda:0")
zeros = torch.zeros(batch_size).to("cuda:0")
loss_f = nn.BCEWithLogitsLoss()
fixed_z = torch.randn(batch_size, nz, 1, 1).to("cuda:0")

from statistics import mean

def train_dcgan(g, d, opt_g, opt_d, loader):
    log_loss_g = []
    log_loss_d = []
    for real_img, _ in tqdm.tqdm(loader):
        batch_len = len(real_img)
        real_img = real_img.to("cuda:0")

        z = torch.randn(batch_len, nz, 1, 1).to("cuda:0")
        fake_img = g(z)

        fake_img_tensor = fake_img.detach()

        out = d(fake_img)
        loss_g = loss_f(out, ones[: batch_len])
        log_loss_g.append(loss_g.item())

        d.zero_grad(), g.zero_grad()
        loss_g.backward()
        opt_g.step()

        real_out = d(real_img)
        loss_d_real = loss_f(real_out, ones[: batch_len])

        fake_img = fake_img_tensor

        fake_out = d(fake_img_tensor)
        loss_d_fake = loss_f(fake_out, zeros[: batch_len])

        loss_d = loss_d_real + loss_d_fake
        log_loss_d.append(loss_d.item())

        d.zero_grad(), d.zero_grad()
        loss_d.backward()
        opt_d.step()

    return mean(log_loss_g), mean(log_loss_d)

# 訓練実行
from torchvision.utils import save_image

for epoch in range(300):
    train_dcgan(g, d, opt_g, opt_d, img_loader)
    if epoch & 10 == 0:
        torch.save(
            g.state_dict(),
            "./outputs/g_{:03d}.prm".format(epoch),
            pickle_protocol=4)
        torch.save(
            d.state_dict(),
            "./outputs/d_{:03d}.prm".format(epoch),
            pickle_protocol=4)
        generated_img = g(fixed_z)
        save_image(generated_img,
            "./outputs/{:03d}.jpg".format(epoch))
