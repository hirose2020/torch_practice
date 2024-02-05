#
# MNISTから始める深層学習
# Chap.1

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# データセットの変換処理 
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5), std=(0.5))
])

# データセットの読み込み
dataset_train = MNIST(root="./data", train=True, download=True, transform=transform)

# データセットの準備
import os
#train_loader = DataLoader(dataset_train, batch_size=50,
#    shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
train_loader = DataLoader(dataset_train, batch_size=10, shuffle=False)

data_iter = iter(train_loader)
images, labels = data_iter.next()

import cv2

for index, (image, label) in enummerate(zip(images, labels)):
    npimg = image.to("cpu").detach().numpy()
    npimg = npimg.transpose((1, 2, 0))
    npimg *= 255
    print(f"index = {index}, label = {label}")

    cv2.imshow("mnist image", npimg)
#    k = cv2.waitKey(-1)
#    if k == ord("q"):
#        exit(1)

