from torchvision import models
alexnet = models.AlexNet()

# 101層畳み込みNN
resnet = models.resnet101(pretrained=True)

# 前処理パイプライン
from torchvision import transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

from PIL import Image
img = Image.open("bobby.jpg")
#img.show()

# パイプラインに通して前処理
img_t = preprocess(img)

# 正規化してテンソルへ変換
import torch
batch_t = torch.unsqueeze(img_t, 0)

# 実行 推論モード
resnet.eval()
out = resnet(batch_t)

## 訓練中ラベルの取り出し
with open("imagenet_classes.txt") as f:
    labels = [line.strip() for line in f.readlines()]
## indexを使ってラベルにアクセス
_, index = torch.max(out, 1)

## テンソルの取り出しと正規化->予測の信頼度
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item())

_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])

# ゴールデンレトリバであることを当てる

