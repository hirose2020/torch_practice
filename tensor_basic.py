import torch

# python listからpytorch tensorへ
a = [4., 1., 5., 3., 2., 1.]
points = torch.tensor(a)
print(points)

# tensorから数値へ
f = float(points[0])
print(f)

# ストレージへのインデックス化
points = torch.tensor(
    [[4.0, 1.0],
     [5.0, 3.0],
     [2.0, 1.0]]
)
s = points.storage() # strogeは常に１次元
print(s)

# storageを変更すると参照しているtensorも変更される
s[0] = 2.0
print(s)

# 使用しているテンソルの中身を変更して動作（インプレース操作）
a = torch.ones(3, 2)
print(a)
a.zero_()
print(a)

