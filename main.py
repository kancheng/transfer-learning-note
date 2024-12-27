import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# 加載 VGG-19 模型，僅提取特徵部分
class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.layers = nn.Sequential(*[vgg[i] for i in range(12)])  # 取前幾層特徵
        for param in self.layers.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features

# 計算格拉姆矩陣
def gram_matrix(features):
    (b, c, h, w) = features.size()
    features = features.view(b, c, h * w)
    gram = torch.bmm(features, features.transpose(1, 2))  # bmm: 批量矩陣乘法
    return gram / (c * h * w)

# 設置設備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 圖像預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 使用內建的 CIFAR-10 數據集
dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 獲取兩張隨機圖像
data_iter = iter(dataloader)
img1, _ = next(data_iter)  # 圖像1
img2, _ = next(data_iter)  # 圖像2

img1 = img1[0].unsqueeze(0).to(device)  # 提取批次內的第一張圖像
img2 = img2[0].unsqueeze(0).to(device)

# 初始化模型
vgg_features = VGGFeatures().to(device)

# 提取特徵並計算格拉姆矩陣
with torch.no_grad():
    features1 = vgg_features(img1)
    features2 = vgg_features(img2)
    gram1 = [gram_matrix(f) for f in features1]
    gram2 = [gram_matrix(f) for f in features2]

# 計算風格相似性（格拉姆矩陣的歐幾里得距離）
style_loss = 0
for g1, g2 in zip(gram1, gram2):
    style_loss += torch.norm(g1 - g2, p='fro')  # Frobenius norm

print(f"Style similarity (lower is more similar): {style_loss.item()}")