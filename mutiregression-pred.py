import os
import warnings
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

# mods = torchvision.models.resnet50(pretrained=None)
# mods.load_state_dict(torch.load(r'E:\Py Ai all\12.22 practice\checkpoints\muti_regre1'))

device = torch.device("cuda" if torch.cuda.is_available() else "")


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # 加载预训练的 ResNet50 模型，不使用预训练参数
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc.add_module('dropout', nn.Dropout(0.5))
        self.resnet50.fc.add_module('last_linear', nn.Linear(1000,6))

    def forward(self, x):
        x = self.resnet50(x)
        x = torch.flatten(x, 1)  # 将输出展平为一维张量
        x = self.resnet50.fc.dropout(x)
        x = self.resnet50.fc.last_linear(x)  # 使用新添加的线性层处理输出
        return x

model = ResNet50()
model.load_state_dict(torch.load(r''))  #加载模型
# print(model)

model = model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

input_path = r"待预测图片的文件夹"
image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".jpg") or f.endswith(".png")]

# 预测循环
try:
    predictions = []
    for img_path in image_paths:
        # 打开图片并进行预处理
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)  # 确保图像张量在正确的设备上

        # 进行预测
        model.eval()
        with torch.no_grad():
            output = model(image_tensor).cpu().numpy()[0]  # 获取输出并转换为NumPy数组

        # 保存图片名和预测分数
        predictions.append({
            "image_name": os.path.basename(img_path),
            "score1": output[0], "score2": output[1], "score3": output[2],
            "score4": output[3], "score5": output[4], "score6": output[5]
        })

    # 打印前5个预测结果进行验证
    for i, pred in enumerate(predictions[:5]):
        print(pred)



    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", index=False)  #输出文件夹
    print("Predictions saved to predictions.csv")

except Exception as e:
    print("An error occurred during prediction:", e)
