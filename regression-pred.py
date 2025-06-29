import os
import warnings
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms
import torch
import torch.nn as nn
import torchvision.models as models
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "")


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # 加载RESNET50模型，不使用预训练参数
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 1000)

        # 定义新的层和激活函数，并添加到类中
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.last_linear = nn.Linear(1000, 1)


    def forward(self, x):

        x = self.resnet50(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)

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

input_path = r"/Database/coastal data/test_5000"  #待预测图片文件夹
image_paths = [os.path.join(input_path, f) for f in os.listdir(input_path) if f.endswith(".jpg") or f.endswith(".png")]


try:
    predictions = []
    for img_path in tqdm(image_paths):
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)


        model.eval()
        with torch.no_grad():
            output = model(image_tensor).cpu().numpy()[0]  # 获取输出转换为NP

        predictions.append({
            "image_name": os.path.basename(img_path),
            "score1": output[0]
        })


    # 打印前5个预测结果验证
    for i, pred in enumerate(predictions[:5]):
        print(pred)


    df = pd.DataFrame(predictions)
    df.to_csv("predictions.csv", index=False)   #预测结果表格的名字
    print("Predictions saved to predictions.csv")

except Exception as e:
    print("An error occurred during prediction:", e)
