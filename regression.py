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
from model import *


warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(torch.cuda.is_available())

# 1. 组件数据集、处理
class coast_Dataset(Dataset):
    def __init__(self, data_dir, dataframe, transform=None):
        self.data_dir = data_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = float(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label


data_dir = "图片文件夹"
label_file = "数据表格.csv"  #第一列为图片名，第二类标签分数（建议归一化）


dataframe = pd.read_csv(label_file)
transform = transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 调整图像大小
    torchvision.transforms.ToTensor(),         # 将图像转换为张量
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 实例化数据集对象
dataset = coast_Dataset(data_dir, dataframe=dataframe, transform=transform)

# 2. 切分train ＆ test
# 定义训练集和测试集的大小
train_size = int(0.8 * len(dataset))  #训练集测试集比例
test_size = len(dataset) - train_size

# 划分训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])



# 遍历数据集并打印图像及其对应的标签
"""
for i in range(len(train_dataset)):
    image, label, img_name = train_dataset[i]
    print("Sample {}: Label - {}, name:{} ".format(i, label, img_name))
"""


# 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  #batch_size一次取样本数量，显存决定
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# 将训练集和测试集中的数据类型转换为 Float 类型
train_loader = [(data.to(torch.float32), target.to(torch.float32)) for data, target in train_loader]
test_loader = [(data.to(torch.float32), target.to(torch.float32)) for data, target in test_loader]


#搭建神经网络




mod = ResNet50()
mod = mod.to(device, dtype=torch.float32)
loss_fn = torch.nn.SmoothL1Loss()
loss_fn = loss_fn.to(device, dtype=torch.float32)
learning_rate = 0.0001   # 学习率
optimizer = torch.optim.Adam(mod.parameters(),lr=learning_rate)

# scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.1)  #学习率逐渐下降，可选择开启


lowest_mae = float('50')  #设置初始MAE阈值
model_save_dir = r"/checkpoints"   #模型保存位置
os.makedirs(model_save_dir, exist_ok=True)
Writer = SummaryWriter(r"logs/xxxxxx")  #训练过程写入tensorboard，每次训练完要重命名保存好



epochs = 200   #训练轮数
for i in range(epochs):
    print("-----Epoch:{}/{}-----".format(i+1, epochs))
    mod.train()
    total_train_loss = 0
    for data in tqdm(train_loader):
        images, labels = data
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        outputs = mod(images)
        loss = loss_fn(outputs, labels.unsqueeze(1))

        total_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 计算整个epoch的平均损失
    average_train_loss = total_train_loss / len(train_loader)

    # 重置train_predictions和train_labels以累积整个epoch的数据
    train_predictions = []
    train_labels = []

    for data in train_loader:
        images, labels = data
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        outputs = mod(images)
        # 将预测结果和真实标签保存下来，使用.detach().cpu().numpy()来避免梯度追踪
        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.detach().cpu().numpy())

    # 计算训练集上的MAE和RMSE
    train_mae = mean_absolute_error(train_labels, train_predictions)
    train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))

    print("Train Loss: {:.4f}".format(average_train_loss))
    print("Train MAE: {:.4f}".format(train_mae))
    print("Train RMSE: {:.4f}".format(train_rmse))


    # 使用TensorBoard记录训练集上的MAE和RMSE
    Writer.add_scalar("train_Loss", average_train_loss, i)
    Writer.add_scalar("train_MAE", train_mae, i)
    Writer.add_scalar("train_RMSE", train_rmse, i)

    # scheduler.step()  #学习率下降，与99行保持同步

    mod.eval()
    all_predictions = []
    all_labels = []
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = mod(images)
            loss = loss_fn(outputs, labels.unsqueeze(1))
            total_test_loss = total_test_loss + loss.item()

            # 将预测结果添加到列表中
            all_predictions.extend(outputs.cpu().numpy())
            # 将真实标签添加到列表中
            all_labels.extend(labels.cpu().numpy())

    mae = mean_absolute_error(all_labels, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_labels, all_predictions))
    print("Test_Loss: {}".format(total_test_loss))
    print("Test_MAE: {}".format(mae))
    print("Test_RMSE: {}".format(rmse))

    Writer.add_scalar("test_Loss", total_test_loss, i)
    Writer.add_scalar("test_MAE", mae, i)
    Writer.add_scalar("test_RMSE", rmse, i)

    #保持保存精度最高的模型
    if mae < lowest_mae:
        lowest_mae = mae
        torch.save(mod.state_dict(), os.path.join(model_save_dir, f"best_model_MAE{mae}.pt"))
        print(f"Best model with MAE {lowest_mae} saved to {model_save_dir}")



