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
        # 使用 iloc 来按位置访问标签数据
        label = torch.tensor(self.dataframe.iloc[idx, 1:7], dtype=torch.float32)  #n个标签为1:n+1， 此处为6个

        if self.transform:
            image = self.transform(image)

        return image, label #, img_name

data_dir = r"图片文件夹"
label_file = r"标签，第2-n列为n-1个标签.csv"


dataframe = pd.read_csv(label_file)



min_labels = dataframe.iloc[:, 1:7].min().min() # 归一化标签数据
max_labels = dataframe.iloc[:, 1:7].max().max()
dataframe.iloc[:, 1:7] = (dataframe.iloc[:, 1:7] - min_labels) / (max_labels - min_labels)
#填充补充缺失值，在归一化之后
dataframe.fillna(0, inplace=True)

# print(dataframe.iloc[:, 1:].apply(pd.isna).sum())
# print(dataframe)
#定义反归一化
def denormalize_predictions(predictions, min_labels, max_labels):
    denormalized_predictions = (predictions * (max_labels - min_labels)) + min_labels
    return denormalized_predictions


#图像预处理
transform = transforms.Compose([
    torchvision.transforms.Resize((224, 224)),  # 调整图像大小
    torchvision.transforms.ToTensor(),         # 将图像转换为张量
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# 实例化数据集对象
dataset = coast_Dataset(data_dir, dataframe=dataframe, transform=transform)

# 2. 切分train ＆ test
# 定义训练集和测试集的大小
train_size = int(0.9 * len(dataset))   #训练集测试集比例
test_size = len(dataset) - train_size

# 划分训练集和测试集
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])




"""
# 遍历数据集并打印图像及其对应的标签
for i in range(len(train_dataset)):
    image, label, img_name = train_dataset[i]
    print("Sample {}: Label - {}, name:{} ".format(i, label, img_name))
"""


# # 3. 定义数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)  #batch_size一次取样本数量
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=False)



# print(ResNet50)
mod = ResNet50()
mod = mod.to(device)
loss_fn = torch.nn.SmoothL1Loss()
loss_fn = loss_fn.to(device)
learning_rate = 0.001
optimizer = torch.optim.Adam(mod.parameters(),lr=learning_rate)
# scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=0.1)

Writer = SummaryWriter(r"E:\TWT\PY\2024--coast\logs-240403-1")

lowest_mae = float('10')
model_save_dir = r"E:\TWT\PY\2024--coast\checkpoint"
os.makedirs(model_save_dir, exist_ok=True)

epochs = 100
for i in range(epochs):
    print("-----Epoch:{}/{}-----".format(i+1, epochs))
    mod.train()
    for data in tqdm(train_loader):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = mod(images)

        # if outputs.isnan().any():       #检查输出是否有NAN
        #     print("Model outputs contain NaN values.")

        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # scheduler.step()


    if i % 1 == 0:
        print("Train_Loss: {:.4f}".format(loss.item())) #打印loss
        print("lr= {:.4f}".format(optimizer.param_groups[0]['lr'])) #打印学习率

        Writer.add_scalar("train_Loss", loss.item(), i)


    mod.eval()
    all_predictions = []
    all_true_values = []
    total_test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = mod(images)

            predictions = outputs.cpu().numpy()

            if np.isnan(predictions).any():
                predictions = np.nan_to_num(predictions)

            denormalized_predictions = (predictions * (max_labels - min_labels)) + min_labels
            denormalized_labels = (labels * (max_labels - min_labels)) + min_labels

            all_predictions.extend(denormalized_predictions)  # 将反归一化的预测值添加到列表中
            all_true_values.extend(denormalized_labels.cpu().numpy())  # 将反归一化的真实值添加到列表中

            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()

    test_step = i + 1
    num_outputs = 6
    mae_values = [mean_absolute_error(all_true_values[i * num_outputs:(i + 1) * num_outputs],
                                      all_predictions[i * num_outputs:(i + 1) * num_outputs]) for i in range(num_outputs)]
    rmse_values = [np.sqrt(mean_squared_error(all_true_values[i * num_outputs:(i + 1) * num_outputs],
                                              all_predictions[i * num_outputs:(i + 1) * num_outputs])) for i in range(num_outputs)]
    print("Test_Loss: {}".format(total_test_loss))

    for i, mae in enumerate(mae_values):
        print(f"MAE for Question{i + 1}: {mae}")
        Writer.add_scalar(f"test_MAE/Question_{i + 1}", mae, test_step)
    average_mae = sum(mae_values) / num_outputs
    print(f"Average MAE across all outputs: {average_mae}")
    Writer.add_scalar("test_MAE", average_mae, test_step)

    for i, rmse in enumerate(rmse_values):
        print(f"RMSE for Question {i + 1}: {rmse}")
        Writer.add_scalar(f"test_RMSE/Question_{i + 1}", rmse, test_step)
    average_rmse = sum(rmse_values) / num_outputs
    print(f"Average RMSE across all outputs: {average_rmse}")
    Writer.add_scalar("test_RMSE", average_rmse, test_step)


    mae = average_mae
    if average_mae < lowest_mae:
        lowest_mae = mae
        torch.save(mod.state_dict(), os.path.join(model_save_dir, f"best_model_epoch_{test_step}.pt"))
        print(f"Best model with MAE {lowest_mae} saved to {model_save_dir}")
torch.save(mod.state_dict(), os.path.join(model_save_dir, "muti_13888_last_model.pt"))
print(f"Last model saved to {model_save_dir}")

