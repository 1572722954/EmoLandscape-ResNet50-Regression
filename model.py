import torch
from torch import nn
import torchvision.models as models


"""  最后一个激活函数： RELU
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # 加载RESNET50模型，不使用预训练参数
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc.add_module('RELU', nn.ReLU())
        self.resnet50.fc.add_module('dropout', nn.Dropout(0.3))
        self.resnet50.fc.add_module('last_linear', nn.Linear(1000,6))
        self.resnet50.fc.add_module('RELU2', nn.ReLU())

    def forward(self, x):
        x = self.resnet50(x)
        x = self.RELU(x)
        x = torch.flatten(x, 1)  # 将输出展平为一维张量
        x = self.resnet50.fc.dropout(x)
        x = self.resnet50.fc.last_linear(x)
        x = self.RELU2(x)
        return x

"""

""" 多输出回归版 ()
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # 加载RESNET50模型，不使用预训练参数
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 1000)

        # 定义新的层和激活函数，并添加到类中
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.last_linear = nn.Linear(1000, 6)


    def forward(self, x):

        x = self.resnet50(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)

        return x
"""

"""不过多修改版"""
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.resnet50(x)
        return x

# mod = ResNet50()
# print(mod)