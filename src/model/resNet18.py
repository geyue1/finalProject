# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> resNet18.py
@Author : yge
@Date : 2023/8/24 10:06
@Desc :

==============================================================
'''

import torch.nn as nn
import torchvision.models as models
from src.util.utils_ import get_num_parameters
class ResNet18_28(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)

        self.resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet18.maxpool = nn.Identity()
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        return self.resnet18(x)

class ResNet18_56(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet18 = models.resnet18(pretrained=False)

        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2, bias=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        return self.resnet18(x)

if __name__ == '__main__':
    net = ResNet18_28()
    print(net)
    print(get_num_parameters(net))

    net2= ResNet18_56()
    print(net2)
    print(get_num_parameters(net2))