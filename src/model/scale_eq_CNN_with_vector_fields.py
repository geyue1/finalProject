# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> scale_eq_CNN_with_vector_fields.py
@Author : yge
@Date : 2024/7/13 17:31
@Desc :
refer : https://github.com/dmarcosg/ScaleEqNet
paper : https://arxiv.org/abs/1807.11783
==============================================================
'''

import torch
import torch.nn as nn
from ScaleEqNet import ScaleConv, VectorMaxPool, VectorBatchNorm, Vector2Magnitude, Vector2Angle
from src.util.utils_ import get_num_parameters

class ScaleEqNet_28(nn.Module):
    def __init__(self,in_channels,out_features):
        super().__init__()
        n_scales_small = 2
        n_scales_big = 2
        self.layer1 = nn.Sequential(
            ScaleConv(in_channels, 12, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big,mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(12)
        )
        self.layer2 = nn.Sequential(
            ScaleConv(12, 32, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(32)
        )
        self.layer3 = nn.Sequential(
            ScaleConv(32, 48, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big, mode=2),
            VectorMaxPool(4)
        )
        self.vector2max = nn.Sequential(
            Vector2Magnitude(),
            nn.Conv2d(48, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256, out_features, 1)
        )
        self.vector2argmax = nn.Sequential(
            Vector2Angle(),
            nn.Conv2d(48, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(48, 1, 1)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_max = self.vector2max(x)
        x_argmax = self.vector2argmax(x)

        x_max = x_max.view(x_max.size()[0],x_max.size()[1])
        x_argmax = x_argmax.view(x_argmax.size()[0], x_argmax.size()[1])
        return  x_max

class ScaleEqNet_56(nn.Module):
    def __init__(self,in_channels,out_features):
        super().__init__()
        n_scales_small = 2
        n_scales_big = 2
        self.layer1 = nn.Sequential(
            ScaleConv(in_channels, 12, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big,mode=1),
            VectorMaxPool(2),
            VectorBatchNorm(12)
        )
        self.layer2 = nn.Sequential(
            ScaleConv(12, 32, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big, mode=2),
            VectorMaxPool(2),
            VectorBatchNorm(32)
        )
        self.layer3 = nn.Sequential(
            ScaleConv(32, 48, [7, 7], 1, padding=3,
                      n_scales_small=n_scales_small,n_scales_big=n_scales_big, mode=2),
            VectorMaxPool(4)
        )
        self.vector2max = nn.Sequential(
            Vector2Magnitude(),
            nn.Conv2d(48, 256, 3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(256, out_features, 1)
        )
        self.vector2argmax = nn.Sequential(
            Vector2Angle(),
            nn.Conv2d(48, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Conv2d(48, 1, 1)
        )
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x_max = self.vector2max(x)
        x_argmax = self.vector2argmax(x)

        x_max = x_max.view(x_max.size()[0],-1)
        x_argmax = x_argmax.view(x_argmax.size()[0], -1)
        return  x_max

if __name__ == '__main__':
    net = ScaleEqNet_28(1, 10)
    print(type(net).__name__)
    print(get_num_parameters(net))
    x = torch.randn(8,1,28,28)
    y = net(x)
    print(y.shape)

    x2 = torch.randn(8,3,56,56)
    net56 = ScaleEqNet_56(3,2)
    print(get_num_parameters(net56))
    y2 = net56(x2)
    print(y2.shape)


