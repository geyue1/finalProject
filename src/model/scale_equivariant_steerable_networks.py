# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> scale_equivariant_steerable_networks.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/24 19:51
@Desc :
https://github.com/isosnovik/sesn
==============================================================
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.impl.ses_conv
from src.util.utils_ import get_num_parameters


class SESN_Scalar_28(nn.Module):

    def __init__(self,in_channels,out_features, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            models.impl.ses_conv.SESConv_Z2_H(in_channels, C1, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            models.impl.ses_conv.SESConv_Z2_H(C1, C2, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            models.impl.ses_conv.SESConv_Z2_H(C2, C3, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(kwargs.get('dropout', 0.7)),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class SESN_Scalar_56(nn.Module):

    def __init__(self,in_channels,out_features, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            models.impl.ses_conv.SESConv_Z2_H(in_channels, C1, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C1),

            models.impl.ses_conv.SESConv_Z2_H(C1, C2, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(C2),

            models.impl.ses_conv.SESConv_Z2_H(C2, C3, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(1520, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(kwargs.get('dropout', 0.7)),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class MNIST_SES_V(nn.Module):

    def __init__(self, pool_size=4, kernel_size=11, scales=[1.0], basis_type='A', dropout=0.7, **kwargs):
        super().__init__()
        C1, C2, C3 = 32, 63, 95
        self.main = nn.Sequential(
            models.impl.ses_conv.SESConv_Z2_H(1, C1, kernel_size, 7, scales=scales,
                                              padding=kernel_size // 2, bias=True,
                                              basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C1),

            models.impl.ses_conv.SESConv_H_H(C1, C2, 1, kernel_size, 7, scales=scales,
                                             padding=kernel_size // 2, bias=True,
                                             basis_type=basis_type, **kwargs),
            nn.ReLU(True),
            nn.MaxPool3d([1, 2, 2], stride=[1, 2, 2]),
            nn.BatchNorm3d(C2),

            models.impl.ses_conv.SESConv_H_H(C2, C3, 1, kernel_size, 7, scales=scales,
                                             padding=kernel_size // 2, bias=True,
                                             basis_type=basis_type, **kwargs),
            models.impl.ses_conv.SESMaxProjection(),
            nn.ReLU(True),
            nn.MaxPool2d(pool_size, padding=2),
            nn.BatchNorm2d(C3),
        )

        self.linear = nn.Sequential(
            nn.Linear(4 * C3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


def mnist_ses_vector_28(**kwargs):
    num_scales = 3
    factor = 3.0
    min_scale = 1.5
    mult = 1.5
    size = 13
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = MNIST_SES_V(pool_size=4, kernel_size=size, scales=scales,
                        basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model
def mnist_sesn_scalar_28(in_channels,out_features,**kwargs):
    num_scales = 4
    factor = 3.0
    min_scale = 1.7
    mult = 1.5
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = SESN_Scalar_28(in_channels,out_features,pool_size=4, kernel_size=size, scales=scales,
                           basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return model

def mnist_ses_scalar_56(in_channels,out_features,**kwargs):
    num_scales = 3
    factor = 2.0
    min_scale = 2.0
    mult = 1.4
    size = 15
    dropout = 0.7
    q = factor ** (1 / (num_scales - 1))
    scales = [min_scale * q**i for i in range(num_scales)]
    scales = [round(s, 2) for s in scales]
    model = SESN_Scalar_56(in_channels,out_features,pool_size=8, kernel_size=size, scales=scales,
                             basis_type='B', mult=mult, max_order=4, dropout=dropout)
    return nn.Sequential(nn.Upsample(scale_factor=2), model)

if __name__ == '__main__':
    net = mnist_sesn_scalar_28(1,10)
    print(get_num_parameters(net))
    x = torch.randn(16,1,28,28)
    y = net(x)
    print(y.shape)

    net_2 = mnist_ses_scalar_56(3,2)
    print(get_num_parameters(net_2))
    x2 = torch.randn(8,3,56,56)
    y2 = net_2(x2)
    print(y2.shape)
