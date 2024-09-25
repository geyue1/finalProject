# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> scale_invariant_cnn.py
@Author : yge
@Date : 2023/9/9 21:39
@Desc :

==============================================================
'''
import os
import src.parameter as p
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from src.util.utils_ import get_num_parameters

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class SiCNNConv2D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,canonical_conv=None,device=None):
        super().__init__()
        self.device = device;
        self.canonical_conv = False if canonical_conv is None else True
        if self.canonical_conv is False:
           self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        else:
            self.conv = canonical_conv
            '''
            original paper:
            In our implementation, we use bicubic interpolation as the scaling method to transform filters
            '''
            #The operator 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device
            self.upsample = nn.Upsample(size=(kernel_size, kernel_size), mode='bilinear', align_corners=True)
    def forward(self,x):
        if self.canonical_conv is True:
           weight = self.upsample(self.conv.weight)
           weight = weight / weight.sum(dim=[2, 3], keepdim=True)  # Normalize weights
           y = F.conv2d(x, weight,
                        bias=self.conv.bias,
                        stride=1,
                        padding=0)
        else:
            y = self.conv(x)
        return y

class SiCNNColumn(nn.Module):
    def __init__(self,channels,kernel_size,stride,padding,canonical_convs,canonical=False,device=None):
        super().__init__()
        if canonical is not True:
            self.conv1 = SiCNNConv2D(channels[0],channels[1],kernel_size,stride,padding,canonical_conv=canonical_convs[0],device=device)
            self.conv2 = SiCNNConv2D(channels[1],channels[2],kernel_size,stride,padding,canonical_conv=canonical_convs[1],device=device)
            self.conv3 = SiCNNConv2D(channels[2],channels[3],kernel_size,stride,padding,canonical_conv=canonical_convs[2],device=device)
        else:
            self.conv1 = canonical_convs[0]
            self.conv2 = canonical_convs[1]
            self.conv3 = canonical_convs[2]

        self.main = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(3,stride=1),
            nn.BatchNorm2d(channels[1],device=device),

            self.conv2,
            nn.ReLU(),
            nn.AvgPool2d(3,stride=1),
            nn.BatchNorm2d(channels[2],device=device),

            self.conv3,
            nn.ReLU(),
            nn.AvgPool2d(3, stride=1),
            nn.BatchNorm2d(channels[3],device=device),
        )

    def forward(self,x):
        return self.main(x)

class SiCNN_28(nn.Module):

    def __init__(self,out_features,kernels,channels,device=None):
        super().__init__()
        self.num_classes = out_features
        stride = 1
        padding = 0
        '''
        the first kernel as canonical filter
        '''
        canonical_convs = [nn.Conv2d(channels[i],channels[i+1],kernels[0],stride,padding,device=device) for i in range(3)]

        self.columns = [SiCNNColumn(channels,kernel_size,stride,padding,canonical_convs,device=device) for kernel_size in kernels]
        self.columns[0] = SiCNNColumn(channels,kernels[0],stride,padding,canonical_convs,canonical=True,device=device)


        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(4*4 * channels[3]*len(kernels), 256, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, self.num_classes)
        )

    def forward(self,x):
        outputs = []
        for column in self.columns:
            output = column(x)
            if not output.size()[-1]==4:
                output = F.interpolate(output,size=4,mode='bilinear', align_corners=True)
            outputs.append(output)
        y = torch.cat(outputs,dim=1)
        return self.linear(y)
    @property
    def net_name(self):
        return "SiCNN_28"


class SiCNN_56(nn.Module):

    def __init__(self,out_features,kernels,channels,device=None):
        super().__init__()
        self.num_classes = out_features
        stride = 1
        padding = 0
        '''
        the first kernel as canonical filter
        '''
        canonical_convs = [nn.Conv2d(channels[i],channels[i+1],kernels[0],stride,padding,device=device) for i in range(3)]

        self.columns = [SiCNNColumn(channels,kernel_size,stride,padding,canonical_convs,device=device) for kernel_size in kernels]
        self.columns[0] = SiCNNColumn(channels,kernels[0],stride,padding,canonical_convs,canonical=True,device=device)


        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Flatten(),
            nn.Linear(4*4 * channels[3]*len(kernels), 256, bias=False),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, self.num_classes)
        )

    def forward(self,x):
        outputs = []
        for column in self.columns:
            output = column(x)
            if not output.size()[-1]==4:
                output = F.interpolate(output,size=4,mode='bilinear', align_corners=True)
            outputs.append(output)
        y = torch.cat(outputs,dim=1)
        return self.linear(y)


if __name__ == '__main__':
    out_faatures = 10
    kernels = [3,5,7]
    channels = [1,16,24,36]
    net = SiCNN_28(out_faatures, kernels, channels)


    print(get_num_parameters(net))

    x = torch.randn(8,1,28,28)

    net(x)

    net2 = SiCNN_56(out_features=2,kernels=kernels,channels=[3,16,24,36])
    x2 = torch.randn(8,3,56,56)
    y2 = net2(x2)
    print(get_num_parameters(net2))
    print(f"y2:{y2.shape}")






