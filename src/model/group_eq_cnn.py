# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> group_eq_cnn.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 16:49
@Desc :
https://github.com/tscohen/gconv_experiments
==============================================================
'''
import torch
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling
from torch import nn
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4ConvZ2, P4ConvP4
from src.util.utils_ import get_num_parameters


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return (P4ConvZ2(in_channels, out_channels, kernel_size, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),nn.ReLU())


class BN_Block(nn.Module):
    def __init__(self,conv,bn=True,act=nn.ReLU(),device=None):
        super(BN_Block,self).__init__()
        self.conv = conv
        if bn:
            out_channels = self.conv.out_channels
            self.bn = [nn.BatchNorm2d(out_channels),nn.BatchNorm3d(out_channels,device=device)]
        else:
            self.bn = None
        self.act = act
    def forward(self,x):
        y = self.conv(x)
        if self.bn and y.dim()==4:
            y = self.bn[0](y)
        elif self.bn and y.dim()==5:
            y = self.bn[1](y)
        if self.act:
            y = self.act(y)
        return y
'''
refer https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/MNIST_ROT/models/Z2CNN.py

CNN with 7 layers of 3 × 3 convolutions (4 × 4 in the final layer), 20 channels
in each layer, relu activation functions, batch normalization, dropout, and max-pooling after layer 2
'''
class Z2CNN(nn.Module):
    def __init__(self,in_channels,out_features):
        super(Z2CNN, self).__init__()
        kernel_size = 3
        bn = True
        act = nn.ReLU()
        self.dr = 0.3
        self.layer_1 = BN_Block(
            conv = nn.Conv2d(in_channels=in_channels,out_channels=20,kernel_size=kernel_size,stride=1,padding=0),
            bn = bn,
            act = act
        )
        self.layer_2 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_3 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_4 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_5 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_6 = BN_Block(
            conv=nn.Conv2d(in_channels=20, out_channels=20, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act
        )
        self.layer_7 = nn.Conv2d(in_channels=20,out_channels=20,kernel_size=4,stride=1,padding=0)
        self.fl = nn.Flatten()
        self.fc = nn.Linear(1*20, out_features)

    def forward(self,x):
        y = self.layer_1(x)
        y = F.dropout(y,self.dr,training=True)

        y = self.layer_2(y)
        y = F.max_pool2d(y,kernel_size=2,stride=2,padding=0)

        y = self.layer_3(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_4(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_5(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_6(y)
        y = F.dropout(y, self.dr, training=True)

        y = self.layer_7(y)
        y = self.fl(y)
        y = self.fc(y)

        return y
    def get_name(self):
        return Z2CNN.__name__


'''
refer https://github.com/tscohen/gconv_experiments/blob/master/gconv_experiments/MNIST_ROT/models/P4CNN.py
G-equivariant CNN with group = p4 (The 4 90-degree rotations)

P4ConvP4 with 7 layers of 3 × 3 convolutions (4 × 4 in the final layer), 10 channels
in each layer, relu activation functions, batch normalization, dropout, 
and plane_group_spatial_max_pooling after layer 2
'''
class GCNN_P4CNN_28(nn.Module):
    def __init__(self,in_channels,out_features,device=None):
        super(GCNN_P4CNN_28, self).__init__()
        kernel_size = 3
        bn = True
        act = nn.ReLU()
        if device is None:
            device = torch.device("cpu")
        self.layer_1 = BN_Block(
            conv=P4ConvZ2(in_channels=in_channels, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_2 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_3 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_4 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_5 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_6 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )

        self.layer_7 = P4ConvP4(in_channels=10, out_channels=10, kernel_size=4, stride=1, padding=0)
        self.fl = nn.Flatten()
        self.fc = nn.Linear(1*1*10*4, out_features)


    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        y = plane_group_spatial_max_pooling(y, ksize=2, stride=2, pad=0)
        y = self.layer_3(y)
        y = self.layer_4(y)
        y = self.layer_5(y)
        y = self.layer_6(y)
        y = self.layer_7(y)
        y = self.fl(y)
        y = self.fc(y)
        return y

class GCNN_P4CNN_56(nn.Module):
    def __init__(self,in_channels,out_features,device=None):
        super(GCNN_P4CNN_56, self).__init__()
        kernel_size = 3
        bn = True
        act = nn.ReLU()
        if device is None:
            device = torch.device("cpu")
        self.layer_1 = BN_Block(
            conv=P4ConvZ2(in_channels=in_channels, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_2 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_3 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_4 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_5 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )
        self.layer_6 = BN_Block(
            conv=P4ConvP4(in_channels=10, out_channels=10, kernel_size=kernel_size, stride=1, padding=0),
            bn=bn,
            act=act,
            device=device
        )

        self.layer_7 = P4ConvP4(in_channels=10, out_channels=10, kernel_size=4, stride=1, padding=0)

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9000, 256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, out_features)
        )


    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        y = plane_group_spatial_max_pooling(y, ksize=2, stride=2, pad=0)
        y = self.layer_3(y)
        y = self.layer_4(y)
        y = self.layer_5(y)
        y = self.layer_6(y)
        y = self.layer_7(y)
        y = self.linear(y)
        return y


if __name__ == '__main__':
    in_channels = 3
    out_features = 2

    z2net = Z2CNN(1,10)
    p4net = GCNN_P4CNN_28(1, 10)

    print(get_num_parameters(p4net))
    p4net_56 = GCNN_P4CNN_56(3, 2)
    print(get_num_parameters(p4net_56))
    # x = torch.randn(32,1,28,28)
    # print(x.shape[0],1,x.shape[-2],x.shape[-1])
    # y1 = p4net(x)
    # print(y1.shape)





