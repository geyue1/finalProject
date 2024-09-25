# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> RMSProp_28.py
@Author : yge
@Date : 2024/8/28 20:28
@Desc :

==============================================================
'''
import os

from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import src.parameter as p
from src.model.deep_scale_spaces import DSS_Scalar_28
from src.model.group_eq_cnn import GCNN_P4CNN_28
from src.model.resNet18 import ResNet18_28
from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_28
from src.model.scale_equivariant_steerable_networks import mnist_sesn_scalar_28
from src.model.scale_invariant_cnn import SiCNN_28
from src.util.train import train
from src.util.utils_ import mnist_data, get_device

if __name__ == '__main__':
    p.lr = 0.01
    p.epoch_num = 50
    p.batch_size = 256

    data_path = os.path.join("..","..", "data")
    device = get_device()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data, test_data = mnist_data(data_path=data_path, transform=transform, batch_size=p.batch_size)

    nets_28 = [ResNet18_28(),
               DSS_Scalar_28(p.in_channels_28, p.out_features_28, device=device),
               GCNN_P4CNN_28(p.in_channels_28, p.out_features_28, device=device),
               ScaleEqNet_28(p.in_channels_28, p.out_features_28),
               mnist_sesn_scalar_28(p.in_channels_28, p.out_features_28),
               SiCNN_28(p.out_features_28, kernels=[3, 5, 7], channels=[1, 16, 24, 36], device=device)]


    for net in nets_28:
        optimizer = optim.RMSprop(net.parameters(), lr=p.lr)
        loss_fn = nn.CrossEntropyLoss().to(device)
        result = train(net,
                       device,
                       p.epoch_num,
                       p.lr, optimizer, loss_fn, train_data, test_data,fix_lr=True)