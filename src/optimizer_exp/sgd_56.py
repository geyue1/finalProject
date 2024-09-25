# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> sgd_56.py
@Author : yge
@Date : 2024/8/28 00:09
@Desc :

==============================================================
'''
import os

from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import src.parameter as p
from src.model.deep_scale_spaces import DSS_Scalar_56
from src.model.group_eq_cnn import GCNN_P4CNN_56
from src.model.resNet18 import ResNet18_56
from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_56
from src.model.scale_equivariant_steerable_networks import mnist_ses_scalar_56
from src.model.scale_invariant_cnn import SiCNN_56
from src.util.train import train
from src.util.utils_ import mnist_data, get_device, local_dataset

if __name__ == '__main__':
    p.lr = 0.01
    p.epoch_num = 50
    p.batch_size = 256

    data_path = os.path.join("..","..", "data","AAR")
    device = get_device()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data, test_data = local_dataset(data_path=data_path, transform=transform, batch_size=p.batch_size)

    nets_56 = [ResNet18_56(),
               DSS_Scalar_56(p.in_channels_56, p.out_features_56, device=device),
               GCNN_P4CNN_56(p.in_channels_56, p.out_features_56, device=device),
               ScaleEqNet_56(p.in_channels_56, p.out_features_56),
               mnist_ses_scalar_56(p.in_channels_56, p.out_features_56),
               SiCNN_56(p.in_channels_56, kernels=[3, 5, 7], channels=[3, 16, 24, 36], device=device)]



    for net in nets_56:
        optimizer = optim.SGD(net.parameters(), lr=p.lr)
        loss_fn = nn.CrossEntropyLoss().to(device)
        result = train(net,
                       device,
                       p.epoch_num,
                       p.lr, optimizer, loss_fn, train_data, test_data,fix_lr=True)