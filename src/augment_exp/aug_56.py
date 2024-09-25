# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> aug_56.py
@Author : yge
@Date : 2024/7/31 11:24
@Desc :

==============================================================
'''
import logging
import os

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, datasets

import src.parameter as p
from src.model.deep_scale_spaces import DSS_Scalar_56
from src.model.group_eq_cnn import GCNN_P4CNN_56
from src.model.resNet18 import ResNet18_56
from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_56
from src.model.scale_equivariant_steerable_networks import mnist_ses_scalar_56
from src.model.scale_invariant_cnn import SiCNN_56
from src.util.train import train, evaluate
from src.util.utils_ import mnist_data, get_device, local_dataset, log

logger = log("aug_56", level=logging.INFO)

if __name__ == '__main__':
    p.epoch_num = 10

    data_path = os.path.join("..","..", "data","AAR_augment")
    device = get_device()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)


    nets_56 = [ResNet18_56(),
               DSS_Scalar_56(p.in_channels_56, p.out_features_56, device=device),
               GCNN_P4CNN_56(p.in_channels_56, p.out_features_56, device=device),
               ScaleEqNet_56(p.in_channels_56, p.out_features_56),
               mnist_ses_scalar_56(p.in_channels_56, p.out_features_56),
               SiCNN_56(p.in_channels_56, kernels=[3, 5, 7], channels=[3, 16, 24, 36], device=device)]



    for net in nets_56:
        net_name = type(net).__name__
        model_params = torch.load(os.path.join("..", "..", "saved_models", f"{net_name}.pth"))
        net.load_state_dict(model_params)
        optimizer = optim.SGD(net.parameters(), lr=p.lr)
        loss_fn = nn.CrossEntropyLoss().to(device)
        acc = []
        for epoch in range(p.epoch_num):
            data_loader = DataLoader(dataset, batch_size=p.batch_size, shuffle=True)
            epoch_acc = evaluate(net,
                                 device,
                                 epoch,
                                 loss_fn,
                                 data_loader)

            acc.append(epoch_acc)
        logger.info(f"{net_name} evaluation mean accuracy :{sum(acc) / len(acc)}")