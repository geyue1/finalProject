# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> aug_28.py
@Author : yge
@Date : 2024/7/30 21:31
@Desc :

==============================================================
'''
import logging
import os
import torch
from torch import optim, nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms,datasets

import src.parameter as p
from src.model.deep_scale_spaces import DSS_Scalar_28
from src.model.group_eq_cnn import GCNN_P4CNN_28
from src.model.resNet18 import ResNet18_28
from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_28
from src.model.scale_equivariant_steerable_networks import mnist_sesn_scalar_28
from src.model.scale_invariant_cnn import SiCNN_28
from src.util.train import train, test, evaluate
from src.util.utils_ import mnist_data, get_device, local_dataset, log

logger = log("aug_28", level=logging.INFO)

if __name__ == '__main__':
    p.epoch_num = 10

    data_path = os.path.join("..", "..", "data", "MNIST_augment")
    device = get_device()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)



    nets_28 = [ResNet18_28(),
               DSS_Scalar_28(p.in_channels_28, p.out_features_28, device=device),
               GCNN_P4CNN_28(p.in_channels_28, p.out_features_28, device=device),
               ScaleEqNet_28(p.in_channels_28, p.out_features_28),
               mnist_sesn_scalar_28(p.in_channels_28, p.out_features_28),
               SiCNN_28(p.out_features_28, kernels=[3, 5, 7], channels=[1, 16, 24, 36], device=device)]


    for net in nets_28:
        net_name = type(net).__name__
        model_params = torch.load(os.path.join("..","..","saved_models",f"{net_name}.pth"))
        net.load_state_dict(model_params)
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