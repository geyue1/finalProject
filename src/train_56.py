# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> train_56.py
@Author : yge
@Date : 2024/8/3 23:38
@Desc :

==============================================================
'''
import os
from torch import optim, nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from src.model.scale_invariant_cnn import SiCNN_56
from src.util.utils_ import *
from src.util.train import train

if __name__ == '__main__':
    data_path = os.path.join("..", "data","AAR")
    device = get_device()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_data, test_data = local_dataset(data_path=data_path, transform=transform, batch_size=p.batch_size)
    net = SiCNN_56(p.out_features_56, kernels=[3, 5, 7], channels=[3, 16, 24, 36], device=device)
    optimizer = optim.SGD(net.parameters(), lr=p.lr)
    loss_fn = nn.CrossEntropyLoss().to(device)

    train(net,
          device,
          100,
          p.lr, optimizer, loss_fn, train_data, test_data)

