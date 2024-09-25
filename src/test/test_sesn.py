# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test_sesn.py
@Author : yge
@Date : 2024/7/17 23:07
@Desc :

==============================================================
'''
import os

import torch
from torch import optim, nn

from src.model.scale_equivariant_steerable_networks import mnist_sesn_scalar_28, mnist_ses_vector_28
from src.util.train import train
from src.util.utils_ import mnist_data, get_device
from torchvision import transforms
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

data_path = os.path.join("..","data","mnist")
device = get_device()
device = torch.device("cpu")

#utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data,test_data = mnist_data(data_path=data_path,transform=transform,batch_size=128)

epoch_num = 5
lr = 0.005

#net = mnist_ses_scalar_28()
net = mnist_ses_vector_28()

optimizer = optim.SGD(net.parameters(), lr=lr)
# optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss().to(device)

train(net,
      device,
      epoch_num,
      lr, optimizer, loss_fn, train_data, test_data)
