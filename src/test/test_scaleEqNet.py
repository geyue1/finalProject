# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test_scaleEqNet.py
@Author : yge
@Date : 2024/7/13 18:26
@Desc :

==============================================================
'''
import os

from torch import optim, nn

from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_28
from src.util.train import train
from src.util.utils_ import mnist_data, get_device
from torchvision import transforms


data_path = os.path.join("..","data","mnist")
device = get_device()

#utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data,test_data = mnist_data(data_path=data_path,transform=transform,batch_size=512)

num_classes = 10
in_channels = 1

net = ScaleEqNet_28(in_channels, num_classes)
epoch_num = 100
lr = 0.005
optimizer = optim.SGD(net.parameters(), lr=lr)
#optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss().to(device)

train(net,
      device,
      epoch_num,
      lr,optimizer,loss_fn,train_data,test_data)

