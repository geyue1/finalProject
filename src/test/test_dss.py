# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test_dss.py
@Author : yge
@Date : 2024/7/16 23:19
@Desc :

==============================================================
'''

import os

from torch import optim, nn

from src.model.deep_scale_spaces import MNIST_DSS_Vector, DSS_Scalar_28
from src.model.scale_eq_CNN_with_vector_fields import ScaleEqNet_28
from src.util.train import train
from src.util.utils_ import mnist_data, get_device
from torchvision import transforms
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

data_path = os.path.join("..","data","mnist")
device = get_device()

#utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data,test_data = mnist_data(data_path=data_path,transform=transform,batch_size=512)

in_channels = 3
out_fetures = 10

nets = [DSS_Scalar_28(in_channels,out_fetures,device=device)]
epoch_num = 100
lr = 0.005

for i in range(len(nets)):
    net = nets[i]
    optimizer = optim.SGD(net.parameters(), lr=lr)
    #optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss().to(device)

    train(net,
          device,
          epoch_num,
          lr,optimizer,loss_fn,train_data,test_data)

