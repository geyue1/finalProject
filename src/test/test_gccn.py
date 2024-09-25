# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test_gccn.py
@Author : yge
@Date : 2024/7/7 10:10
@Desc :

==============================================================
'''
from src.model.group_eq_cnn import GCNN_P4CNN_28

# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> exp_mnist.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/19 18:55
@Desc :

==============================================================
'''
import os.path

from torch import optim, nn
from torchvision import transforms

from src.util.utils_ import mnist_data, get_device

from src.util.train import train



data_path = os.path.join("..","data","mnist")
device = get_device()

#utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data,test_data = mnist_data(data_path=data_path,transform=transform,batch_size=512)
#net = MNIST_SES_V()
in_channels = 1
out_features = 10
net = GCNN_P4CNN_28(in_channels, out_features, device)
epoch_num = 5
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)

train(net,
      device,
      epoch_num,
      lr,optimizer,loss_fn,train_data,test_data)
