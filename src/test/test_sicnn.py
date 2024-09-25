# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> test_sicnn.py
@Author : yge
@Date : 2024/7/10 22:22
@Desc :

==============================================================
'''

import os.path

from torch import optim, nn
from torchvision import transforms

from src.model.scale_invariant_cnn import SiCNN_28
from src.util.utils_ import mnist_data, get_device

from src.util.train import train

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

data_path = os.path.join("..","data","mnist")
device = get_device()

#utils.log("exp_mnist").debug(device)
transform = transforms.Compose([
    transforms.ToTensor()
])
train_data,test_data = mnist_data(data_path=data_path,transform=transform,batch_size=512)
#net = MNIST_SES_V()


num_classes = 10
kernels = [3, 5, 7]
channels = [1, 16, 24, 36]

net = SiCNN_28(num_classes, kernels, channels, device)
epoch_num = 100
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss().to(device)

train(net,
      device,
      epoch_num,
      lr,optimizer,loss_fn,train_data,test_data)
