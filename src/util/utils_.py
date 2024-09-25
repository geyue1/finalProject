# -* UTF-8 *-
'''
==============================================================
@Project -> File : eccn -> utils_.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 21:01
@Desc :

==============================================================
'''
import os
import logging

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import src.parameter as p


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    if torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

def mnist_data(data_path,transform,batch_size):

    train_data = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=True, download=True,transform=transform,),
        batch_size=batch_size, shuffle=True
    )

    test_data = torch.utils.data.DataLoader(
        dataset=datasets.MNIST(data_path, train=False, download=True,transform=transform,),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    return train_data,test_data

def local_dataset(data_path,transform,batch_size):
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader,test_loader

def log(name="root",level=logging.DEBUG,log_file=None):
    if log_file is None:
        log_file = os.path.join("..", "logs")
        if not os.path.isdir(log_file):
            os.makedirs(log_file)
    # logging.basicConfig(filename=os.path.join("..", "logs", "train-20240829.log"), level=level,
    #                     format=format'%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # 创建一个logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler(os.path.join("..", "logs", "train-20240829.log"))
    file_handler.setLevel(level)

    # 再创建一个handler，用于输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def get_num_parameters(module):
    params = [p.nelement() for p in module.parameters() if p.requires_grad]
    num = sum(params)
    return num



if __name__ == '__main__':
    transform_ = transforms.Compose([
        transforms.ToTensor()
    ])
    path = os.path.join("..","data","mnist")
    print(path)
    mnist_data(path,transform_,128)
