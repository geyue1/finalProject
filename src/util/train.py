# -* UTF-8 *-
'''
==============================================================
@Project -> File : finalProject -> train.py
@Author : Yue Ge
@Email : psxyg15@nottingham.ac.uk
@Date : 2023/10/18 19:11
@Desc :

==============================================================
'''
import logging
import os.path

import torch
import src.parameter as p
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from src.util.utils_  import *

# log_file = os.path.join("..","logs")
# if not os.path.isdir(log_file):
#     os.makedirs(log_file)
# logging.basicConfig(filename=os.path.join("..","logs","train-20240829.log"), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = log("train", level=logging.INFO)
best_acc = 0
writer = SummaryWriter(p.tensorboard_path)
def train(net,device,epoch_num,lr,optimizer,loss_fn,train_data,test_data,fix_lr=False):
    net_name = type(net).__name__
    result = {}
    net.to(device)
    net.train()

    for epoch in range(epoch_num):
        train_loss = 0
        correct = 0
        total = 0
        with tqdm(train_data,desc=f"{net_name} train epoch {epoch+1}",unit="batch") as pbar:
            for batch_id, (inputs, targets) in enumerate(train_data):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(loss=train_loss / total, accuracy=correct / total)
                pbar.update()
            pbar.close()
        epoch_loss = train_loss / total
        epoch_acc = correct / total
        writer.add_scalars(f'{net_name} training', {"loss":epoch_loss,"acc":epoch_acc}, epoch+1)
        logger.info(f"Train Epoch:{epoch + 1} Losss:{100. * epoch_loss:.2f}% Acc:{100. * epoch_acc :.2f}%")
        #logger.info(f"train_loss={train_loss}")
       # logger.info(f"train total={total}")
        #logger.info(f"train size={len(train_data)}")
        test_loss,test_acc =  test(net,device,epoch,loss_fn,test_data)
        if not fix_lr:
           adjust_learning_rate(optimizer,lr,epoch)
        result[epoch+1] = [epoch_loss,epoch_acc,test_loss,test_acc]
    return result

def test(net,device,epoch,loss_fn,test_data):
    net_name = type(net).__name__
    test_loss = 0
    correct = 0
    total = 0
    global best_acc
    net.to(device)
    net.eval()

    with torch.no_grad():
        with tqdm(test_data,desc=f"{net_name} test epoch {epoch+1}",unit="batch") as pbar:
            for inputs, targets in test_data:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = loss_fn(outputs, targets)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(loss=test_loss / total, accuracy=correct / total)
                pbar.update()
            pbar.close()
        epoch_loss = test_loss / total
        epoch_acc = correct / total
        writer.add_scalars(f'{net_name} test', {"loss":epoch_loss,"acc":epoch_acc}, epoch + 1)
        logger.info(f"Test Epoch:{epoch + 1} Losss:{100. * epoch_loss:.2f}% Acc:{100. * epoch_acc :.2f}%")
        temp = correct / total
        if p.SAVE_MODEL and temp>best_acc:
            best_acc = temp
            logger.info(f"******best_acc={best_acc}")
            path = os.path.join("..","saved_models")
            if not os.path.isdir(path):
                os.makedirs(path)
            torch.save(net.state_dict(), os.path.join("..","..","saved_models",f"{net_name}.pth"))
        return epoch_loss,epoch_acc

def evaluate(net,device,epoch,loss_fn,data):
    net_name = type(net).__name__
    net.to(device)
    net.eval()

    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        with tqdm(data,desc=f"{net_name} evaluate epoch {epoch+1}",unit="batch") as pbar:
            for inputs, targets in data:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                ll = loss_fn(outputs, targets)

                loss += ll.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix(loss=loss / total, accuracy=correct / total)
                pbar.update()
            pbar.close()
        epoch_loss = loss / total
        epoch_acc = correct / total
        writer.add_scalars(f'{net_name} evaluate', {"loss":epoch_loss,"acc":epoch_acc}, epoch+1)
        logger.info(f"evaluate Epoch:{epoch + 1} Losss:{100. * epoch_loss:.2f}% Acc:{100. * epoch_acc :.2f}%")
        return epoch_acc

def adjust_learning_rate(optimizer,start_lr, epoch):
        """Gradually decay learning rate"""
        if epoch == 20:
            lr = start_lr / 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 30:
            lr = start_lr / 20
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch == 40:
            lr = start_lr / 40
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


writer.close()