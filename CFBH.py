import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torch.backends.cudnn as cudnn
from util.read_data import read_dataset
import random
from model.resnet18_CFBH import MainNet
from train_model import train
import argparse

def set_seed(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def running(config):
    print(config)
    set_seed(111)
    classes,train_dataloader, test_dataloader, base_dataloader=read_dataset(config.dataset,config.batch_size)
    model =MainNet(num_classes=classes, num_parts=config.num_parts,bit=config.bit_length,ratio=config.ratio)
    model.cuda()
    criterion = nn.CrossEntropyLoss()
    criterion_qua = nn.MSELoss()
    optimizer = optim.SGD(model.parameters() ,lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    save_dir = './training_checkpoint/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train(config,model, train_dataloader, test_dataloader, base_dataloader, criterion, criterion_qua, optimizer, lr_scheduler, save_dir, config.epoch, classes)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CFBH')
    parser.add_argument('--info', default='CFBH')
    parser.add_argument('--dataset', type=str, default='cub_bird')
    parser.add_argument('--ratio', type=int, default=0.25)
    parser.add_argument('--num_parts', type=int, default=64,help='cub_bird 64, others 128')
    parser.add_argument('--batch_size', type=int, default=128,help='food veg 256,others 128' )
    parser.add_argument('--bit_list', type=str, default='16,32,48,64')
    parser.add_argument('--epoch', type=int, default=90)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.02) 
    config= parser.parse_args()
    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config.bit_list = list(map(int, config.bit_list.split(',')))
    for bit in config.bit_list:
        config.bit_length=bit
        running(config)


