import torch
import numpy as np
import os

user=os.getenv('SLURM_JOB_USER')
job=os.getenv('SLURM_JOB_ID')
#tmp='/ssd/'+user+'/'+job+'/fine_grained_dataset'
#tmp='/home/yajie/文档/fine_grained_dataset'
tmp='/home/data_10501005'

def one_hot_label(label,num_class):
    label=label.view(-1,1).long()
    onehot=torch.zeros(label.shape[0],num_class)
    onehot=onehot.scatter(1,label,1)
    return onehot


