import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torchvision.datasets as dsets
from torchvision import transforms
from torch.autograd import Variable
import torchvision
import math
import numpy as np
from util.cal_map import calculate_map, compress
from util.calc_hr_new import calc_map_all, shot_eval
import pandas as pd
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
import scipy.io


from util.utils import AverageMeter

mse = nn.MSELoss()

def calc_sim(label1, label2):
    S = (label1.mm(label2.t()) > 0).type(torch.float32)
    '''
    soft constraint
    '''
    S = (S - 0.5) * 2
    return S

def pairwise_loss(code,S,code_length):
    inner_dot = code.mm(code.T) / code_length
    loss = mse(inner_dot, S)
    return loss

def quan_loss(code):
    one_mat = torch.ones_like(code).cuda()
    loss = mse(code.abs(),one_mat)
    return loss


def load_modal(model,model_dir):
    if not model_dir.endswith('.pth'):
        model_dir = os.path.join(model_dir, 'last.pth')
    print('Loading model from %s' % (model_dir))
    weights = torch.load(model_dir) 

    model.load_state_dict(weights)        

    return model    



def encoding_onehot(target, nclasses=10):
    target_onehot = torch.FloatTensor(target.size(0), nclasses).cuda()
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot



def predict_hash_code(model, data_loader,hash_bit,classes):       

    is_start = True
    total_codes = torch.empty((0,hash_bit)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()
    model.eval()

    with torch.no_grad():
        for batch_cnt_val, data_val in enumerate(data_loader):
            inputs = data_val[0].cuda()
            labels = torch.from_numpy(np.array(data_val[1])).cuda()

            outputs_codes= model(inputs)

            if batch_cnt_val == 0:
                ground = labels
                pred_out = outputs_codes
            else:
                ground = torch.cat((ground,labels))
                pred_out = torch.cat((pred_out,outputs_codes))

            total_codes = torch.cat((total_codes,outputs_codes.sign()))
            total_labels = torch.cat((total_labels,labels))
    return total_codes, total_labels


def main(config,matdir,train_dataloader,test_dataloader,base_dataloader,model,classes,max_map,epoch):
    
    print(config)
    os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_loader = train_dataloader
    test_loader = test_dataloader
    database_loader = base_dataloader
    print('Testing')
    qB,qL = predict_hash_code(model, test_loader,config.bit_length,classes)
    rB, rL = predict_hash_code(model, database_loader,config.bit_length,classes)
    qL_binary = encoding_onehot(qL,classes)
    rL_binary = encoding_onehot(rL,classes)
    print('save hash code')

    map_all, p100_all, pr2_all = calc_map_all(qB=qB,rB=rB,queryL=qL_binary,retrievalL=rL_binary,device=qB.device,knn=100)
    MAP = map_all.mean()
    p100 = p100_all.mean()
    pr2 = pr2_all.mean()
    
    eval_dict = [{'epoch': epoch,'map':MAP,'p100':p100,'pr2':pr2,}] 
    print('MAP:%.4f, p100:%.4f, pr2:%.4f'%(MAP,p100,pr2))   
    if (MAP>max_map):
        scipy.io.savemat('%s/%sbits.mat'%(matdir,str(config.bit_length)),
                    mdict={'db_binary':rB.detach().cpu().numpy(),'tst_binary':qB.detach().cpu().numpy(),
                    'db_label':rL.detach().cpu().numpy(),'tst_label':qL.detach().cpu().numpy()})
        df =pd.DataFrame(eval_dict,columns=['epoch','map','p100','pr2',])
        df.to_csv('%s/%sbits_final.csv'%(matdir,str(config.bit_length)),index=False)
    return MAP

    

   
def eval(config,train_dataloader,test_dataloader,base_dataloader,model,classes,max_map,epoch):

    # logdir = 'logs/' + config.dataset + '/' + str(config.bit_length) + 'bits'
   
    # if not os.path.isdir(logdir):
    #     os.makedirs(logdir)

    matdir = './CFBH'+ '/' +config.dataset+'/' + str(config.bit_length) + 'bits'
    if not os.path.isdir(matdir):
        os.makedirs(matdir)
        
    MAP=main(config,matdir,train_dataloader,test_dataloader,base_dataloader,model,classes,max_map,epoch)
    return MAP