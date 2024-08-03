import torch
from torch import nn
import torch.nn.functional as F
from model import resnet
import numpy as np
import os
import math
pretrain_path='./pretrained/resnet18-f37072fd.pth'
class MainNet(nn.Module):
    def __init__(self, num_classes, num_parts,bit,ratio,channels=512):
        super(MainNet, self).__init__()
        self.num_classes = num_classes
        self.num_parts=num_parts
        self.channels=channels
        self.ratio=ratio
        self.pretrained_model = resnet.resnet18(pretrained=True, pth_path=pretrain_path)
        self.rawcls_net = nn.Linear(channels*2, num_classes)
        self.w = torch.randn(self.num_parts,channels).clamp_(min=0).cuda()
        self.w_ep_sum = self.w.data.clone().cuda()
        self.w_ep_num= torch.ones(self.num_parts).cuda()
        self.relu=nn.ReLU(inplace=True)
        self.hash_layer=nn.Sequential(
            nn.Linear(channels,bit),
        )
        self.hash_cls = nn.Linear(bit, num_classes)
    def BNI(self,x):
        batch_size = x.size(0)
        bit = x.size(1)
        l=int(bit*self.ratio)
        num_b=batch_size*l
        x=x.cuda()/(1-self.ratio)
        if num_b%2!=0:
            l=l-1
            num_b=batch_size*l
        noise_b=torch.from_numpy(np.random.permutation(int(num_b/2)*[-1]+int(num_b/2)*[1])).float().view(batch_size,-1).cuda()
        for i in range(batch_size):
            idx = torch.from_numpy(np.random.permutation(x.size(1)))[:l]
            x[i,idx] =noise_b[i,:].cuda()
        return x
    def collect_weight(self,fm,assign):
        with torch.no_grad():
            update_w=torch.einsum('bmn,bmc->bmnc',assign,fm).sum(1).sum(0)    
            assign_mean=assign.sum(1).sum(0)
            for i in range(self.num_parts):
                if(assign_mean[i].data>1):
                    self.w_ep_sum[i]=self.w_ep_sum[i]+(update_w[i])
                    self.w_ep_num[i]=self.w_ep_num[i]+assign_mean[i]
    def cal_w(self):
         with torch.no_grad():
            for i in range(self.num_parts):
                if(self.w_ep_num[i].data>0):
                    self.w_ep_sum[i]=self.w_ep_sum[i]/self.w_ep_num[i]
            self.w=self.w_ep_sum.data.clone()
            self.w_ep_num=torch.ones(self.num_parts).cuda()
    def do_inter(self,fm,embedding,epoch):
            inputs=fm
            assert inputs.dim() == 4
            batch_size = inputs.size(0)
            in_channels = inputs.size(1)
            input_h = inputs.size(2)
            input_w = inputs.size(3) 
            assert in_channels == self.channels
            inputs = fm.permute(0, 2, 3, 1).contiguous().clone().cuda()
            inputs_flatten = inputs.view(batch_size * input_h * input_w, in_channels)
            distances = (torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)
                            + torch.sum(self.w.data ** 2, dim=1)
                            - 2 * torch.matmul(inputs_flatten, self.w.data.t()))
            distances = distances.view(batch_size,input_h*input_w,-1)
            assign = -distances
            assign = nn.functional.softmax(assign, dim=-1).clone()
            assign[assign.data<1/self.num_parts]=torch.tensor(0.0).cuda()
            assign_sum=assign.mean(1)
            if epoch>2:
                c=self.relu(torch.einsum('bn,nc->bnc',assign_sum,self.w.data).mean(1))
                embedding_cat=torch.cat((embedding,c),dim=-1)
            else :
                embedding_cat=torch.cat((embedding,torch.zeros_like(embedding.data)),dim=-1)
            self.collect_weight(inputs.view(batch_size,-1,in_channels).data,assign.data)
            return embedding_cat
    def forward(self, x,epoch=0,is_train=False):
        if is_train==True:
            fm, embedding = self.pretrained_model(x)
            embedding_re=self.do_inter(fm,embedding,epoch)
            y_hat = self.rawcls_net(embedding_re)
            hash_code=self.hash_layer(embedding)
            y_hat_hash = self.hash_cls(self.BNI(hash_code))
            return y_hat,y_hat_hash,hash_code
        else:
            V, g = self.pretrained_model(x)
            hash_code = self.hash_layer(g)
            return hash_code


