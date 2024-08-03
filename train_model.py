import torch
import numpy as np
import time
import os
from main_eval import eval
import pandas as pd
def train(config,model, train_dataloader, test_dataloader, base_dataloader, criterion, criterion_qua, optimizer, scheduler, save_dir, epochs, classes):
    MAP_save=0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels, _ in train_dataloader:
            inputs = inputs.cuda()
            labels = torch.from_numpy(np.array(labels)).cuda()
            optimizer.zero_grad()
            y_hat, y_hash,b = model(inputs,epoch,is_train=True)
            L_g = criterion(y_hat, labels)+criterion(y_hash,labels)
            L_h = criterion_qua(torch.abs(b),torch.ones(b.data.shape).cuda())
            loss = L_g + L_h
            loss.backward()
            optimizer.step()
            total_loss += loss.item() 
        model.cal_w()
        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
        print('epoch:{},bit:{},loss:{:.4f},time:{}'.format(epoch, config.bit_length,total_loss,current_time ))
        
        scheduler.step()
        if(epoch>85):
            save_path = os.path.join(save_dir, 'weights_%s_%sbit_%depoch.pth' % (config.dataset,config.bit_length,epoch)) 
            torch.save(model.state_dict(), save_path)
        if ((epoch+1)%30==0)or (epoch>85):
            print('#####################################################')
            MAP=eval(config,train_dataloader, test_dataloader, base_dataloader,model,classes,MAP_save,epoch)
            if (MAP>=MAP_save):
                MAP_save=MAP

