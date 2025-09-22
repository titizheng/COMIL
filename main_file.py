import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

from utils.utils import make_parse ,cosine_scheduler
from utils.core import train ,seed_torch
from utils.createmode import create_model
from torch.utils.data import DataLoader
from datasets.loda_datasets import h5file_Dataset
import torch.nn as nn
import pandas as pd
import torch
import numpy as np
import random





def train_main(args):



    fold = 'fold0' #_fold1_fold2_fold3_fold4'
    args.h5 = './dataset/tcga-brca/h5_files' #features
    args.csv = f'./dataset/tcga-brca/five_flod_csv/{fold}.csv' #csv

    seed_torch(seed=1)
    from datetime import datetime
    time_str = datetime.now().strftime("%m%d_%H%M")  #
    args.save_dir = os.path.join(
    './result',
    f"{fold}_{time_str}")
    os.makedirs(args.save_dir, exist_ok=True)

    

    basedmodel,classifymodel,memory, NCL_model,Fusion_hispseudobag= create_model(args)
   
    train_dataset = h5file_Dataset(args.csv,args.h5,'train') 
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True ) # , num_workers=2, persistent_workers=True
    
    valid_dataset = h5file_Dataset(args.csv,args.h5,'val')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2, persistent_workers=True )

    test_dataset = h5file_Dataset(args.csv,args.h5,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, persistent_workers=True )

    lambda_pseudo_bag_list = cosine_scheduler(0,0.1,epochs=args.epoch,niter_per_ep=len(train_dataset),start_warmup_value=1.)
    lambda_ncls_list = cosine_scheduler(0.5,0,epochs=args.epoch,niter_per_ep=len(train_dataset),start_warmup_value=1.)

    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(list(basedmodel.parameters())+list(classifymodel.parameters())+list(Fusion_hispseudobag.parameters())+list(NCL_model.parameters()), lr=1e-4, weight_decay=1e-5) 

    best_acc = 0.0
    best_auc = 0.0
    best_F1 = 0.0
    res_list = []
    for epo in range(args.epoch): 
        best_acc,best_auc,best_F1,test_precision, test_recall, test_f1, test_auc, test_acc = train(args,basedmodel,classifymodel,memory,Fusion_hispseudobag,NCL_model,epo,train_loader,valid_loader,lambda_ncls_list[epo],lambda_pseudo_bag_list[epo],loss_fn,optimizer,best_acc,best_auc,best_F1,test_dataloader)

        res_list.append([epo,test_acc,test_auc, test_precision, test_recall, test_f1 ])
       
        df = pd.DataFrame(res_list, columns=['epoch', 'test_acc', 'test_auc', 'test_precision', 'test_recall', 'test_f1'])
        from datetime import datetime

        current_time = datetime.now().strftime("%Y-%m-%d")
        df.to_csv(os.path.join(args.save_dir,f'{current_time}_result.csv'), index=False)
        
if __name__ == "__main__":
    
    args = make_parse()
    train_main(args)
    

