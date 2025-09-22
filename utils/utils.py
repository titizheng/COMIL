import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,accuracy_score
from sklearn.preprocessing import label_binarize
import os
import shutil






def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='512',type=str,help=['512','1024','384']) 
 
    parser.add_argument('--seed', default=1,type=int) #2021
    parser.add_argument('--epoch', default=100,type=int)
    parser.add_argument('--lr', default=0.00001,type=int)
    parser.add_argument('--pseudobag_size', type=int, default=512)
    parser.add_argument('--train_T', type=int, default=5)
    parser.add_argument('--test_T', type=int, default=1)

    parser.add_argument('--feature_dim', type=int, default=512)# 
    parser.add_argument('--in_chans', default=512,type=int, help=['1024','512']) 
    parser.add_argument('--embed_dim', default=512,type=int)
    

    # parser.add_argument('--num_subbags', default=0,type=int,help='') #
    
    # parser.add_argument('--attn', default='normal',type=str)
    # parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--num_layers', default=2,type=int) 
    parser.add_argument('--save_dir', default='',type=str,help='saver model path pth')

    
    # parser.add_argument('--ape_class', default=False,type=bool,help=' ') # 
    


    # NCL loss
    parser.add_argument('--instaceclass', default=True,type=bool,help=' ') # 
    parser.add_argument('--CE_CL', default=True,type=bool,help=' ') # 
    parser.add_argument('--num_prototypes', type=int, default=128,help='pos,neg')

    #datasets
    parser.add_argument('--h5',default='',type=str) 
    parser.add_argument('--csv', default='',type=str) 
    parser.add_argument('--n_classes', default=2,type=int)

   
    args = parser.parse_args()
    return args






def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    schedule_per_epoch = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            value = np.linspace(start_warmup_value, base_value, warmup_epochs)[epoch]
        else:
            iters_passed = epoch * niter_per_ep
            iters_left = epochs * niter_per_ep - iters_passed
            alpha = 0.5 * (1 + np.cos(np.pi * iters_passed / (epochs * niter_per_ep)))
            value = final_value + (base_value - final_value) * alpha
        schedule_per_epoch.append(value)
    return schedule_per_epoch




def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error


#!class--2
def calculate_metrics_two(y_true, probs):
    pos_probs = probs[:, 1] if probs.ndim > 1 else probs
    y_pred = (pos_probs >= 0.5).astype(int)
    precision = precision_score(y_true, y_pred)  #
    recall = recall_score(y_true, y_pred)  #
    f1 = f1_score(y_true, y_pred)  
    roc_auc = roc_auc_score(y_true, pos_probs)
    acc = accuracy_score(y_true, y_pred)
    return precision, recall, f1, roc_auc, acc


# !class--3
def calculate_metrics_multi(y_true, probs):
    y_pred = np.argmax(probs, axis=1) 
    precision = precision_score(y_true, y_pred, average='macro') 
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro') 
    y_true_one_hot = label_binarize(y_true, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_true_one_hot, probs, average='macro')
    acc = np.mean(y_pred == y_true)
    return precision, recall, f1, roc_auc,acc






class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.flag = False

    def __call__(self, epoch, val_loss, model, args, ckpt_name = ''):
        ckpt_name = './ckp/{}_checkpoint_{}_{}.pt'.format(str(args.type),str(args.seed),str(epoch))
        score = -val_loss
        self.flag = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
            self.counter = 0
        

    def save_checkpoint(self, val_loss, model, ckpt_name, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose and not args.overfit:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)
        elif self.verbose and args.overfit:
            print(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)           
        torch.save(model.state_dict(), ckpt_name)
        print(ckpt_name)
        self.val_loss_min = val_loss
        self.flag = True


def save_checkpoint(state,epoch,best_acc,best_auc,checkpoint, filename='checkpoint.pth.tar'):
    best_acc = f"{best_acc:.4f}"
    best_auc = f"{best_auc:.4f}"
    # 转换为浮点数
    best_acc_float = float(best_acc)
    best_auc_float = float(best_auc)

    sum_value = best_acc_float + best_auc_float

    sum_str = str(int(sum_value))

    filepath = os.path.join(checkpoint, str(epoch)+"_"+best_acc+"_"+best_auc+"_"+sum_str+"_"+filename)
    torch.save(state, filepath)
    
