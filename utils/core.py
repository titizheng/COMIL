import torchmetrics
import torch.nn as nn
import torch
import copy
from utils.utils import calculate_error,calculate_metrics_two,calculate_metrics_multi,save_checkpoint

import numpy as np
# from sklearn.cluster import KMeans
 
from tqdm import tqdm
import torch.nn.functional as F
from models.functions import info_ncl_loss # 



def val (args,basedmodel,classifymodel,Fusion_hispseudobag,NCL_model,memory,valid_loader,device):

    basedmodel.eval()
    classifymodel.eval()
    NCL_model.eval()
    Fusion_hispseudobag.eval()
    
    valid_error = 0.

    with torch.no_grad():
        val_label_list = []
        val_Y_prob_list = []
        for idx, (coords, data, label) in enumerate (valid_loader):


            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()
            if args.type == 'camelyon16_Res50_imageNet':
                update_data = basedmodel.fc1(update_data) 
            else:
                update_data = update_data.float() 
            if args.ape: 
                update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim)
                
            update_coords, update_data ,total_T = expand_data(update_coords, update_data, pseudobag_size = args.pseudobag_size, total_steps=args.test_T)
            grouping_instance = grouping(pseudobag_size=args.pseudobag_size)  
           
            for patch_step in range(0, total_T):
            
                features_group , update_coords,update_data ,memory = grouping_instance.make_subbags(memory,update_coords,update_data,pseudobag_size = args.pseudobag_size ) 
                results_dict, memory = basedmodel(Fusion_hispseudobag, features_group, memory, update_coords,mask_ratio=0)
                # logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat']       
                
            W_results_dict,memory = classifymodel (memory)  
            W_logits, W_Y_prob, W_Y_hat = W_results_dict['logits'], W_results_dict['Y_prob'], W_results_dict['Y_hat']


            error = calculate_error(W_Y_hat, label)
            valid_error += error

            memory.clear_memory()
            val_label_list.append(label)
            val_Y_prob_list.append(W_Y_prob)



        targets = np.asarray(torch.cat(val_label_list, dim=0).cpu().numpy()).reshape(-1) 
        probs = np.asarray(torch.cat(val_Y_prob_list, dim=0).cpu().numpy()) 
        if args.n_classes == 2:
            precision, recall, f1, auc, acc = calculate_metrics_two(targets, probs) 
        else:
            precision, recall, f1, auc, acc = calculate_metrics_multi(targets, probs)
 
        
        print(f'Accuracy: {acc:.4f} , Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
        return precision, recall, f1, auc, acc 




def train(args,basedmodel,classifymodel,memory,Fusion_hispseudobag,NCL_model,epoch,train_loader,valid_loader,lambda_ncls,lambda_pseudo_bag,loss_fn,optimizer,best_acc,best_auc,best_F1,test_loader):

    print('-------training-------')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    basedmodel.train()
    classifymodel.train()
    NCL_model.train()
    Fusion_hispseudobag.train()
    

    
    train_error = 0.
    train_bag_loss = 0.
    train_pseudo_bag_loss = 0.
    train_NCL_loss = 0.
    
    progress_bar = tqdm(train_loader, desc="Training", position=0, leave=True)
   
    for idx, (coords, data, label) in enumerate(progress_bar):
        update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long() 
        grouping_instance = grouping(pseudobag_size=args.pseudobag_size) 
        train_instancetoken_loss =[] 

        if args.type != '512':
            update_data = basedmodel.fc1(update_data) 
        else:
            update_data = update_data.float() 
        if args.ape: 
            update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim) 
        
        update_coords, update_data, total_T = expand_data(update_coords, update_data,pseudobag_size = args.pseudobag_size, total_steps=args.train_T)

        for patch_step in range(0, total_T):
            features_group , update_coords,update_data ,memory = grouping_instance.make_subbags(memory,update_coords,update_data,pseudobag_size = args.pseudobag_size)  
            results_dict, memory = basedmodel(Fusion_hispseudobag,features_group, memory, update_coords,mask_ratio=0)

            
            logits, Y_prob, Y_hat = results_dict['logits'], results_dict['Y_prob'], results_dict['Y_hat'] 
            train_class_instance_loss = torch.nn.functional.cross_entropy(logits, label).view(1, -1) 
            train_instancetoken_loss.append(train_class_instance_loss) 
            

        
        s_a= NCL_model(memory.merge_msg_states )  
        loss_NCL = info_ncl_loss(s_a)  

        W_results_dict,memory = classifymodel (memory)  
        W_logits, W_Y_prob, W_Y_hat = W_results_dict['logits'], W_results_dict['Y_prob'], W_results_dict['Y_hat']

        pseudo_bag_level_loss =torch.mean(torch.cat(train_instancetoken_loss, dim=0))
        bag_level_loss = loss_fn(W_logits, label)

        classify_loss = bag_level_loss+ lambda_ncls*loss_NCL +  lambda_pseudo_bag*pseudo_bag_level_loss 
      
        error = calculate_error(W_Y_hat, label)
        train_error += error

        train_bag_loss = train_bag_loss + bag_level_loss.item()
        train_pseudo_bag_loss = train_pseudo_bag_loss + pseudo_bag_level_loss.item()
        train_NCL_loss = train_NCL_loss + loss_NCL.item()

        optimizer.zero_grad()
        classify_loss.backward()
        optimizer.step()
        memory.clear_memory()
        progress_bar.set_description(f"Training (epoch = {epoch},loss={bag_level_loss.item():.4f}, error={error:.4f})")

    train_error /= len(train_loader)
    train_bag_loss /=  len(train_loader)
    train_pseudo_bag_loss /= len(train_loader)
    train_NCL_loss /= len(train_loader)

    print('train_acc: {:.4f},train_bag_loss:{:.4f},train_pseudo_bag_loss:{:.4f},train_NCL_loss:{:.4f}'.format(1-train_error,train_bag_loss,train_pseudo_bag_loss,train_NCL_loss))
    progress_bar.close()

    print('-------val-------')
    precision, recall, f1, auc, acc = val (args,basedmodel,classifymodel,Fusion_hispseudobag,NCL_model,memory,valid_loader,device)
    state = {
        'epoch': epoch + 1,
        'model_state_dict': basedmodel.state_dict(),
        'classifymodel': classifymodel.state_dict(),
        'Fusion_hispseudobag':Fusion_hispseudobag.state_dict(),
        'NCL_model': NCL_model.state_dict(),
    }
    if  acc + f1  >= best_acc+ best_F1 : 
        best_acc = acc
        best_F1 = f1
        save_checkpoint(state,epoch,best_acc,best_auc,str(args.save_dir))
        print('-------test-------')
        test_precision, test_recall, test_f1, test_auc, test_acc = val (args,basedmodel,classifymodel,Fusion_hispseudobag,NCL_model,memory,test_loader,device)

    else:
        test_precision, test_recall, test_f1, test_auc, test_acc = -1,-1,-1,-1,-1
    return best_acc,best_auc,best_F1,test_precision, test_recall, test_f1, test_auc, test_acc



def seed_torch(seed=2021):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False 


class grouping:

    def __init__(self,pseudobag_size = 512):

        self.pseudobag_size = pseudobag_size
        self.action_std = 0.1
        
    
    def make_subbags(self,memory, update_coords,update_features,pseudobag_size= None,restart= False,delete_begin=False): 
        B, N, C = update_features.shape
        indices = np.arange(N)
        random_subset = np.random.choice(indices, size=pseudobag_size, replace=False)
        idx = torch.tensor(random_subset)
        features_group = update_features[:, idx[:], :]
        mask = torch.ones(update_features.size(1), dtype=torch.bool) 
        mask[idx] = False  
        updated_features = update_features[:, mask, :] 
        updated_coords = update_coords[:, mask, :]
        return features_group, updated_coords, updated_features ,memory

    
    
    


def expand_data(update_coords, update_datas,pseudobag_size = None, total_steps=None):
    """
    Extend the input coordinates and data to the specified total length to 
    ensure each pseudo-bag has a consistent number of samples during training.
    """
    
    total_length = pseudobag_size * total_steps
    B, current_length, C = update_coords.shape
    

    if current_length >= total_length:
        StopT = int(update_coords.shape[1] / pseudobag_size)  
        remaining_length = current_length - pseudobag_size * StopT
       

        if remaining_length > 0:
            required_length = pseudobag_size - remaining_length

      
            segment_length = 1  
            num_segments = current_length
            
            random_indices = np.random.choice(num_segments, size=required_length, replace=False)
            
            random_coords_segments = [update_coords[:, i:i+segment_length, :] for i in random_indices]
            random_data_segments = [update_datas[:, i:i+segment_length, :] for i in random_indices]
            
            random_coords = torch.cat(random_coords_segments, dim=1)
            random_datas = torch.cat(random_data_segments, dim=1)

            update_coords = torch.cat([update_coords, random_coords], dim=1)
            update_datas = torch.cat([update_datas, random_datas], dim=1)
            return update_coords, update_datas, StopT+1

        return update_coords, update_datas, StopT
    else:
    
        repeat_times = total_length // current_length
        
        update_coords = update_coords.repeat(1, repeat_times, 1) 
        update_datas = update_datas.repeat(1, repeat_times, 1)

        B, current_length, C = update_coords.shape
        remaining_length = total_length - current_length

        if remaining_length <= 0:
            StopT = int(update_coords.shape[1] / pseudobag_size) 
            return update_coords, update_datas, StopT
        
        segment_length = 1 
        num_segments = current_length
        
        random_indices = np.random.choice(num_segments, size=remaining_length, replace=False)
        
        random_coords_segments = [update_coords[:, i:i+segment_length, :] for i in random_indices]
        random_data_segments = [update_datas[:, i:i+segment_length, :] for i in random_indices]
        
        random_coords = torch.cat(random_coords_segments, dim=1)
        random_data = torch.cat(random_data_segments, dim=1)

        update_coords = torch.cat([update_coords, random_coords], dim=1)
        update_datas = torch.cat([update_datas, random_data], dim=1)

        StopT = int(update_coords.shape[1] / pseudobag_size)  
        
        return update_coords, update_datas, StopT









