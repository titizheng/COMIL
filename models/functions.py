
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
import torch.nn as nn

class Memory:
    def __init__(self):
        '''
        memory pseudo-bag representation
        '''   
        self.msg_states = [] # pseudo-bag representation
        self.merge_msg_states = [] # cross pseudo-bag representation
        self.results_dict = []

        

    def clear_memory(self):
        del self.msg_states[:]
        del self.merge_msg_states[:]
        del self.results_dict[:]
        
        
class cross_Attention(nn.Module):
    def __init__(self):
        super(cross_Attention, self).__init__()
    def forward(self, query, keys, values): 
        similarity = torch.matmul(query, keys.transpose(-2, -1))
        attention_weights = F.softmax(similarity, dim=-1)
        output = torch.matmul(attention_weights, values) 
        output = output + query
        return output



class FusionHistoryFeatures(nn.Module):
    def __init__(self, embed_dim):
        super(FusionHistoryFeatures, self).__init__()
        
        # self.hidden_state_dim = hidden_state_dim
        self.embed_dim = embed_dim
        self.cross_Attention = cross_Attention()

    def forward(self):
        raise NotImplementedError
    
    def SFFR(self, msg_state, memory):
        # x_groups, msg_tokens_num = state_ini
        # msg_state = x_groups[0][:,:,0:1].squeeze(dim=0)#([1, 1, 512])
        old_msg_state = torch.stack(memory.msg_states[:], dim=1).view(1,-1,self.embed_dim) #([1, 1, 512])
        msg_state = self.cross_Attention(msg_state,old_msg_state,old_msg_state)
        memory.merge_msg_states[-1] = msg_state  #
 
        
        
class NCL_block(nn.Module):
    def __init__(self, feature_dim, num_prototypes):
        super(NCL_block, self).__init__()
        self.Linear = nn.Linear(feature_dim, num_prototypes, bias=False)
    
    def forward(self, x):
        x =  torch.stack(x[:], dim=1).view(-1,512).to(x[1].device)
        proto_c = self.Linear(x)
        p_a= F.relu(proto_c)  # 
        return  p_a




def info_ncl_loss(features, window_size=1, temperature=0.1):
    """
    Args:
        features (Tensor): anchor [N, D]。
        window_size (int): 
        temperature (float): 
    """
    N, D = features.size()
    
    norm_features = F.normalize(features, p=2, dim=1)
    all_similarities = torch.matmul(norm_features, norm_features.T) / temperature  # [N, N]

    total_loss = 0.0
    for index in range(N):
        
        positive_indices = list(range(max(0, index - window_size), min(N, index + window_size + 1)))
        positive_indices.remove(index)

        
        negative_indices = list(set(range(N)) - set(positive_indices) - {index})

        
        positive_sim = all_similarities[index, positive_indices]
        negative_sim = all_similarities[index, negative_indices]

        
        
  
        logits = torch.cat([positive_sim.unsqueeze(0), negative_sim.unsqueeze(0)], dim=1)  # [1, P+K]
        log_prob = F.log_softmax(logits, dim=1)  # [1, P+K]
        loss_i = -torch.logsumexp(log_prob[:, :len(positive_indices)], dim=1)  # [1]
        
        total_loss += loss_i
    return total_loss / N
