import torch
import torch.nn as nn
import math

class positionalEncodingLayer(nn.Module):
    def __init__(self,max_seq_len,emb_dim):
        super().__init__()
        self.max_seq_len=max_seq_len
        self.emb_dim=emb_dim
        static_positional_info=torch.zeros((self.max_seq_len,self.emb_dim),dtype=torch.float16)
        positions=torch.arange(0,max_seq_len,dtype=torch.float16).reshape(max_seq_len,-1)
        indices_for_denominator=torch.arange(0,emb_dim,2,dtype=torch.float16) ### 2i
        denominators=torch.exp((-2*indices_for_denominator*math.log(1e4))/emb_dim)
        static_positional_info[:,0::2]=torch.sin(positions*denominators)
        static_positional_info[:,1::2]=torch.cos(positions*denominators)
        self.register_buffer('static_positional_info',static_positional_info)
    def forward(self,x):
        position_encoded_embedding=x+self.static_positional_info[:x.shape[1],:]
        return position_encoded_embedding
