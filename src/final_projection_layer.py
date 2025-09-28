import torch
import torch.nn as nn

class finalProjectionLayer(nn.Module):
    def __init__(self,emb_dim,vocab_size):
        super().__init__()
        self.linear=nn.Linear(emb_dim,vocab_size)
    
    def forward(self,x):
        output=self.linear(x)
        return output