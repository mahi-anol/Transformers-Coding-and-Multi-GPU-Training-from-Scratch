import torch
import torch.nn as nn
class feed_forward_block(nn.Module):
    ### Expansion Contraction layer.....
    def __init__(self,emb_dim,expand_dim,dropout):
        super().__init__()
        self.emb_dim=emb_dim
        self.expand_dim=expand_dim
        self.dropout=dropout
        self.network=nn.Sequential(
            nn.Linear(emb_dim,expand_dim,dtype=torch.float16),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(expand_dim,emb_dim,dtype=torch.float16),
        )
    def forward(self,x):
        output=self.network(x)
        return output