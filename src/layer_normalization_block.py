import torch
import torch.nn as nn
class layerNormalizationBlock(nn.Module):
    def __init__(self,emb_dim,eps=1e-5):
        super().__init__()
        self.scale=nn.Parameter(torch.ones(emb_dim,dtype=torch.float16))
        self.shift=nn.Parameter(torch.zeros(emb_dim,dtype=torch.float16))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(dim=-1,keepdim=True)
        standard_deviation=x.std(dim=-1,keepdim=True,unbiased=False)
        normalized_x=(x-mean)/(standard_deviation+self.eps)
        scale_n_shift=self.scale*normalized_x+self.shift
        return scale_n_shift