import torch.nn as nn
from encoder_block import encoderBlock
class encoder(nn.Module):
    def __init__(self,no_of_enc_blk,emb_dim,n_heads,mha_dropout,expand_dim,ff_dropout,sk_dropout):
        super().__init__()
        self.enc_blks=nn.ModuleList([encoderBlock(emb_dim,n_heads,mha_dropout,expand_dim,ff_dropout,sk_dropout) for _ in range(no_of_enc_blk)])
    def forward(self,x,mask=None):
        for blk in self.enc_blks:
            x=blk(x,mask)
        return x