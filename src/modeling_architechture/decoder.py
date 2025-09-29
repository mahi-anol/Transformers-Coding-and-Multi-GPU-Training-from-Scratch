import torch.nn as nn
from decoder_block import decoderBlock
class decoder(nn.Module):
    def __init__(self,no_of_enc_blk,emb_dim,n_heads,mha_dropout,expand_dim,ff_dropout,sk_dropout):
        super().__init__()
        self.dec_blks=nn.ModuleList([decoderBlock(emb_dim,n_heads,mha_dropout,expand_dim,ff_dropout,sk_dropout) for _ in range(no_of_enc_blk)])
    def forward(self,x,mask=None):
        for blk in self.dec_blks:
            x=blk(x,mask)
        return x