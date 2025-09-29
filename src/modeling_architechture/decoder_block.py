import torch
import torch.nn as nn
from multihead_attention_block import multiHeadAttentionBlock
from feed_forward_layer import feed_forward_block
from skip_connection import skipConnection
from layer_normalization_block import layerNormalizationBlock

class decoderBlock(nn.Module):
    def __init__(self,emb_dim,n_heads,mha_dropout,expand_dim,ff_dropout,sk_dropout):
        super().__init__()
        self.emb_dim=emb_dim
        self.n_heads=n_heads
        self.ff_dropout=ff_dropout
        self.expand_dim=expand_dim
        self.mha_dropout=mha_dropout
        self.sk_dropout=sk_dropout
        self.mha_block1=multiHeadAttentionBlock(self.emb_dim,self.n_heads,self.mha_dropout) ### casual attention block
        self.mha_block2=multiHeadAttentionBlock(self.emb_dim,self.n_heads,self.mha_dropout) #### cross attention block
        self.feed_forward_block=feed_forward_block(self.emb_dim,self.expand_dim,self.ff_dropout)
        self.skip_connections=nn.ModuleList([skipConnection(self.sk_dropout) for _ in range(3)])
        self.layerNormalizationBlocks=nn.ModuleList([layerNormalizationBlock(self.emb_dim) for _ in range(3)])
    def forward(self,x,enc_out,mask1=None,mask2=None):
        output1=self.skip_connections[0](x,lambda x: self.mha_block1(x,x,x,mask1))
        layer_normalized_output1=self.layerNormalizationBlocks[0](output1)
        output2=self.skip_connections[1](layer_normalized_output1,lambda x: self.mha_block2(x,enc_out,enc_out,mask2))
        layer_normalized_output2=self.layerNormalizationBlocks[1](output2)
        output3=self.skip_connections[2](layer_normalized_output2,self.feed_forward_block)
        layer_normalized_output3=self.layerNormalizationBlocks[2](output3)
        return layer_normalized_output3