import torch
import torch.nn as nn
import math

class inputEmbeddingLayer(nn.Module):
    def __init__(self,vocab_size,emb_dim):
        super().__init__()
        self.vocab_size=vocab_size
        self.emb_dim=emb_dim
        self.embedding_layer=nn.Embedding(self.vocab_size,self.emb_dim,dtype=torch.float16)

    def forward(self,x):
        embeddings=self.embedding_layer(x)
        return embeddings*math.sqrt(self.emb_dim)