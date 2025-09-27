import torch
import torch.nn as nn
import math

class multiHeadAttentionBlock(nn.Module):
    def __init__(self,emb_dim,n_heads):
        super().__init__()
        assert emb_dim%n_heads==0 ### checking if multi head splitting is possible.
        self.w_q=nn.Linear(emb_dim,emb_dim,dtype=torch.float16)
        self.w_k=nn.Linear(emb_dim,emb_dim,dtype=torch.float16)
        self.w_v=nn.Linear(emb_dim,emb_dim,dtype=torch.float16)
        self.w_o=nn.Linear(emb_dim,emb_dim,dtype=torch.float16) ### multi-head-projection-layer
        self.emb_dim=emb_dim
        self.single_head_dim=self.emb_dim//n_heads
        self.n_heads=n_heads

    @staticmethod
    def contextual_embedding(m_q,m_k,m_v,per_head_emb_dim,mask):
        ### return contexual embedding and attention scores
        attention_scores=m_q@m_k.transpose(2,3)/math.sqrt(per_head_emb_dim)
        ##batch,head,seq,dim @ batch,head,dim,seq==batch,head,seq,seq
        if mask is not None:
            attention_scores.masked_fill_(mask,value=float('-inf'))
        normalized_attention_scores=torch.softmax(attention_scores,dim=-1)
        ### batch,head,seq,seq @ batch,head,seq,dim=batch,head,seq,dim
        contexual_embeddings=normalized_attention_scores@m_v
        return normalized_attention_scores,contexual_embeddings
    
    def forward(self,q,k,v,mask):
        query=self.w_q(q) ### batch, seqeunce, dim
        key=self.w_k(k)
        value=self.w_v(v)

        multihead_query=query.view(query.shape[0],query.shape[1],self.n_heads,self.single_head_dim).transpose(1,2)
        multihead_key=key.view(key.shape[0],key.shape[1],self.n_heads,self.single_head_dim).transpose(1,2)
        multihead_value=value.view(value.shape[0],value.shape[1],self.n_heads,self.single_head_dim).transpose(1,2)
        _,contextual_embeddings=multiHeadAttentionBlock.contextual_embedding(multihead_query,multihead_key,multihead_value,self.single_head_dim,mask)
        final_contextual_embeddings=contextual_embeddings.transpose(1,2).contiguous().view(value.shape[0],value.shape[1],self.n_heads*self.single_head_dim)
        multihead_final_contextual_embedding_proj=self.w_o(final_contextual_embeddings)
        return multihead_final_contextual_embedding_proj