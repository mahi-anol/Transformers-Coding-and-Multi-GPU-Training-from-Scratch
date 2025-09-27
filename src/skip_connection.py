import torch
import torch.nn as nn
class skipConnection(nn.Module):
    def __init__(self,dropout):
        super().__init__()
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,sublayer):
        output=x+sublayer(x)
        dropped_output=self.dropout(output)
        return dropped_output