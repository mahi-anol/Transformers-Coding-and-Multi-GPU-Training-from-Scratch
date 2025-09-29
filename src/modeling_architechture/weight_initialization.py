import torch.nn as nn
def weight_initialization(model):
    for p in model.parametes():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)

