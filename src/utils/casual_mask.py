import torch
def casual_mask_generator(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0
