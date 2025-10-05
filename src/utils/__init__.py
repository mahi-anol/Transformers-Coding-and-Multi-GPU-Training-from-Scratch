import torch

def casual_mask_generator(size):
    mask=torch.triu(torch.ones(1,size,size),diagonal=1).type(torch.int)
    return mask==0

def get_max_seq_len(train_data,test_data,validation_data):
    max_len=0
    for data in train_data:
        max_len=max(max_len,len(tokenizer.encode(data['translation']['en']).ids))
        max_len=max(max_len,len(tokenizer.encode(data['translation']['fr']).ids))

    for data in test_data:
        max_len=max(max_len,len(tokenizer.encode(data['translation']['en']).ids))
        max_len=max(max_len,len(tokenizer.encode(data['translation']['fr']).ids))

    for data in validation_data:
        max_len=max(max_len,len(tokenizer.encode(data['translation']['en']).ids))
        max_len=max(max_len,len(tokenizer.encode(data['translation']['fr']).ids))

    return max_len    