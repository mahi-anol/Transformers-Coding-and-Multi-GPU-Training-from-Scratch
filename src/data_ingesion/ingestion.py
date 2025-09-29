from datasets import load_dataset
def data_ingestion():
    ### this method returns the train train_validation_test split for opus books dataset
    ds = load_dataset(path="Helsinki-NLP/opus_books", name="en-fr") ### getting the dataset from huggingface.
    train_test_data=ds['train'].train_test_split(test_size=0.2,seed=42)
    test_data=train_test_data['test']
    train_val_split=train_test_data['train'].train_test_split(test_size=0.2,seed=42)
    train_data=train_val_split['train']
    validation_data=train_val_split['test']
    return train_data,validation_data,test_data

#train_data,validation_data,test_data=data_ingestion()