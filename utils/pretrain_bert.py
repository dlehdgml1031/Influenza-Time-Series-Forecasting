import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from transformers import BertTokenizer, BertForMaskedLM, BertConfig, TrainingArguments, Trainer


class MeditationsDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}


def _pretrain_bert(
    bert_model_id:str,
    raw_text_path:str,
    text_type:str,
    batch_size:int = 256,
    train_epochs:int = 4,
    ):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_id)
    bert_model = BertForMaskedLM.from_pretrained(bert_model_id)
    bert_model.to(device)
    
    df = pd.read_csv(raw_text_path, index_col = 0)
    df.dropna(inplace=True)
    text = df['Text'].to_list()
    
    inputs = bert_tokenizer(
        text,
        return_tensors = 'pt',
        max_length = 512,
        truncation = True,
        padding = 'max_length'
    )
    
    inputs['labels'] = inputs.input_ids.detach().clone()
    
    rand = torch.rand(inputs.input_ids.shape)
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)
    
    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )
    
    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103
        
    dataset = MeditationsDataset(inputs)
    
    args = TrainingArguments(
        output_dir = 'results',
        per_device_train_batch_size = batch_size,
        num_train_epochs = train_epochs,
        save_strategy = 'epoch',
        fp16 = True,
        seed = 0,
    )
    
    trainer = Trainer(
        model = bert_model,
        args = args,
        train_dataset = dataset
    )
    
    print('Start training BERT')
    trainer.train()
    print('Finish training BERT')
    
    bert_model.save_pretrained(f'./pretrained_bert/{text_type}')
    bert_tokenizer.save_pretrained(f'./pretrained_bert/{text_type}')
   
 
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"
    
    model_dir = "/mnt/nvme01/huggingface/models/"
    model_id = 'Google/bert-base-uncased'
    
    _pretrain_bert(
        bert_model_id = model_dir + model_id,
        raw_text_path = './data/Text_PT/patent_combined_raw.csv',
        text_type = 'Patent',
        batch_size = 64,
        train_epochs = 2
    )