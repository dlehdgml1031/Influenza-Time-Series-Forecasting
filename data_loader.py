import os
import sys
from typing import Optional, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.time_series_analysis import dymanic_time_series_lag_analysis

from transformers import BertModel, BertTokenizer

class NaiveILINationalDataset(Dataset):
    def __init__(self, pred_type:str, seq_len:int, train_flag:str,
                 pred_len:int = 1,
                 scale:Optional[bool] = False,
                 ili_path:str = './data/ILI/national_illness_2018.csv',
                 ):
        type_map = {'train': 0, 'test': 1}
        self.pred_type = pred_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_flag = train_flag
        self.ili_path = ili_path
        self.set_type = type_map[train_flag]
        self.scale = scale
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()
        
    def __read_data__(self):
        self.total_patients = pd.read_csv(self.ili_path, index_col = [0, 1])['TOTAL PATIENTS'].to_list()
        labels = pd.read_csv(self.ili_path, index_col = [0, 1])['Label'].to_list()
        self.labels = [1 if label == 'UP' else 0 for label in labels]
        
        assert len(self.total_patients) == len(self.labels)
        
        if self.scale:
            pass
        
        # train, test split (7:3)
        num_train = int(len(self.labels) * 0.7)
        num_test = int(len(self.labels) * 0.3)
        
        border1s = [0, len(self.labels) - num_test - self.seq_len - self.pred_len]
        border2s = [num_train + self.pred_len, len(self.labels)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.total_patients = self.total_patients[border1:border2]
        self.labels = self.labels[border1:border2]
        
    def __len__(self):
        return len(self.labels) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = index + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        total_patients = torch.tensor(self.total_patients[s_begin:s_end], dtype = torch.float32)
        label = torch.tensor(self.labels[r_begin:r_end], dtype = torch.long)
        
        return total_patients, label
        
        

class FluCastDataset(Dataset):
    def __init__(self, pred_type:str, seq_len:int, train_flag:str,
                 pred_len:int = 1,
                 scale:Optional[bool] = False,
                 news_texts_path:str = './data/Text_Summarization/news_summarization.csv',
                 abstract_texts_path:str = './data/Text_Summarization/abstract_summarization.csv',
                 patent_texts_path:str = './data/Text_Summarization/patent_summarization.csv',
                 ili_path:str = './data/ILI/national_illness_2018.csv',
                 news_bert_model_id:str = './pretrained_bert/News',
                 abstract_bert_model_id:str = './pretrained_bert/Abstract',
                 patent_bert_model_id:str = './pretrained_bert/Patent'):
        type_map = {'train': 0, 'test': 1}
        self.pred_type = pred_type
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_flag = train_flag
        self.news_texts_path = news_texts_path
        self.abstract_texts_path = abstract_texts_path
        self.patent_texts_path = patent_texts_path
        self.ili_path = ili_path
        self.news_bert_model_id = news_bert_model_id
        self.abstract_bert_model_id = abstract_bert_model_id
        self.patent_bert_model_id = patent_bert_model_id
        self.scale = scale
        self.set_type = type_map[train_flag]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__read_data__()
    
    def __read_data__(self):
        news_texts = pd.read_csv(self.news_texts_path, index_col = 0)['Summary'].to_list()
        self.news_embeddings = self._get_embeddings(news_texts, self.news_bert_model_id)
        
        abstract_texts = pd.read_csv(self.abstract_texts_path, index_col = 0)['Summary'].to_list()
        self.abstract_embeddings = self._get_embeddings(abstract_texts, self.abstract_bert_model_id)
        
        patent_texts = pd.read_csv(self.patent_texts_path, index_col = 0)['Summary'].to_list()
        self.patent_embeddings = self._get_embeddings(patent_texts, self.patent_bert_model_id)
        
        self.total_patients = pd.read_csv(self.ili_path, index_col = [0, 1])['TOTAL PATIENTS'].to_list()
        labels = pd.read_csv(self.ili_path, index_col = [0, 1])['Label'].to_list()
        self.labels = [1 if label == 'UP' else 0 for label in labels]
        
        assert len(self.news_embeddings) == len(self.abstract_embeddings) == len(self.patent_embeddings) == len(self.labels) == len(self.total_patients)
        
        if self.scale:
            pass
        
        # train, test split (7:3)
        num_train = int(len(self.labels) * 0.7)
        num_test = int(len(self.labels) * 0.3)
        
        # border1s = [0, len(self.labels) - num_test - self.seq_len]
        # border2s = [num_train, len(self.labels)]
        
        border1s = [0, len(self.labels) - num_test - self.seq_len - self.pred_len]
        border2s = [num_train + self.pred_len, len(self.labels)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.news_embeddings = self.news_embeddings[border1:border2]
        self.abstract_embeddings = self.abstract_embeddings[border1:border2]
        self.patent_embeddings = self.patent_embeddings[border1:border2]
        self.total_patients = self.total_patients[border1:border2]
        self.labels = self.labels[border1:border2]
        
    def __len__(self):
        return len(self.labels) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = index + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        news_embeddings = torch.tensor(self.news_embeddings[s_begin:s_end], dtype = torch.float32)
        abstract_embeddings = torch.tensor(self.abstract_embeddings[s_begin:s_end], dtype = torch.float32)
        patent_embeddings = torch.tensor(self.patent_embeddings[s_begin:s_end], dtype = torch.float32)
        total_patients = torch.tensor(self.total_patients[s_begin:s_end], dtype = torch.float32)
        label = torch.tensor(self.labels[r_begin:r_end], dtype = torch.long)
        
        return news_embeddings, abstract_embeddings, patent_embeddings, total_patients, label
    
    def _get_embeddings(self, texts:List, bert_model_id:str):
        embeddings = []
        
        bert_model = BertModel.from_pretrained(bert_model_id).to(self.device)
        bert_tokenizer = BertTokenizer.from_pretrained(bert_model_id)
        
        for text in texts:
            inputs = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')

            with torch.no_grad():
                outputs = bert_model(**inputs)
            
            mean_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            embeddings.append(mean_embedding)
        
        return embeddings


class Dataset_ILI_National(Dataset):
    def __init__(self,
                 data_path:str,
                 seq_len:int,
                 pred_len:int,
                 scale:bool = True,
                 time_predence:str = 'D',
                 fixed_time_predence:int = 3,
                 flag:str = 'train',
                 target:str = 'TOTAL PATIENTS'):
        """
        ILI National Dataset for training and testing.

        Parameters
        ----------
        data_path : str
            _description_
        seq_len : int
            _description_
        pred_len : int
            _description_
        scale : bool, optional
            _description_, by default True
        time_predence : str, optional
            _description_, by default 'D'
        fixed_time_predence : int, optional
            _description_, by default 3
        flag : str, optional
            _description_, by default 'train'
        target : str, optional
            _description_, by default 'TOTAL PATIENTS'
        """
        
        # init
        # assert flag in ['train', 'test', 'val']
        # type_map = {'train': 0, 'val': 1, 'test': 2}
        
        assert flag in ['train', 'test']
        assert time_predence in ['D', 'F', 'N']
        assert fixed_time_predence <= pred_len and fixed_time_predence >= 1
        
        type_map = {'train': 0, 'test': 1}
        
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.time_predence = time_predence
        self.target = target
        self.fixed_time_predence = fixed_time_predence
        self.__read_data__()
    
    def __read_data__(self):
        # read csv file
        df_data = pd.read_csv(self.data_path, index_col = [0,1])
        cols = df_data.columns.tolist()
        cols.remove(self.target)
        df_data = df_data[cols + [self.target]]
        
        # train, test split (7:3)
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.3)
        
        # border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_data)]
        
        if self.time_predence == 'D':
            border1s = [0, len(df_data) - num_test - self.seq_len - self.pred_len]
            border2s = [num_train + self.pred_len, len(df_data)]
            
        elif self.time_predence == 'F':
            border1s = [0, len(df_data) - num_test - self.seq_len - self.fixed_time_predence]
            border2s = [num_train + self.fixed_time_predence, len(df_data)]
        
        elif self.time_predence == 'N':
            border1s = [0, len(df_data) - num_test - self.seq_len]
            border2s = [num_train, len(df_data)]
        
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        if self.time_predence == 'D':
            # input seq index
            s_begin = index + self.pred_len
            s_end = index + self.seq_len + self.pred_len
            
            # traget seq index
            r_begin = s_end
            r_end = r_begin + self.pred_len
            
            # define input seq and target seq
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end][:, -1] # target is the last column
            seq_y = seq_y.reshape(-1, 1) # reshape to (pred_len, 1)
            
            """
            The code below will be modified in the future to reflect the time difference for each variable differently
            """
            # claculate the lag and optimal lag
            lag, optimal_lag = dymanic_time_series_lag_analysis(seq_x, seq_y)
            
            # redefine the input seq with lag
            s_begin = s_begin - optimal_lag
            s_end = s_end - optimal_lag
            
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end][:, -1]
            
            return seq_x, seq_y
        
        elif self.time_predence == 'F':
            # input seq index
            s_begin = index + self.fixed_time_predence
            s_end = index + self.seq_len + self.fixed_time_predence
            
            # traget seq index
            r_begin = s_end
            r_end = r_begin + self.pred_len
            
            # define input seq and target seq
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end][:, -1] # target is the last column
            # seq_y = seq_y.reshape(-1, 1) # reshape to (pred_len, 1)
            
            return seq_x, seq_y
        
        else:
            # input seq index
            s_begin = index
            s_end = index + self.seq_len
            
            # pred seq index
            r_begin = s_end
            r_end = r_begin + self.pred_len
            
            # define input seq and target seq
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end][:, -1] # target is the last column
            # seq_y = seq_y.reshape(-1, 1) # reshape to (pred_len, 1)
            return seq_x, seq_y
        
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        # return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        """
        need to make sure __len__ is the right length
        """
        if self.time_predence == 'D':
            return len(self.data_x) - self.seq_len - (self.pred_len * 2) + 1
        elif self.time_predence == 'F':
            return len(self.data_x) - self.seq_len - self.pred_len - self.fixed_time_predence + 1
        else:
            return len(self.data_x) - self.seq_len - self.pred_len + 1

    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)