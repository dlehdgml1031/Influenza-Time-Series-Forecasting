import os
import sys

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# from utils.timefeatures import time_features

from utils.time_series_analysis import time_series_lag_analysis

class Dataset_ILI_National(Dataset):
    def __init__(self,
                 data_path:str,
                 seq_len:int,
                 pred_len:int,
                 scale:bool = True,
                 flag:str = 'train',
                 target:str = 'TOTAL PATIENTS'):
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.target = target
        self.__read_data__()
    
    def __read_data__(self):
        # read csv file
        df_data = pd.read_csv(self.data_path, index_col = [0,1])
        cols = df_data.columns.tolist()
        cols.remove(self.target)
        df_data = df_data[cols + [self.target]]
        
        # train, test, vlidation split (7:2:1)
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[border1s[0]:border2s[0]]
            self.sacler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        # self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        # input seq index
        s_begin = index
        s_end = index + self.seq_len
        
        # input token seq index
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # define input seq and target seq
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end][:, -1] # target is the last column
        seq_y = seq_y.reshape(-1, 1) # reshape to (pred_len, 1)
        
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[r_begin:r_end]
        
        # return seq_x, seq_y, seq_x_mark, seq_y_mark
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == '__main__':
    # test
    data_path = 'data/national_illness.csv'
    seq_len = 20
    pred_len = 3
    scale = False
    target = 'TOTAL PATIENTS'
        
    dataset = Dataset_ILI_National(
        data_path = data_path,
        seq_len = seq_len,
        pred_len = pred_len,
        scale = scale,
        target = target,
    )
    
    data_loader = DataLoader(dataset, batch_size = 2, shuffle = False)

    # Iterate through the data loader
    for batch_x, batch_y in data_loader:
        print(batch_x.shape, batch_y.shape)
        print(batch_x, batch_y)
        break