import os
import sys

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.time_series_analysis import dymanic_time_series_lag_analysis

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


if __name__ == '__main__':
    # test
    data_path = 'data/ILI/national_illness.csv'
    seq_len = 20
    pred_len = 3
    scale = True
    target = 'TOTAL PATIENTS'
        
    dataset = Dataset_ILI_National(
        data_path = data_path,
        seq_len = seq_len,
        pred_len = pred_len,
        scale = scale,
        target = target,
        flag = 'train'
    )
    
    data_loader = DataLoader(dataset, batch_size = 2, shuffle = False)
    print(len(data_loader))

    # Iterate through the data loader
    for batch_x, batch_y in data_loader:
        print(batch_x.shape, batch_y.shape)
        print(batch_x, batch_y)