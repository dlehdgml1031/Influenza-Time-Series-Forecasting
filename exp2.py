from exp.exp_basic import Exp_Basic

import os
import sys
import random
from typing import Optional, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


class FluCastDataset(Dataset):
    def __init__(self, pred_type:str, seq_len:int, train_flag:str,
                 pred_len:Optional[int] = 1,
                 numeric_multi_var_flag:Optional[bool] = True,
                 scale:Optional[bool] = False,
            ):
        type_map = {'train': 0, 'test': 1}
        self.pred_type = pred_type
        self.seq_len = seq_len
        self.train_flag = train_flag
        self.numeric_multi_var_flag = numeric_multi_var_flag
        self.scale = scale
        self.ili_path = './data/ILI/national_illness_2018_v3.csv'
        self.set_type = type_map[train_flag]
                    
        if self.pred_type == 'cls':
            self.pred_len = 1
        elif self.pred_type == 'pred':
            self.pred_len = pred_len
        
        self.__read_data__()
    
    def __read_data__(self):
        if self.numeric_multi_var_flag:
            self.ili_features = pd.read_csv(self.ili_path, index_col = [0, 1]).drop(columns=['Label', 'Label 5', 'Label 7']).values
        else:
            self.ili_features = pd.read_csv(self.ili_path, index_col = [0, 1])[['TOTAL PATIENTS']].values

        # Determine labels based on prediction type
        if self.pred_type == 'cls':
            # Classification task
            labels = pd.read_csv(self.ili_path, index_col = [0, 1])['Label'].tolist()
            self.labels = [1 if label == 'UP' else 0 for label in labels]
        elif self.pred_type == 'pred':
            # Prediction task
            self.labels = pd.read_csv(self.ili_path, index_col = [0, 1])['TOTAL PATIENTS'].tolist()
        
        # train, test split (7:3)
        num_train = int(len(self.labels) * 0.7)
        num_test = int(len(self.labels) * 0.3)
        
        border1s = [0, len(self.labels) - num_test - self.seq_len - self.pred_len]
        border2s = [num_train + self.pred_len, len(self.labels)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Scale features if required
        if self.scale:
            self.scaler = MinMaxScaler()
            # Fit the scaler on the training portion of the data
            self.scaler.fit(self.ili_features[border1s[0]:border2s[0]])
            # Transform the entire feature set
            self.ili_features = self.scaler.transform(self.ili_features)
            
            # If using single-variable features, flatten back to a 1D array
            if not self.numeric_multi_var_flag:
                self.ili_features = self.ili_features.flatten()
            
            if self.pred_type == 'pred':
                self.labels = np.array(self.labels).reshape(-1, 1)
                self.label_scaler = MinMaxScaler()
                self.label_scaler.fit(self.labels[border1s[0]:border2s[0]])
                self.labels = self.label_scaler.transform(self.labels)
                self.labels = self.labels.flatten()
        
        self.ili_features = self.ili_features[border1:border2]
        self.labels = self.labels[border1:border2]
        
    def __len__(self):
        return len(self.labels) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, index):
        s_begin = index
        s_end = index + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        ili_features = torch.tensor(self.ili_features[s_begin:s_end], dtype = torch.float32)
        label = torch.tensor(self.labels[r_begin:r_end], dtype = torch.float32)
        
        return ili_features, label


class FluCastILIV1(nn.Module):
    def __init__(self, pred_type: str, num_layers:int, hidden_dim:int, numeric_multi_var_flag:bool,
                 pred_len:Optional[int] = 1,
                 lstm_batch_first: Optional[bool] = True):
        super(FluCastILIV1, self).__init__()
        self.pred_type = pred_type
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        if numeric_multi_var_flag:
            self.ili_input_size = 5
        else:
            self.ili_input_size = 1
        
        if pred_type == 'pred':
            self.pred_len = pred_len
        elif pred_type == 'cls':
            self.pred_len = 1
        
        # ILI 데이터를 위한 LSTM
        self.lstm = nn.LSTM(
            input_size=self.ili_input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=lstm_batch_first
        )
        
        # 예측 레이어 정의
        self.fc_forecast = nn.Linear(self.hidden_dim, self.pred_len)
        
        # 이진 분류 작업을 위한 시그모이드 함수
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, ili_input):
        # LSTM에 입력
        lstm_out, (hn, cn) = self.lstm(ili_input)
        hn = hn[-1]  # shape: (batch_size, hidden_dim)
        
        # 예측 레이어 적용
        forecast_output = self.fc_forecast(hn)
        
        if self.pred_type == 'cls':
            output = self.sigmoid(forecast_output)
        else:
            output = forecast_output
        
        return output


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self._get_data()
        
        if args.pred_type == 'cls':
            self.criterion = nn.BCELoss()
            
        else:
            self.criterion = nn.MSELoss()
        
    def _build_model(self):
        model = FluCastILIV1(
            pred_type=self.args.pred_type,
            num_layers=self.args.num_layers,
            hidden_dim=self.args.hidden_dim,
            numeric_multi_var_flag=self.args.numeric_multi_var_flag,
        )
        
        return model

    def _get_data(self):
        self.train_dataset = FluCastDataset(
            pred_type=self.args.pred_type,
            seq_len=self.args.seq_len,
            train_flag='train',
            numeric_multi_var_flag=self.args.numeric_multi_var_flag,
            scale=self.args.scale
        )
        self.test_dataset = FluCastDataset(
            pred_type=self.args.pred_type,
            seq_len=self.args.seq_len,
            train_flag='test',
            numeric_multi_var_flag=self.args.numeric_multi_var_flag,
            scale=self.args.scale
        )
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle = False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle = False)

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.args.learning_rate)
        for epoch in range(self.args.num_epochs):
            self.model.train()
            total_loss = 0
            for batch in self.train_loader:
                optimizer.zero_grad()
                ili_features, labels = batch
                ili_features = ili_features.to(self.device)
                labels = labels.to(self.device)
                                
                # 단변량 특징인 경우 차원 확장
                if not self.args.numeric_multi_var_flag:
                    ili_features = ili_features.unsqueeze(-1)  # (batch_size, seq_len, 1)

                outputs = self.model(ili_features)
                
                loss = self.criterion(outputs.squeeze(), labels.float().squeeze())
                    
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{self.args.num_epochs}], Loss: {avg_loss:.4f}")

    def test(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for batch in self.test_loader:
                ili_features, labels = batch
                ili_features = ili_features.to(self.device)
                labels = labels.to(self.device)

                # 단변량 특징인 경우 차원 확장
                if not self.args.numeric_multi_var_flag:
                    ili_features = ili_features.unsqueeze(-1)  # (batch_size, seq_len, 1)

                outputs = self.model(ili_features)
                                
                preds = (outputs >= 0.5).float()
                labels = labels.float()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='binary')
        recall = recall_score(all_labels, all_preds, average='binary')
        f1 = f1_score(all_labels, all_preds, average='binary')
    
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1 Score: {f1:.4f}")
        
        return accuracy, precision, recall, f1
        
if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_type', type=str, default = 'cls')
    parser.add_argument('--seq_len', type=int, default = 2)
    parser.add_argument('--batch_size', type=int, default = 32)
    parser.add_argument('--num_epochs', type=int, default = 500)
    parser.add_argument('--learning_rate', type=float, default = 1e-3)
    parser.add_argument('--num_layers', type=int, default = 1)
    parser.add_argument('--hidden_dim', type=int, default = 128)
    parser.add_argument('--numeric_multi_var_flag', type = bool, default = False)
    parser.add_argument('--scale', type=bool, default = False)
    args = parser.parse_args()
    
    # Set random seed
    def seed_everything(seed):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    total_res_dict = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for seed in range(5):
        seed_everything(seed)

        exp = Exp_Classification(args)
        exp.train()
        acc, prec, rec, f1 = exp.test()
        total_res_dict['accuracy'].append(acc)
        total_res_dict['precision'].append(prec)
        total_res_dict['recall'].append(rec)
        total_res_dict['f1'].append(f1)
        
    print('======= Total Results =======')
    print(f"평균 Accuracy: {np.mean(total_res_dict['accuracy']):.4f}")
    print(f"평균 Precision: {np.mean(total_res_dict['precision']):.4f}")
    print(f"평균 Recall: {np.mean(total_res_dict['recall']):.4f}")
    print(f"평균 F1 Score: {np.mean(total_res_dict['f1']):.4f}")
