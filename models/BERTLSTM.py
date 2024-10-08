import os
import sys
from typing import Optional
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import FluCastDataset

import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

class FluCastText(nn.Module):
    def __init__(self, pred_type: str, text_input_size:int, num_layers:int, hidden_dim:int, numeric_multi_var_flag:bool,
                 pred_len:Optional[int] = 1,
                 lstm_batch_first: Optional[bool] = True):
        super(FluCastText, self).__init__()
        self.pred_type = pred_type
        self.text_input_size = text_input_size
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
        elif pred_type == 'cls5':
            self.pred_len = 5
        elif pred_type == 'cls7':
            self.pred_len = 7
        
        # Define LSTM Layer
        self.lstm_news = nn.LSTM(
            input_size = self.text_input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first,
        )
        
        self.lstm_abstract = nn.LSTM(
            input_size = self.text_input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first
        )
        
        self.lstm_patent = nn.LSTM(
            input_size = self.text_input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first
        )
        
        self.lstm_ILI = nn.LSTM(
            input_size = self.ili_input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first
        )
        
        # Define VariableSelectionNetwork
        # self.fc_variable_selection = nn.Linear(self.hidden_dim * 3, 3)
        # self.variable_weight = nn.Softmax(dim = 1) # dim check
        
        # Define Forecasting Layer
        self.fc_forecast = nn.Linear(self.hidden_dim * 4, self.pred_len)
        
        # Sigmoid for binary classification tasks
        self.sigmoid = nn.Sigmoid()
        
        # Softmax for classification tasks
        self.softmax = nn.Softmax(dim = 1)
        

    def forward(self, news_input: torch.Tensor, patent_input: torch.Tensor, abstract_input: torch.Tensor, ili_input: torch.Tensor):
        # Pass each input through its respective LSTM
        _, (hn_news, _) = self.lstm_news(news_input)
        _, (hn_patent, _) = self.lstm_patent(patent_input)
        _, (hn_abstract, _) = self.lstm_abstract(abstract_input)
        _, (hn_ili, _) = self.lstm_ILI(ili_input)
        
        # Extract the hidden state of the last LSTM layer for each input
        hn_news = hn_news[-1]  # shape: (batch_size, hidden_dim)
        hn_patent = hn_patent[-1]
        hn_abstract = hn_abstract[-1]
        hn_ili = hn_ili[-1]
        
        # Concatenate the hidden states from all sources
        concat_hidden_states = torch.cat((hn_news, hn_patent, hn_abstract, hn_ili), dim=1)  # shape: (batch_size, hidden_dim * 4)
        
        # Apply the forecasting layer
        forecast_output = self.fc_forecast(concat_hidden_states)  # shape: (batch_size, pred_len)
        
        if self.pred_type == 'cls':
            # Apply sigmoid to get the final prediction probabilities
            output = self.sigmoid(forecast_output)
        elif self.pred_type == 'cls5':
            output = self.softmax(forecast_output)
        elif self.pred_type == 'cls7':
            output = self.softmax(forecast_output)
        else:
            # For prediction tasks, do not apply activation function
            output = forecast_output
        
        return output


def train_model(pred_type: str, seq_len: int, train_flag: str, batch_size: int, num_epochs: int, 
                learning_rate: float, num_layers: int, hidden_dim: int, bert_model_id: str,
                text_input_size: int = 768,
                numeric_multi_var_flag: bool = True,
                pred_len: int = 1,
                scale: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 초기화 시 필요한 매개변수 전달
    dataset = FluCastDataset(pred_type = pred_type, seq_len = seq_len, train_flag = train_flag, model_id = bert_model_id,
                             numeric_multi_var_flag = numeric_multi_var_flag, pred_len = pred_len, scale = scale)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=False)
    
    # 모델 초기화 시 필요한 매개변수 전달
    model = FluCastText(pred_type = pred_type, text_input_size = text_input_size, num_layers = num_layers, hidden_dim = hidden_dim,
                        numeric_multi_var_flag = numeric_multi_var_flag, pred_len = pred_len)
    model.to(device)
    
    # 예측 유형에 따라 손실 함수 설정
    if pred_type == 'cls':
        criterion = nn.BCELoss()
    elif pred_type == 'cls5':
        criterion = nn.CrossEntropyLoss()
    elif pred_type == 'cls7':
        criterion = nn.CrossEntropyLoss()
    elif pred_type == 'pred':
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            news_embeddings, abstract_embeddings, patent_embeddings, ili_features, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            ili_features = ili_features.to(device)
            label = label.to(device)
            
            # 단변량 특징인 경우 차원 확장
            if not numeric_multi_var_flag:
                ili_features = ili_features.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, ili_features)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
                
    print('Training finished.')

    return model


def evaluate_model(model: nn.Module, pred_type: str, seq_len: int, batch_size: int, bert_model_id: str,
                   text_input_size: int = 768,
                   numeric_multi_var_flag: bool = True,
                   pred_len: int = 1,
                   scale: bool = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 평가용 데이터셋 로드
    dataset = FluCastDataset(pred_type = pred_type, seq_len = seq_len, train_flag='test', model_id = bert_model_id,
                             numeric_multi_var_flag = numeric_multi_var_flag, pred_len = pred_len, scale = scale)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle = False)

    model.to(device)
    model.eval()  # 모델을 평가 모드로 설정
    
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            news_embeddings, abstract_embeddings, patent_embeddings, ili_features, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            ili_features = ili_features.to(device)
            label = label.to(device)
            
            # 단변량 특징인 경우 차원 확장
            if not numeric_multi_var_flag:
                ili_features = ili_features.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            # 모델 예측
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, ili_features)
            
            if pred_type == 'cls':
                # 출력 이진화 (임계값 0.5 사용)
                preds = (output >= 0.5).float()
                # 레이블과 예측값 수집
                all_labels.append(label.cpu())
                all_preds.append(preds.cpu())
            elif pred_type == 'pred':
                # 회귀 작업의 경우 출력과 레이블 그대로 수집
                all_labels.append(label.cpu())
                all_preds.append(output.cpu())

    # 배치별 결과를 하나로 합침
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    if pred_type == 'cls':
        # 분류 작업 평가 지표 계산
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        
        return accuracy, precision, recall, f1
    
    elif pred_type == 'pred':
        # 회귀 작업 평가 지표 계산
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        r2 = r2_score(all_labels, all_preds)
        
        print(f'Mean Squared Error: {mse:.4f}')
        print(f'Mean Absolute Error: {mae:.4f}')
        print(f'R^2 Score: {r2:.4f}')
        
        return mse, mae, r2

            
if __name__ == '__main__':
    import os

    # 사용할 GPU 장치 설정 (필요에 따라 수정)
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"

    # 공통 설정
    seq_len = 3
    bert_model_id = 'BERT'
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-3
    num_layers = 1
    hidden_dim = 512
    numeric_multi_var_flag = True  # 다변량 수치 특징 사용 여부
    pred_len = 1  # 예측 작업의 경우 예측 길이 설정
    scale = False  # 특징 스케일링 여부

    # 분류와 예측 작업에 대한 설정
    pred_types = ['cls', 'pred']  # ['cls', 'pred']로 설정하여 분류와 예측 모두 실행

    for pred_type in pred_types:
        print(f"\n=== {pred_type.upper()} 작업 실행 ===")
        if pred_type == 'cls':
            total_res_dict = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1': []
            }
        elif pred_type == 'pred':
            total_res_dict = {
                'mse': [],
                'mae': [],
                'r2': []
            }

        for seed in range(5):
            seed_everything(seed)
            
            # 모델 학습
            model = train_model(
                pred_type=pred_type,
                seq_len=seq_len,
                train_flag='train',
                batch_size=batch_size,
                num_epochs=num_epochs,
                learning_rate=learning_rate,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                bert_model_id=bert_model_id,
                numeric_multi_var_flag=numeric_multi_var_flag,
                pred_len=pred_len,
                scale=scale
            )
            
            # 테스트 데이터로 모델 평가
            if pred_type == 'cls':
                acc, pre, recall, f1 = evaluate_model(
                    model=model,
                    pred_type=pred_type,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    bert_model_id=bert_model_id,
                    numeric_multi_var_flag=numeric_multi_var_flag,
                    pred_len=pred_len,
                    scale=scale
                )
                total_res_dict['accuracy'].append(acc)
                total_res_dict['precision'].append(pre)
                total_res_dict['recall'].append(recall)
                total_res_dict['f1'].append(f1)
            elif pred_type == 'pred':
                mse, mae, r2 = evaluate_model(
                    model=model,
                    pred_type=pred_type,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    bert_model_id=bert_model_id,
                    numeric_multi_var_flag=numeric_multi_var_flag,
                    pred_len=pred_len,
                    scale=scale
                )
                total_res_dict['mse'].append(mse)
                total_res_dict['mae'].append(mae)
                total_res_dict['r2'].append(r2)
        
        # 결과의 평균 계산 및 출력
        if pred_type == 'cls':
            total_res_dict['accuracy'] = np.mean(total_res_dict['accuracy'])
            total_res_dict['precision'] = np.mean(total_res_dict['precision'])
            total_res_dict['recall'] = np.mean(total_res_dict['recall'])
            total_res_dict['f1'] = np.mean(total_res_dict['f1'])
            print(f"분류 작업 평균 결과:")
            print(f"Accuracy: {total_res_dict['accuracy']:.4f}")
            print(f"Precision: {total_res_dict['precision']:.4f}")
            print(f"Recall: {total_res_dict['recall']:.4f}")
            print(f"F1 Score: {total_res_dict['f1']:.4f}")
        elif pred_type == 'pred':
            total_res_dict['mse'] = np.mean(total_res_dict['mse'])
            total_res_dict['mae'] = np.mean(total_res_dict['mae'])
            total_res_dict['r2'] = np.mean(total_res_dict['r2'])
            print(f"예측 작업 평균 결과:")
            print(f"Mean Squared Error: {total_res_dict['mse']:.4f}")
            print(f"Mean Absolute Error: {total_res_dict['mae']:.4f}")
            print(f"R^2 Score: {total_res_dict['r2']:.4f}")