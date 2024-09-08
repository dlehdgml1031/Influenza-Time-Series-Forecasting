import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import FluCastDataset

from typing import Optional

import random
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer

# Set random seed
SEED = 0

def seed_everything(seed = SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


class FluCastText(nn.Module):
    def __init__(self, pred_type: str, text_input_size:int, ili_input_size:int, num_layers:int, hidden_dim:int,
                 pred_len:Optional[int] = 1,
                 lstm_batch_first: Optional[bool] = True):
        super(FluCastText, self).__init__()
        self.pred_type = pred_type
        self.text_input_size = text_input_size
        self.ili_input_size = ili_input_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        if pred_type == 'fs':
            self.pred_len = pred_len
        
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
            batch_first = True
        )
        
        self.lstm_ILI = nn.LSTM(
            input_size = ili_input_size,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first
        )
        
        # Define VariableSelectionNetwork
        self.fc_variable_selection = nn.Linear(self.hidden_dim * 3, 3)
        self.variable_weight = nn.Softmax(dim = 1) # dim check
        
        # Define Time-series forecasting layer
        self.fc_forecast = nn.Linear(self.hidden_dim * 4, 2)
        
        # Softmax layer
        # self.softmax = nn.Softmax(dim = 1) # dim check
        
        # Sigmoid Layer
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, news_input: torch.Tensor, patent_input: torch.Tensor, abstract_input: torch.Tensor, ili_input: torch.Tensor):
        # Pass each input through its respective LSTM
        _, (hn_news, _) = self.lstm_news(news_input)
        _, (hn_patent, _) = self.lstm_patent(patent_input)
        _, (hn_abstract, _) = self.lstm_abstract(abstract_input)
        _, (hn_ili, _) = self.lstm_ILI(ili_input)
        
        # Extract the hidden state of the last LSTM layer for each input
        hn_news = hn_news[-1]  # shape: (batch_size, hidden_dim)
        hn_patent = hn_patent[-1]  # shape: (batch_size, hidden_dim)
        hn_abstract = hn_abstract[-1]  # shape: (batch_size, hidden_dim)
        hn_ili = hn_ili[-1]  # shape: (batch_size, hidden_dim)
        
        # Concatenate the hidden states from all sources
        concat_hidden_states = torch.cat((hn_news, hn_patent, hn_abstract), dim=1)  # shape: (batch_size, hidden_dim * 3)
        
        # Apply the variable selection network (linear layer + softmax)
        variable_weights = self.variable_weight(self.fc_variable_selection(concat_hidden_states))  # shape: (batch_size, 3)
        
        weighted_hn_news = hn_news * variable_weights[:, 0].unsqueeze(-1)        # shape: (batch_size, hidden_dim)
        weighted_hn_patent = hn_patent * variable_weights[:, 1].unsqueeze(-1)    # shape: (batch_size, hidden_dim)
        weighted_hn_abstract = hn_abstract * variable_weights[:, 2].unsqueeze(-1) # shape: (batch_size, hidden_dim)
        
        # # Apply the variable weights to each hidden state
        # weighted_hidden_states = torch.stack([hn_news, hn_patent, hn_abstract], dim = 1)  # shape: (batch_size, 3, hidden_dim)
        # weighted_sum = torch.sum(weighted_hidden_states * variable_weights.unsqueeze(-1), dim = 1)  # shape: (batch_size, hidden_dim)
        
        # Concatenate weighted sum with ili hidden state for forecasting
        forecast_input = torch.cat((weighted_hn_news, weighted_hn_patent, weighted_hn_abstract, hn_ili), dim=1)  # shape: (batch_size, hidden_dim * 4)
        
        # Apply the forecasting layer
        forecast_output = self.fc_forecast(forecast_input)  # shape: (batch_size, 2)
        
        # Apply softmax to get the final prediction probabilities
        # output = self.softmax(forecast_output)  # shape: (batch_size, 2)
        output = self.sigmoid(forecast_output) # 시그모이드 함수 적용 Temp
        
        return output

def train_model(pred_type:str, seq_len:int, train_flag:str, batch_size:int, num_epochs:int, learning_rate:float, num_layers:int, hidden_dim:int,
                text_input_size:int = 768,
                ili_input_size:int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = FluCastDataset(pred_type = pred_type, seq_len = seq_len, train_flag = train_flag)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    model = FluCastText(pred_type = pred_type, text_input_size = text_input_size, ili_input_size = ili_input_size, num_layers = num_layers, hidden_dim = hidden_dim)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            news_embeddings, abstract_embeddings, patent_embeddings, total_patients, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            total_patients = total_patients.to(device)
            label = label.to(device)
            label = label.squeeze(dim=1)  # (batch_size,) 형태로 변환

            total_patients = total_patients.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            optimizer.zero_grad()
            
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, total_patients)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            
    print('Training finished.')

    return model

def evaluate_model(model, dataloader):
    model.eval()  # 평가 모드로 전환 (Dropout, BatchNorm 등의 레이어가 평가 모드로 동작)
    criterion = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 평가 시에는 기울기 계산을 하지 않음
        for batch_idx, batch in enumerate(dataloader):
            news_embeddings, abstract_embeddings, patent_embeddings, total_patients, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            total_patients = total_patients.to(device)
            label = label.to(device)
            label = label.squeeze(dim=1)  # (batch_size,) 형태로 변환

            total_patients = total_patients.unsqueeze(-1)  # (batch_size, seq_len, 1)

            # 모델의 예측값을 얻음
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, total_patients)

            # 손실(loss) 계산
            loss = criterion(output, label)
            total_loss += loss.item()

            # 예측값과 실제 레이블 비교
            _, predicted = torch.max(output, 1)
            correct += (predicted == label).sum().item()
            total += label.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"
    
    # Training model
    model = train_model(
        pred_type = 'cls',
        seq_len = 100,
        train_flag = 'train',
        batch_size = 32,
        num_epochs = 100,
        learning_rate = 1e-3,
        num_layers = 1,
        hidden_dim = 512
    )
    
    # Load test data
    test_dataset = FluCastDataset(pred_type='cls', seq_len = 5, train_flag='test')
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Evaluate the model on test data
    evaluate_model(model, test_dataloader)