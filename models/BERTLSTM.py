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

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seed
def seed_everything(seed):
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
        self.fc_forecast = nn.Linear(self.hidden_dim * 4, 1)
        
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
        
        # Concatenate weighted sum with ili hidden state for forecasting
        forecast_input = torch.cat((weighted_hn_news, weighted_hn_patent, weighted_hn_abstract, hn_ili), dim=1)  # shape: (batch_size, hidden_dim * 4)
        
        # Apply the forecasting layer
        forecast_output = self.fc_forecast(forecast_input)  # shape: (batch_size, 1)
        
        # Apply sigmoid to get the final prediction probabilities
        output = self.sigmoid(forecast_output)
        
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
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            news_embeddings, abstract_embeddings, patent_embeddings, total_patients, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            total_patients = total_patients.to(device)
            label = label.to(device)
            total_patients = total_patients.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, total_patients)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
            
    print('Training finished.')

    return model


def evaluate_model(model: nn.Module, pred_type: str, seq_len: int, batch_size: int, 
                   text_input_size: int = 768, ili_input_size: int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset (Assuming test_flag or a similar flag to indicate evaluation data)
    dataset = FluCastDataset(pred_type = pred_type, seq_len = seq_len, train_flag = 'test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for batch_idx, batch in enumerate(dataloader):
            news_embeddings, abstract_embeddings, patent_embeddings, total_patients, label = batch
            news_embeddings = news_embeddings.to(device)
            abstract_embeddings = abstract_embeddings.to(device)
            patent_embeddings = patent_embeddings.to(device)
            total_patients = total_patients.to(device)
            label = label.to(device)
            
            total_patients = total_patients.unsqueeze(-1)  # (batch_size, seq_len, 1)
            
            # Get model predictions
            output = model(news_embeddings, patent_embeddings, abstract_embeddings, total_patients)
            
            # Binarize the output with a threshold (e.g., 0.5)
            preds = (output >= 0.5).float()
            
            # Collect all true labels and predictions for evaluation
            all_labels.append(label.cpu())
            all_preds.append(preds.cpu())

    # Concatenate all batches for evaluation
    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')

    return accuracy, precision, recall, f1

            
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"
    
    pred_type = 'cls'
    seq_len = 5
    batch_size = 16
    num_epochs = 30
    learning_rate = 1e-3
    num_layers = 1
    hidden_dim = 512
    
    total_res_dict = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    for seed in range(5):
        seed_everything(seed)
    
        # Training model
        model = train_model(
            pred_type = pred_type,
            seq_len = seq_len,
            train_flag = 'train',
            batch_size = batch_size,
            num_epochs = num_epochs,
            learning_rate = learning_rate,
            num_layers = num_layers,
            hidden_dim = hidden_dim
        )
        
        # Evaluate the model on test data
        acc, pre, recall, f1 = evaluate_model(model = model, pred_type = pred_type, seq_len = seq_len, batch_size = batch_size)
        
        total_res_dict['accuracy'].append(acc)
        total_res_dict['precision'].append(pre)
        total_res_dict['recall'].append(recall)
        total_res_dict['f1'].append(f1)
    
    total_res_dict['accuracy'] = np.mean(total_res_dict['accuracy'])
    total_res_dict['precision'] = np.mean(total_res_dict['precision'])
    total_res_dict['recall'] = np.mean(total_res_dict['recall'])
    total_res_dict['f1'] = np.mean(total_res_dict['f1'])
    
    print(total_res_dict)