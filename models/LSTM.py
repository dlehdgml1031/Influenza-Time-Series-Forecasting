import os
import sys
import random
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from data_loader import Dataset_ILI_National, NaiveILINationalDataset

from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


class ILILSTM(nn.Module):
    def __init__(self, pred_type:str, input_dim:int, hidden_dim:int, num_layers:int,
                 pred_len:Optional[int] = 1,
                 lstm_batch_first:Optional[bool] = True):
        super(ILILSTM, self).__init__()
        self.pred_type = pred_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if pred_type == 'pred':
            self.pred_len = pred_len
        
        self.lstm_ILI = nn.LSTM(
            input_size = self.input_dim,
            hidden_size = self.hidden_dim,
            num_layers = self.num_layers,
            batch_first = lstm_batch_first
        )
        
        self.fc_forecast = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm_ILI(x, (h0, c0))
        
        # Only take the output from the last time step
        lstm_out_last = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.fc_forecast(lstm_out_last)
        
        # Apply Sigmoid activation function
        output = self.sigmoid(output)
        
        return output
    
def train_model(pred_type:str, seq_len:int, train_flag:str, batch_size:int, num_epochs:int, learning_rate:float, num_layers:int, hidden_dim:int,
                ili_input_size:int = 1):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = NaiveILINationalDataset(pred_type = pred_type, seq_len = seq_len, train_flag = train_flag)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    model = ILILSTM(pred_type = pred_type, input_dim = ili_input_size, hidden_dim = hidden_dim, num_layers = num_layers)
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        model.train()
        
        for batch_idx, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            seq, labels = batch
            seq = seq.to(device)
            seq = seq.unsqueeze(-1)
            labels = labels.to(device)
            
            output = model(seq)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    print('Training finished.')
    
    return model

def evaluate_model(model: nn.Module, pred_type: str, seq_len: int, batch_size: int,
                   ili_input_size: int = 1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the dataset (Assuming test_flag or a similar flag to indicate evaluation data)
    dataset = NaiveILINationalDataset(pred_type = pred_type, seq_len = seq_len, train_flag = 'test')
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)

    model.to(device)
    model.eval()  # Set model to evaluation mode
    
    all_labels = []
    all_preds = []

    with torch.no_grad():  # Disable gradient calculations for evaluation
        for batch_idx, batch in enumerate(dataloader):
            seq, labels = batch
            seq = seq.to(device)
            seq = seq.unsqueeze(-1)
            labels = labels.to(device)
            
            output = model(seq)
            
            # Binarize the output with a threshold (e.g., 0.5)
            preds = (output >= 0.5).float()
            
            # Collect all true labels and predictions for evaluation
            all_labels.append(labels.cpu())
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

        

# define LSTM model
class ILI_LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ILI_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def plot_loss(
    loss_values: List[float],
    epochs: int,
):
    plt.figure(figsize=(10, 6))
    sns.lineplot(x = range(1, epochs + 1), y = loss_values)
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    

def train_LSTM(
    input_dim:int,
    hidden_dim:int,
    num_layers:int,
    output_dim:int,
    data_path:str,
    seq_len:int,
    pred_len:int,
    time_predence:str = 'D',
    fixed_time_predence:int = 3,
    batch_size:int = 32,
    epochs: int = 30,
    learning_rate:float = 1e-3,
    scale:bool = True,
):
    # fix random seed
    torch.manual_seed(0)
    
    # define dataset and dataloader
    train_dataset = Dataset_ILI_National(data_path, seq_len, pred_len, scale, time_predence = time_predence, fixed_time_predence = fixed_time_predence, flag='train')
    test_dataset = Dataset_ILI_National(data_path, seq_len, pred_len, scale, time_predence = time_predence, fixed_time_predence = fixed_time_predence, flag='test')

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"current device: {device}")
    
    # input_dim = num of features
    model = ILI_LSTM(input_dim = input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim).to(device)
    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    loss_values = [] # for plotting
    
    # train model
    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            seq = seq.float().to(device)
            labels = labels.float().to(device)
            preds = model(seq)
            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        loss_values.append(loss.item())
    
    # test model
    model.eval()
    predictions, actuals = [], []
    for seq, labels in test_loader:
        with torch.no_grad():
            seq = seq.float().to(device)  # convert input data to Float type
            labels = labels.float().to(device)  # convert target data to Float type
            outputs = model(seq)
            predictions.extend(outputs.cpu().numpy())  # moving from GPU to CPU and converting to a NumPy array
            actuals.extend(labels.cpu().numpy())  # moving from GPU to CPU and converting to a NumPy array
    
    actuals = np.array(actuals).reshape(1,-1)
    predictions = np.array(predictions).reshape(1,-1)
    
    # actuals = scaler.inverse_transform(actuals).flatten()
    # predictions = scaler.inverse_transform(predictions).flatten()
    
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mse)
    
    print('#' * 150)
    print(f'Model: LSTM / Sequence Len: {seq_len} / Prediction Len: {pred_len} / Time Predence: {time_predence} / Fixed Time predence: {fixed_time_predence} / Num of Features(Input dim): {input_dim} / Hidden dim: {hidden_dim} / Num layers: {num_layers} / Output dim: {output_dim}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')
    print('#' * 150)
    
    metric_dict = {
        'MSE' : mse,
        'RMSE' : rmse,
        'MAE' : mae,
    }
        
    return model, actuals, predictions, metric_dict

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "8"
    
    pred_type = 'cls'
    seq_len = 50
    batch_size = 16
    num_epochs = 50
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