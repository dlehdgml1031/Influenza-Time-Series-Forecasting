from data_loader import Dataset_ILI_National

from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

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

