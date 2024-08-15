from data_loader import Dataset_ILI_National
from models.LSTM import train_LSTM

from typing import Tuple, List, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


data_path = 'data/ILI_v1.csv'

input_dim = 2
hidden_dim = 100
num_layers = 2
pred_len = 3
output_dim = pred_len
seq_len = 15
time_predence = 'D'
fixed_time_predence = 3
batch_size = 32
epochs = 50
learning_rate = 1e-3
scale = True

_ , actuals, predictions, metric_dict= train_LSTM(
    input_dim = input_dim,
    hidden_dim = hidden_dim,
    num_layers = num_layers,
    output_dim = output_dim,
    data_path = data_path,
    seq_len = seq_len,
    pred_len = pred_len,
    time_predence = time_predence,
    fixed_time_predence = fixed_time_predence,
    batch_size = batch_size,
    epochs = epochs,
    learning_rate = learning_rate,
    scale = scale
)

pd.DataFrame({'Ground Truth': actuals[0], 'Predictions': predictions[0]}).plot(kind = 'line', figsize = (10, 6))
plt.show()