import os
import sys
import warnings

import torch
import numpy as np
from scipy.signal import correlate

def time_series_lag_analysis(
    feature_data:torch.Tensor,
    target_data:torch.Tensor,
):
    """
    Calculate the lag that each feature has on the target by cross-correlation.

    Parameters
    ----------
    seq_x : torch.Tensor 
        _description_
    seq_y : torch.Tensor
        _description_
    max_lag : int
        _description_
        
    Returns
    -------
    
    """
    batch_size, seq_len , n_features = feature_data.size()
    _, pred_len, _ = target_data.size()
    
    max_lag = pred_len
    
    feature_lags = torch.zeros(n_features)
    
    # Loop over each feature
    for feature_idx in range(n_features):
        cross_corrs = np.zeros(max_lag)
        
        # Calculate cross-correlation for each lag
        for lag in range(1, max_lag + 1):
            corr_sum = 0
            
            for batch_idx in range(batch_size):
                feature_seq = feature_data[batch_idx, :, feature_idx].numpy()
                target_seq = target_data[batch_idx, :, 0].numpy()
                
                print(feature_seq)
                print(target_seq)
                return None
            

if __name__ == '__main__':
    pass