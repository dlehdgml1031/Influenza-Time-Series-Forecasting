from .timefeatures import date_to_year_week

import os
import sys
import warnings

import torch
import numpy as np
import pandas as pd
import statsmodels.api as sm

# To be fixed
def dymanic_time_series_lag_analysis(
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
    
    assert int(max_lag) >= 3, 'max_lag (pred_len) should be greater than or equal to 3'
    
    feature_lags = torch.zeros(n_features)
    
    # batch_size -> feature -> lag
    # caluclate cross-correlation
    
    # Loop over each batch
    for batch_idx in range(batch_size):
        
        # Loop over each feature
        for feature_idx in range(n_features):
            cross_corrs = np.zeros(max_lag)
            
            # Loop over each lag
            for lag in range(1, max_lag + 1):
                corr_sum = 0
                
                feature_seq = feature_data[batch_idx, :, feature_idx].numpy()
                target_seq = target_data[batch_idx, :, 0].numpy()
                target_seq = target_seq[lag:]
                
                feature_seq_mean = np.mean(feature_seq)
                target_seq_mean = np.mean(target_seq)
                
                numerator = np.sum()
                denominator = np.sum() * np.sum()
                
                corr = numerator / denominator


def calculate_time_series_lag(
    feature_data_path:str,
    target_data_path:str,
    feature_col_name:str,
    target_col_name:str,
    max_lags:int,
):
    if 'GT_influenza.csv' in feature_data_path:
        feature_df = pd.read_csv(feature_data_path)
    else:
        feature_df = pd.read_csv(feature_data_path, index_col = 0)
    target_df = pd.read_csv(target_data_path, index_col = [0,1])
    
    feature_df[['YEAR', 'WEEK']] = feature_df['Weekly_Date_Custom'].apply(lambda x: pd.Series(date_to_year_week(x)))
    feature_df.set_index(['YEAR', 'WEEK'], inplace = True)
    
    target_df = target_df.loc[feature_df.index, :]
    
    x = feature_df.loc[:, feature_col_name].values
    y = target_df.loc[:, target_col_name].values
    
    ccf = sm.tsa.stattools.ccf(x, y, nlags = max_lags, adjusted = False)
    optimal_lag = np.argmax(ccf)
    
    return ccf, optimal_lag