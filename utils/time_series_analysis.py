from .timefeatures import date_to_year_week

import os
import sys
import warnings

import torch
import numpy as np
import pandas as pd
# import statsmodels.api as sm


def dymanic_time_series_lag_analysis(
    feature_data:torch.Tensor,
    target_data:torch.Tensor,
):
    """
    Calculate the lag that each feature has on the target by cross-correlation.

    Parameters
    ----------
    feature_data : torch.Tensor
        _description_
    target_data : torch.Tensor
        _description_
    """
    _, n_features = feature_data.shape
    pred_len, _ = target_data.shape
    
    max_lag = pred_len
    
    assert int(max_lag) >= 3, 'max_lag (pred_len) should be greater than or equal to 3'
    
    feature_lags = np.zeros(n_features)
    
    # feature -> lag
    # caluclate cross-correlation
    # Loop over each feature
    for feature_idx in range(n_features):
        feature_seq = feature_data[:, feature_idx]
        target_seq = target_data.reshape(-1)
        
        ccf = sm.tsa.stattools.ccf(feature_seq, target_seq, nlags = max_lag, adjusted = False)
        lag = np.argmax(ccf) + 1
        feature_lags[feature_idx] = lag
    
    optimal_lag_index = np.argmax(feature_lags)
    optimal_lag = int(feature_lags[optimal_lag_index])
    
    return feature_lags, optimal_lag

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
    optimal_lag = np.argmax(ccf) + 1
    
    return ccf, optimal_lag