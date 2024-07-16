import os
import sys
import warnings

import numpy as np


def time_series_lag_analysis(seq_x,seq_y):
    pass

import numpy as np
from scipy.signal import correlate

def calculate_time_precedence(web_data, ili_data, max_lag=15):
    """
    웹 데이터의 시간 선행도를 ILI 데이터에 대해 계산합니다.
    
    Parameters:
    web_data (np.array): 웹 데이터 시계열 배열.
    ili_data (np.array): ILI 데이터 시계열 배열.
    max_lag (int): 교차 상관 관계를 고려할 최대 시차.
    
    Returns:
    int: 교차 상관 관계가 최대가 되는 시간 선행도(시차).
    """
    # 데이터를 numpy 배열 형식으로 변환
    web_data = np.array(web_data)
    ili_data = np.array(ili_data)
    
    # 데이터 정규화
    web_data_mean = np.mean(web_data)
    ili_data_mean = np.mean(ili_data)
    
    web_data_std = np.std(web_data)
    ili_data_std = np.std(ili_data)
    
    normalized_web_data = (web_data - web_data_mean) / web_data_std
    normalized_ili_data = (ili_data - ili_data_mean) / ili_data_std
    
    # 0부터 max_lag까지 시차에 대한 교차 상관 관계 계산
    cross_corr = np.array([np.correlate(normalized_web_data, np.roll(normalized_ili_data, lag))[0]
                           for lag in range(1, max_lag+1)])
    
    # 교차 상관 관계가 최대가 되는 시차 찾기
    optimal_lag = np.argmax(cross_corr) + 1
    
    return optimal_lag

# 예제 데이터로 사용
web_data = np.random.randn(100)
ili_data = np.random.randn(100)

time_precedence = calculate_time_precedence(web_data, ili_data)
print(f'교차 상관 관계가 최대가 되는 시간 선행도(시차): {time_precedence}')