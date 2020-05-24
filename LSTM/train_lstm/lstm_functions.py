import pandas as pd
import numpy as np
import random




def preprocesing_data(series, time_window_len):
    
    sequential_data = []
    for idx in range(len(series) - time_window_len):
        sequential_data.append([np.array(series.values[idx : idx+time_window_len]), series.values[idx+time_window_len]])
    
    X = []
    y = []
    for seq, target in sequential_data: 
        X.append(seq)  
        y.append(target) 
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    indexes = series.index[time_window_len:]
    return X, np.array(y), indexes

def shuffle_dataset(X, y, rseed=0):
    random.seed(rseed)
    a = np.arange(len(y))
    random.shuffle(a)
    return X[a], y[a]