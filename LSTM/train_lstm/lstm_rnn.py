import pandas as pd
import numpy as np
import random
import sys
sys.path.append('/home/maciej/Documents/gazspot-prediction/')
import time

from preprocessing_danych.dataset_config import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

# !pip install tensorflow==1.6
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, SimpleRNN
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

def build_model(input_shape,
                rnn_size, 
                dropout_level, 
                batch_normalization, 
                hidden_dense_layer_size,
                optimazer,
                learning_rate, 
                batch_size):
    model = Sequential()


    model.add(LSTM(rnn_size, input_shape=(input_shape)))
    if (dropout_level > 0): model.add(Dropout(dropout_level))
    if batch_normalization: model.add(BatchNormalization())
    if (hidden_dense_layer_size > 0):
        model.add(Dense(hidden_dense_layer_size, activation='relu'))
        if (dropout_level > 0): model.add(Dropout(0.2))

    model.add(Dense(1))
    opts = {'rmsprop': tf.keras.optimizers.RMSprop(learning_rate=learning_rate, decay=1e-6),
            'adam': tf.keras.optimizers.Adam(lr=learning_rate, decay=1e-6)}
    opt = opts[optimazer]
   
    # Compile model
    model.compile(
        loss='mse',
        optimizer=opt
    )
    return model

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


# preprocessing configs

PRED_LEN = 1


diff_functions = ['pct_change', 'diff']
scalers = ['MiMax', 'STD']
sequence_sizes = [7, 14, 21, 28, 49]


# LSTM configs 

MODEL_TYPE = 'LSTM1'
EPOCHS = 30

rnn_sizes = [64, 256]
dropout_levels = [0.1]
batch_normalizations = [True]
hidden_dense_layer_sizes = [0, 32]
optimazers = ['adam']
learning_rates = [0.001]
batch_sizes = [8]


#TODO: wykorzystanie analizy z ARIMY odnośnie wpływających danych - sezonowość

# Wczytanie danych
# df = pd.read_csv("/content/drive/My Drive/Mgr_gas_transaction/tge_spot_preprocessed.csv",index_col=['TransactionDate'], parse_dates=['TransactionDate'])
df = pd.read_pickle("data/tge_spot_preprocessed.p")

for diff_function in diff_functions:
    if (diff_function == 'pct_change'):
        df_new = df.pct_change().dropna()
    if (diff_function == 'diff'):
        df_new = df.diff().dropna()
    
    for scaler_name in scalers:
        if (scaler_name == 'STD'):
            scaler = StandardScaler()
        if (scaler_name == 'MiMax'):
            scaler = MinMaxScaler(feature_range=(0,1))
        
        df_scalled = pd.DataFrame(scaler.fit_transform(df_new), index=df_new.index, columns=['TGEgasDA'])
        df_scalled.dropna(inplace=True)

        for sequence_size in sequence_sizes:

            X, y, idxes = preprocesing_data(df_scalled, sequence_size)

            split_test_idx = idxes.astype(str).to_list().index(test_index[0])
            split_idx = idxes.astype(str).to_list().index(val_index[0])


            train_X, train_y = X[:split_idx], y[:split_idx]
            val_X, val_y = X[split_idx : split_test_idx], y[split_idx : split_test_idx]
            test_X, test_y = X[split_test_idx:], y[split_test_idx:]

            # randomize batch before train network
            train_X, train_y = shuffle_dataset(train_X, train_y)

            for rnn_size in rnn_sizes:
                for dropout_level in dropout_levels:
                    for batch_normalization in batch_normalizations:
                        for hidden_dense_layer_size in hidden_dense_layer_sizes:
                            for optimazer in optimazers:
                                for learning_rate in learning_rates:
                                    for batch_size in batch_sizes:
                                        
                                        model = build_model(train_X.shape[1:],
                                                            rnn_size, 
                                                            dropout_level, 
                                                            batch_normalization, 
                                                            hidden_dense_layer_size,
                                                            optimazer,
                                                            learning_rate, 
                                                            batch_size)
                                        
                                        NAME = f"{MODEL_TYPE}-{sequence_size}-W_LEN-{scaler_name}-SCL-{rnn_size}-RNN_S-{int(time.time())}"
                                        tensorboard = TensorBoard(log_dir=f"logs_{diff_function}/{NAME}")
                                    
                                        model.fit(train_X, train_y, 
                                                    epochs=EPOCHS, batch_size=batch_size, 
                                                    validation_data=(val_X, val_y), 
                                                    callbacks=[tensorboard])
                                        model.save(f"models_{diff_function}/{NAME}")

                                        # score model
                                        pred_y = scaler.inverse_transform(model.predict(train_X))
                                        true_y = scaler.inverse_transform(train_y)
                                        print(f"RMSE_train = {mse(true_y, pred_y) ** (1/2)}")


                                        pred_y = scaler.inverse_transform(model.predict(val_X))
                                        true_y = scaler.inverse_transform(val_y)
                                        print(f"RMSE_val = {mse(true_y, pred_y) ** (1/2)}")









# # scaler.inverse_transform(train_y.reshape(-1,1))
# # scaler.inverse_transform(model.predict(train_X))

# data = 
# # df_true = df[train_index[0]: val_index[0]][:-1]
# # df_true = df[test_index[0]: test_index[1]]
# df_true = df[val_index[0]: val_index[1]]

# df_true['pct_change'] = df_true['TGEgasDA'].pct_change()
# df_true['TGEgasDA_shift'] = df_true['TGEgasDA'].shift()
# df_true.dropna(inplace=True)
# df_true = df_true[TIME_WINDOW_LEN:]
# print(len(df_true))
# print(len(data))
# df_true['pred'] = data[1:]
# df_true['TGEgasDA_pred'] = df_true['TGEgasDA_shift'] + df_true['TGEgasDA_shift'] * df_true['pred']
# df_true[-30:]

# mse(df_true['pct_change'], df_true['pred']) ** (1/2)

# df_true[['pct_change','pred']].plot(figsize=(20,4))

# mse(df_true['TGEgasDA'][1:],df_true['TGEgasDA_pred'].shift().dropna()) ** (1/2)

# mse(df_true['TGEgasDA'],df_true['TGEgasDA_pred']) ** (1/2)

# df_true[['TGEgasDA','TGEgasDA_pred']].plot(figsize=(20,4))


# # TODO: print RMSE on val_dataset _ invert on scaler