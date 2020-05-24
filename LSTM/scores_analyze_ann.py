import pandas as pd
import numpy as np
from modele_autoregresyjne.sarima_functions import sarima_forecast, grid_search
from preprocessing_danych.dataset_config import train_index, val_index, test_index
from LSTM.train_lstm.lstm_functions import preprocesing_data
import re


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
import tensorflow as tf

def results_from_logs(file_path):
    model_type = file_path.split('_results')[0]   
    f = open(file_path)
    names = []
    rmse_val = []
    rmse_train = []
    results = pd.DataFrame()

    for line in f:
        if line.startswith('Dumped tool data for kernel_stats.pb to '):
            name = line.split('Dumped tool data for kernel_stats.pb to ')[1]
            name = name.split('/train/')[0]
            names.append(name)
        if line.startswith('RMSE_train'):
            rmse_ = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0]
            rmse_train.append(rmse_)
        if line.startswith('RMSE_val'):
            rmse_ = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)[0]
            rmse_val.append(rmse_)

    results['name'] = names
    results['RMSE_val'] = rmse_val
    results['RMSE_train'] = rmse_train
    results['normalize_function'] = [name.split('/')[0].replace('logs_','') for name in results['name']]
    results['model_name'] = [name.split('/')[1] for name in results['name']]
    results['RMSE_val'] = pd.to_numeric(results['RMSE_val'])
    results['RMSE_train'] = pd.to_numeric(results['RMSE_train'])
    results['model_type'] = [s_.split('-')[0] for s_ in results['model_name']]
    results['window_len'] = [s_.split('-')[1] for s_ in results['model_name']]
    results['window_len'] = results['window_len'].astype(int)
    results['scaller'] = [s_.split('-')[3] for s_ in results['model_name']]
    results['rnn_cells'] = [s_.split('-')[5] for s_ in results['model_name']]
    results['rnn_cells'] = results['rnn_cells'].astype(int)
    results['extra_hidden_layer'] = np.where(results.index % 2 == 1, 1, 0)

    return results.drop('name', axis=1)



spot_df = pd.read_pickle("data/tge_spot_preprocessed.p")

df_ = spot_df.diff().dropna()
df_test = df_.loc[test_index[0]:test_index[1]]
n_test = len(df_test)


score_df = spot_df.copy()
score_df['diff'] = spot_df['TGEgasDA'].diff()
score_df['pct_change'] = spot_df['TGEgasDA'].pct_change()
score_df['reference'] = 0
score_df.dropna(inplace=True)
df_score_ = score_df.loc[train_index[0]:val_index[0]][90:-1]
train_score_pct = mse(df_score_['pct_change'], df_score_['reference'])**(1/2)
train_score_diff = mse(df_score_['diff'], df_score_['reference'])**(1/2)
train_score_pct, train_score_diff

df_score_ = score_df.loc[val_index[0]:val_index[1]]
val_score_pct = mse(df_score_['pct_change'], df_score_['reference'])**(1/2)
val_score_diff = mse(df_score_['diff'], df_score_['reference'])**(1/2)
val_score_pct, val_score_diff

df_score_ = score_df.loc[test_index[0]:test_index[1]]
test_score_pct = mse(df_score_['pct_change'], df_score_['reference'])**(1/2)
test_score_diff = mse(df_score_['diff'], df_score_['reference'])**(1/2)
test_score_pct, test_score_diff



files = ['LSTM/LSTM2_RNN_results/lstm_2warst_rnn.log', 
        'LSTM/LSTM2_results/lstm_2warstw.log',
        'LSTM/LSTM3_results/lstm_3warstw.log',
        'LSTM/LSTM1_results/lstm_1warstw.log']


results_all = pd.DataFrame()
for file_ in files:
    print(file_)
    results_all = results_all.append(results_from_logs(file_))

# results_all.to_csv('Documentation/LSTM_results.csv')
# results_all.read_csv('Documentation/LSTM_results.csv')



best_pct_df = results_all.query(f"normalize_function == 'pct_change' & RMSE_train < {train_score_pct} & RMSE_val < {val_score_pct}").sort_values('RMSE_val').head(5)
best_diff_df = results_all.query(f"normalize_function == 'diff' & RMSE_train < {train_score_diff} & RMSE_val < {val_score_diff}").sort_values('RMSE_val').head(5)



# for k, row in best_pct_df[:5].iterrows():
#     sequence_size = row['window_len']
#     model_file = f"LSTM/{row['model_type']}_results/models_{row['normalize_function']}/{row['model_name']}"
#     model = tf.keras.models.load_model(model_file)
#     data_set = spot_df.pct_change() if (row['normalize_function'] == 'pct_change') else spot_df.diff()
#     data_set.dropna(inplace=True)
#     scaler = StandardScaler() if (row['scaller'] == 'STD') else MinMaxScaler(feature_range=(0,1))
#     df_scalled = pd.DataFrame(scaler.fit_transform(data_set), index=data_set.index, columns=['TGEgasDA'])
#     df_scalled.dropna(inplace=True)
#     X, y, idxes = preprocesing_data(df_scalled, sequence_size)
#     split_test_idx = idxes.astype(str).to_list().index(val_index[0])
#     test_X, test_y = X[split_test_idx:], y[split_test_idx:]
#     pred_y = scaler.inverse_transform(model.predict(test_X))
#     df_pred = pd.DataFrame(pred_y, index=spot_df.loc[val_index[0]:].index)
#     column_name = row['normalize_function']
#     df_pred.columns = ['forecast_'+column_name]
#     df_pred['forecast_'+column_name].to_pickle(f"forecast/{row['normalize_function']}/{row['model_name']}.p",protocol=2)

# for k, row in best_diff_df[:5].iterrows():
#     sequence_size = row['window_len']
#     model_file = f"LSTM/{row['model_type']}_results/models_{row['normalize_function']}/{row['model_name']}"
#     model = tf.keras.models.load_model(model_file)
#     data_set = spot_df.pct_change() if (row['normalize_function'] == 'pct_change') else spot_df.diff()
#     data_set.dropna(inplace=True)
#     scaler = StandardScaler() if (row['scaller'] == 'STD') else MinMaxScaler(feature_range=(0,1))
#     df_scalled = pd.DataFrame(scaler.fit_transform(data_set), index=data_set.index, columns=['TGEgasDA'])
#     df_scalled.dropna(inplace=True)
#     X, y, idxes = preprocesing_data(df_scalled, sequence_size)
#     split_test_idx = idxes.astype(str).to_list().index(val_index[0])
#     test_X, test_y = X[split_test_idx:], y[split_test_idx:]
#     pred_y = scaler.inverse_transform(model.predict(test_X))
#     df_pred = pd.DataFrame(pred_y, index=spot_df.loc[val_index[0]:].index)
#     column_name = row['normalize_function']
#     df_pred.columns = ['forecast_'+column_name]
#     df_pred['forecast_'+column_name].to_pickle(f"forecast/{row['normalize_function']}/{row['model_name']}.p",protocol=2)



#################### ploting

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


plt.style.use('ggplot')


def sarima_forecast_plots(df, title, ylabels, xlabel, residuals=True, save_file=''):
    if residuals:
        fig, (ax_series, ax_residuals) = plt.subplots(2, 1, figsize=(20, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    else: 
        fig, ax_series = plt.subplots(figsize=(20, 5))

    fig.suptitle(title, fontsize=20)

    ax_series.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax_series.xaxis.set_minor_formatter(mdates.DateFormatter('%Y-%m'))
    ax_series.set_xlim(df.index.min()-pd.Timedelta('14d'), df.index.max()+pd.Timedelta('14d'))
    ax_series.set_ylabel(ylabels[0], fontsize=15)
    if residuals:
        ax_residuals.set_ylabel(ylabels[1], fontsize=10)
        ax_residuals.set_xlabel(xlabel, fontsize=15)
    else:
        ax_series.set_xlabel(xlabel, fontsize=15)
    plt.xticks(rotation=45)
    
    ax_series.plot(df.index, df['TGEgasDA'], label='cena rzeczywista', c='darkblue', linewidth=3, alpha=0.4)
    ax_series.plot(df.index, df['forecast_val'], label='prognoza - zbiór walidacyjny', linestyle='--', c='gold')
    ax_series.plot(df.index, df['forecast_test'], label='prognoza - zbiór testowy', linestyle='--', c='darkorange')

    if residuals:
        ax_residuals.fill_between(df.index, df['residuals_val'], label=['forecast_val'], color='gold')
        ax_residuals.fill_between(df.index, df['residuals_test'], label=['forecast_test'], color='darkorange')
    ax_series.legend(loc='upper right', prop={"size":15})
    if save_file != '': 
        plt.savefig(f"{save_file}.png", bbox_inches = "tight")
    else: 
        plt.show()        
    pass




for a, row in best_diff_df[:5].iterrows():
    df_forecast = pd.read_pickle(f"forecast/{row['normalize_function']}/{row['model_name']}.p")
    if (row['extra_hidden_layer'] == 1):
        file_name = f"{row['model_name'].rsplit('-',1)[0]}-HL-{row['normalize_function']}"
    else: 
        file_name = f"{row['model_name'].rsplit('-',1)[0]}_{row['normalize_function']}"
    
    if (row['extra_hidden_layer'] == 1):
        title = f"Prognoza TGEgasDA modelu {row['model_name'].rsplit('-',1)[0]}_HL"
    else:
        title = f"Prognoza TGEgasDA modelu {row['model_name'].rsplit('-',1)[0]}"
    df_joined = spot_df.join(df_forecast)


    df_joined['forecast'] = df_joined['forecast_diff'] + df_joined['TGEgasDA'].shift()
    df_joined['residuals'] =  (df_joined['TGEgasDA'] - df_joined['forecast']).abs()

    df_joined['forecast_val'] = df_joined.loc[val_index[0] : val_index[1], 'forecast']
    df_joined['residuals_val'] = df_joined.loc[val_index[0] : val_index[1], 'residuals']

    df_joined['forecast_test'] = df_joined.loc[test_index[0] : test_index[1], 'forecast']
    df_joined['residuals_test'] = df_joined.loc[test_index[0] : test_index[1], 'residuals']

    df_joined.head()



    sarima_forecast_plots(df_joined, 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'], 
                        'Dzień dostawy kontraktu',
                        save_file=f"Documentation/{file_name}_full",
                        residuals=False)

    sarima_forecast_plots(df_joined[~df_joined['forecast'].isna()], 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'],
                        'Dzień dostawy kontraktu',
                        save_file=f"Documentation/{file_name}_test")


#pct_change
for a, row in best_pct_df[:5].iterrows():
    df_forecast = pd.read_pickle(f"forecast/{row['normalize_function']}/{row['model_name']}.p")
    if (row['extra_hidden_layer'] == 1):
        file_name = f"{row['model_name'].rsplit('-',1)[0]}-HL-{row['normalize_function']}"
    else: 
        file_name = f"{row['model_name'].rsplit('-',1)[0]}_{row['normalize_function']}"
    
    if (row['extra_hidden_layer'] == 1):
        title = f"Prognoza TGEgasDA modelu {row['model_name'].rsplit('-',1)[0]}_HL"
    else:
        title = f"Prognoza TGEgasDA modelu {row['model_name'].rsplit('-',1)[0]}"
    df_joined = spot_df.join(df_forecast)


    df_joined['forecast'] = (1 + df_joined['forecast_pct_change']) * df_joined['TGEgasDA'].shift()
    df_joined['residuals'] =  (df_joined['TGEgasDA'] - df_joined['forecast']).abs()

    df_joined['forecast_val'] = df_joined.loc[val_index[0] : val_index[1], 'forecast']
    df_joined['residuals_val'] = df_joined.loc[val_index[0] : val_index[1], 'residuals']

    df_joined['forecast_test'] = df_joined.loc[test_index[0] : test_index[1], 'forecast']
    df_joined['residuals_test'] = df_joined.loc[test_index[0] : test_index[1], 'residuals']

    df_joined.head()



    sarima_forecast_plots(df_joined, 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'], 
                        'Dzień dostawy kontraktu',
                        save_file=f"Documentation/{file_name}_full",
                        residuals=False)

    sarima_forecast_plots(df_joined[~df_joined['forecast'].isna()], 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'],
                        'Dzień dostawy kontraktu',
                        save_file=f"Documentation/{file_name}_test")

