import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')

import sys
sys.path.append('/home/maciej/Documents/gazspot-prediction/')

from modele_autoregresyjne.sarima_functions import plot_decomposition, possible_config, grid_search, sarima_forecast, check_stationary

from preprocessing_danych.dataset_config import test_index, train_index, val_index



###############################################
###############################################
###############################################

# wczytanie plików
spot_df = pd.read_pickle("data/tge_spot_preprocessed.p")

#####

cfgs_ = possible_config(
    p_params = [0, 1, 2],
    d_params = [0],
    q_params = [0, 1],
    t_params = [None],
    P_params = [0, 1, 2],
    D_params = [0],
    Q_params = [0, 1],  
    m_params = [0, 7])



df_ = spot_df.diff().dropna()
df_train = spot_df.loc[train_index[0]:train_index[1]]
df_val = spot_df.loc[val_index[0]:val_index[1]]
n_test = len(df_val)
# grid_search(data = df_train['TGEgasDA'],
#     cfg_list=cfgs_, 
#     n_test=n_test)

# cfgs_windows = [
#     ((1, 0, 1), (1, 0, 1, 7), None), 
#     ((1, 0, 1), (2, 0, 1, 7), None), 
#     ((2, 0, 1), (1, 0, 1, 7), None)]

#TODO: Wybór okna czasowego przed grid searchem
# opcje: [730, 365, 90]
for window in [3000, 730, 365, 90]:
    print(window)
    grid_search(data=df_train['TGEgasDA'],
                cfg_list=cfgs_,
                n_test=n_test,
                n_history=window)
#### PCT change
print(">>>pct change<<<<<")
df_ = spot_df.pct_change().dropna()
df_train = spot_df.loc[train_index[0]:train_index[1]]
df_val = spot_df.loc[val_index[0]:val_index[1]]
n_test = len(df_val)
# grid_search(data = df_train['TGEgasDA'],
#     cfg_list=cfgs_, 
#     n_test=n_test)

# cfgs_windows = [
#     ((1, 0, 1), (1, 0, 1, 7), None), 
#     ((1, 0, 1), (2, 0, 1, 7), None), 
#     ((2, 0, 1), (1, 0, 1, 7), None)]

#TODO: Wybór okna czasowego przed grid searchem
# opcje: [730, 365, 90]
for window in [3000, 730, 365, 90]:
    print(window)
    grid_search(data=df_train['TGEgasDA'],
                cfg_list=cfgs_,
                n_test=n_test,
                n_history=window)

































# gaz_peak2018_range = ("2018-02-21", "2018-03-06")
# spot_df.loc[pd.date_range(*gaz_peak2018_range), "Price"] = np.nan
# spot_df.plot()
# plt.title("TGE_RDN_GAZ_ usunięcie okresu PEAKu z 2018", color = 'black')
# plt.ylabel("Cena [PLN]")
# # plt.savefig(PROJECT_PATH + "Documentation/tge_spot_rozliczeniowa_cleandata.png")

# # Decompozycja przebiegu

# plot = plot_decomposition(spot_df[:'2018-01']['Price'].diff().dropna())
# # plt.savefig(PROJECT_PATH + "Documentation/tge_spot_decomposiotionPlot.png")


# ####################################################
# ####################################################
# # best_sarima_model

# liczbadni_test = 56 # 8 tygodni testowych
# data = spot_df[spot_df.index < spot_df.isna().idxmax()[0]].values

# cfg_list = possible_config(m_params=[0, 7])

# ### Training grid search SARIMA model takes few hours
# # scores = grid_search(data, cfg_list, liczbadni_test)
# # print('done')
# # for cfg, error in scores[:20]:
# #     print(cfg, error)
# # scores.to_pickle("scores_test.p")


# # SARIMA to uzupełnienia dziur w danych
# # Wybór parametrów SARIMA_config
# order = (2, 1, 1)
# sorder = (1, 0, 1, 7)
# trend = 't'
# cfg = [order, sorder, trend]

# spot_resample = spot_df.resample('1d').mean()
# spot_resample['forecast'] = np.nan

# for k, row in spot_resample[spot_resample['Price'].isna()].iterrows():
#     history_df = spot_resample.loc[:k, 'Price'][:-1]
#     forecast = sarima_forecast(history_df, cfg)
#     spot_resample.loc[k, 'forecast'] = forecast.values[0]
#     # print(forecast)

# spot_resample.plot()
# plt.title("TGE_RDN_GAZ uzupełnienie PEAKu modelem SARIMA", color = 'black')
# plt.ylabel("Cena [PLN]")
# plt.savefig(PROJECT_PATH + "Documentation/tge_spot_rozliczeniowa_filledSARIMA.png")

# spot_filled = pd.DataFrame(spot_resample.sum(axis=1), columns=['Price'])
# spot_filled.to_pickle("/data/PGE_SA_RynekGazu/RynekGazu_SieciNeuronowe/data/tge_spot_preporcessedSARIMA.p")


# #Zastosowanie SARIMy do predykcji okresu testowego

# # okres_testowy = 100 #dni

# # order = (2, 1, 1)
# # sorder = (1, 0, 1, 7)
# # trend = 't'
# # cfg = [order, sorder, trend]


# # for k, row in spot_fillna[-okres_testowy:].iterrows():
# #     history_df = spot_fillna.loc[:k, 'Price'][:-1]
# #     forecast = sarima_forecast(history_df, cfg)
# #     spot_fillna.loc[k, 'forecast'] = forecast.values[0]
# # # sarima_forecast(spot_history, cfg, len(spot_to_forecast))
# # price_acctual = spot_fillna['Price'][-okres_testowy:]
# # price_forecast = spot_fillna['forecast'][-okres_testowy:]

# # print(_measure_rmse(price_acctual, price_forecast))



# # # 
# # spot_fillna['residuals'] = spot_fillna['Price'] - spot_fillna['forecast']

# # spot_fillna.plot()

# # # 
# # spot_fillna['forecast'].dropna().to_pickle('arima_test100.p')
# # # 

