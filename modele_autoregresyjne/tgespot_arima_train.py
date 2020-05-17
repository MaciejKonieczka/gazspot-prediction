import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('ggplot')

try:
    from sarima_functions import plot_decomposition, possible_config, grid_search, sarima_forecast, check_stationary
except:
    from modele_autoregresyjne.sarima_functions import plot_decomposition, possible_config, grid_search, sarima_forecast, check_stationary



def decomposition_plot(df, title, ylabel, time_, save_file=''):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(20,20))
    sd = seasonal_decompose(df.loc[time_])
    fig.suptitle(f'{title} {time_}', fontsize=20, y=0.9)
    ax[0].plot(sd._observed.index, sd._observed, c='darkblue')
    ax[0].set_ylabel(ylabel, fontsize=15)
    ax[1].plot(sd._observed.index, sd.trend, c='darkorange')
    ax[1].set_ylabel('Trend', fontsize=15)
    ax[2].plot(sd._observed.index, sd.seasonal, c='darkorange')
    ax[2].set_ylabel('Sezonowość', fontsize=15)
    ax[3].plot(sd._observed.index, sd.resid, c='darkorange')
    ax[3].set_ylabel('Szum', fontsize=15)
    ax[3].set_xlabel("Dzień dostawy kontraktu", fontsize=15)
    plt.xticks(rotation=45)
    if save_file != '': 
        plt.savefig(f"{save_file}_seasonaldecomp_{time_}.png", bbox_inches = "tight")
    else: 
        plt.show()        
    pass


def rolling_plot(df, title, ylabel, xlabel, roll, save_file=''):
    df_ = df.rolling(roll, min_periods=roll//2).agg(['mean', 'std'])
    df_.columns = ['średnia', 'odchylenie standardowe']
    fig, ax = plt.subplots(figsize=(20,6))
    fig.suptitle(f"{title} w oknie czasowym: {roll} dni", fontsize=20)
    
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_xlim(df.index.min()-pd.Timedelta('14d'), df.index.max()+pd.Timedelta('14d'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    ax.plot(df_.index, df_['średnia'], label='średnia', c='forestgreen')
    ax.plot(df_.index, df_['odchylenie standardowe'], label='odchylenie standardowe', c='darkorange')
    ax.legend(loc='upper right', prop={"size":15})
    if save_file != '': 
        plt.savefig(f"{save_file}_roll_{roll}.png", bbox_inches = "tight")
    else: 
        plt.show()        
    pass


def timeseries_plot(df, title, ylabel, xlabel, save_file=''):
    fig, ax = plt.subplots(figsize=(20,6))
    fig.suptitle(title, fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_xlim(df.index.min()-pd.Timedelta('14d'), df.index.max()+pd.Timedelta('14d'))
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    plt.xticks(rotation=45)

    ax.plot(df.index, df, label=df.columns[0], c='darkblue')
    ax.legend(loc='upper right', prop={"size":15})
    if save_file != '': 
        plt.savefig(save_file, bbox_inches = "tight")
    else: 
        plt.show()
    pass 


###############################################
###############################################
###############################################

# wczytanie plików
spot_df = pd.read_pickle("data/tge_spot_preprocessed.p")

######################################
######################################
##### Testy stacjonarności

## Zwykły przebieg

# Wykres przebiegu

print(">>>>>> Zwykły przebieg <<<<<<")
Path("Documentation/TGEgasDA").mkdir(parents=True, exist_ok=True)
timeseries_plot(
    df=spot_df, 
    title="Cena rozliczeniowa na Rynku Dnia Następnego na TGE",
    ylabel="Cena [PLN/MWh]",
    xlabel="Dzień dostawy kontraktu",
    save_file="Documentation/TGEgasDA/series.png"
    )

### Test ADFullera
check_stationary(spot_df['TGEgasDA'])
# Wykres zmianny średniej i zmiany odchylenia standardowego
for roll in [7, 30, 90, 365]:
    rolling_plot(
        df=spot_df,
        title="Przebieg średniej i odchylenia standardowego ceny rozliczeniowej TGEgasDA",
        ylabel="Cena [PLN/MWh]",
        xlabel="Dzień dostawy kontraktu - koniec okna czasowego",
        roll=roll,
        save_file='Documentation/TGEgasDA/series')

for time_ in ['2017Q4', '2019Q1']:
    decomposition_plot(
        df=spot_df,
        title="Dekompozycja sezonowa ceny rozliczeniowej TGEgasDA",
        ylabel="Cena [PLN/MWh]",
        time_=time_,
        save_file='Documentation/TGEgasDA/series'
    )


# ## Zmiana (diff)

print(">>>>>> Zmiana (diff) <<<<<<")
Path("Documentation/TGEgasDA_diff").mkdir(parents=True, exist_ok=True)

spot_diff_ = spot_df[['TGEgasDA']].diff()
check_stationary(spot_diff_)

timeseries_plot(spot_diff_,
    title="Zmiana dzień do dnia ceny rozliczeniowej na Rynku Dnia Następnego na TGE",
    ylabel="Zmiana ceny [PLN/MWh]",
    xlabel="Dzień dostawy kontraktu",
    save_file="Documentation/TGEgasDA_diff/series.png")

for roll in [7, 30, 90, 365]:
    rolling_plot(
        df=spot_diff_,
        title="Przebieg średniej i odchylenia standardowego zmiany dzień do dnia ceny rozliczeniowej",
        ylabel="Zmiana ceny [PLN/MWh]",
        xlabel="Dzień dostawy kontraktu - koniec okna czasowego",
        roll=roll,
        save_file="Documentation/TGEgasDA_diff/series")

for time_ in ['2017Q4', '2019Q1']:
    decomposition_plot(
        df=spot_diff_,
        title="Dekompozycja sezonowa zmiany dzień do dnia ceny rozliczeniowej TGEgasDA",
        ylabel='Zmiana ceny [PLN/MWh]',
        time_=time_,
        save_file="Documentation/TGEgasDA_diff/series"
    )



## Zmiana procentowa (pct_change)

print(">>>>>> Zmiana procentowa (pct_change)<<<<<<")

Path("Documentation/TGEgasDA_pctchange").mkdir(parents=True, exist_ok=True)

spot_pct_change_ = spot_df[['TGEgasDA']].pct_change()
check_stationary(spot_pct_change_)


timeseries_plot(spot_pct_change_*100,
    title= "Zmiana procentowa dzień do dnia ceny rozliczeniowej na Rynku Dnia Następnego na TGE",
    ylabel="Zmiana ceny [%]",
    xlabel="Dzień dostawy kontraktu",
    save_file="Documentation/TGEgasDA_pctchange/series.png")

for roll in [7, 30, 90, 365]:
    rolling_plot(
        df=spot_pct_change_,
        title="Przebieg średniej i odchylenia standardowego zmiany procentowej dzień do dnia ceny rozliczeniowej",
        ylabel="Zmiana ceny [%]",
        xlabel="Dzień dostawy kontraktu - koniec okna czasowego",
        roll=roll,
        save_file="Documentation/TGEgasDA_pctchange/series")

for time_ in ['2017Q4', '2019Q1']:
    decomposition_plot(
        df=spot_pct_change_,
        title="Dekompozycja sezonowa zmiany procentowej dzień do dnia ceny rozliczeniowej TGEgasDA",
        ylabel="Zmiana ceny [%]",
        time_=time_,
        save_file="Documentation/TGEgasDA_pctchange/series"
    )




















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

