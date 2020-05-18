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