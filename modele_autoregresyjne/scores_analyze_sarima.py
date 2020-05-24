import pandas as pd
from modele_autoregresyjne.sarima_functions import sarima_forecast, grid_search
from preprocessing_danych.dataset_config import test_index, val_index, test_index
from tqdm import tqdm
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



#SARIMA

DIFF_FUNCTION = 'pct_change'

if (DIFF_FUNCTION == 'pct_change'):
    df = pd.read_csv('Documentation/sarima_hypertuning_pct_change.csv', na_values=['None'])

if (DIFF_FUNCTION == 'diff'):    
    df = pd.read_csv('Documentation/sarima_hypertuning_diff.csv', na_values=['None'])


df.sort_values('RMSE').groupby("time_window").head(5).groupby("time_window")['RMSE'].mean()
df.sort_values('RMSE').groupby("time_window").head(10).groupby("time_window")['RMSE'].mean()
df.sort_values('RMSE').groupby("time_window")['RMSE'].mean()
df.sort_values('RMSE').query('time_window == 3000').head(10)


# diff
if (DIFF_FUNCTION == 'pct_change'):
    cfgs_best = {
        1: ((0, 0, 0), (1, 0, 1, 7), None), 
        2: ((2, 0, 0), (1, 0, 1, 7), None), 
        3: ((0, 0, 1), (1, 0, 1, 7), None),
        4: ((2, 0, 0), (2, 0, 1, 7), None), 
        5: ((2, 0, 1), (1, 0, 1, 7), None),
        6: ((2, 0, 1), (2, 0, 0, 7), None),
        7: ((2, 0, 0), (2, 0, 0, 7), None)
        }

if (DIFF_FUNCTION == 'diff'):
    cfgs_best = {
        1: ((0, 0, 0), (1, 0, 1, 7), None), 
        2: ((0, 0, 0), (2, 0, 1, 7), None), 
        3: ((0, 0, 1), (1, 0, 1, 7), None),
        4: ((2, 0, 1), (2, 0, 1, 7), None), 
        5: ((0, 0, 1), (2, 0, 1, 7), None)
        }





spot_df = pd.read_pickle("data/tge_spot_preprocessed.p")

if (DIFF_FUNCTION == 'pct_change'):
    df_ = spot_df.pct_change().dropna() 

if (DIFF_FUNCTION == 'diff'):    
    df_ = spot_df.diff().dropna() # if diff



df_test = df_.loc[test_index[0] : test_index[1]]
df_val = df_.loc[val_index[0] : val_index[1]]
liczba_dni_testowych = len(df_test) + len(df_val)

for idx in range(1, 6):
    cfg = cfgs_best[idx]
    price_forecast = pd.DataFrame()
    for k, row in df_[-liczba_dni_testowych:].iterrows():
        history_df = df_.loc[:k, 'TGEgasDA'][:-1]
        forecast = sarima_forecast(history_df, cfg)
        price_forecast = forecast if price_forecast.empty else price_forecast.append(forecast)
    price_forecast = pd.DataFrame(price_forecast, columns=['diff_forecast'])
    price_forecast.to_pickle(f"forecast/{DIFF_FUNCTION}/sarima_forecast_{idx}.p", protocol=2)



# diff
for idx in range(1,6):
    
    df_forecast = pd.read_pickle(f'forecast/diff/sarima_forecast_{idx}.p')

    cfg_title = cfgs_best[idx][:2]
    file_name = '-'.join([str(i) for i in cfg_title[0]]) + '_' + '-'.join([str(i) for i in cfg_title[1]])
    
    title = f"Prognoza TGEgasDA modelu SARIMA {cfg_title}"
    df_joined = spot_df.join(df_forecast)

    df_joined['forecast'] = df_joined['diff_forecast'] + df_joined['TGEgasDA'].shift()
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
                        save_file=f'Documentation/SARIMA_forecast_diff_{file_name}_full',
                        residuals=False)

    sarima_forecast_plots(df_joined[~df_joined['forecast'].isna()], 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'], 
                        'Dzień dostawy kontraktu', 
                        save_file=f'Documentation/SARIMA_forecast_diff_{file_name}_test')




# pct
for idx in range(7,8):
    
    df_forecast = pd.read_pickle(f'forecast/pct_change/sarima_forecast_{idx}.p')

    cfg_title = cfgs_best[idx][:2]
    file_name = '-'.join([str(i) for i in cfg_title[0]]) + '_' + '-'.join([str(i) for i in cfg_title[1]])
    
    title = f"Prognoza TGEgasDA modelu SARIMA {cfg_title}"
    df_joined = spot_df.join(df_forecast)

    df_joined['forecast'] = (1 + df_joined['diff_forecast']) * df_joined['TGEgasDA'].shift()
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
                        save_file=f'Documentation/SARIMA_forecast_pct_change_{file_name}_full',
                        residuals=False)

    sarima_forecast_plots(df_joined[~df_joined['forecast'].isna()], 
                        title, 
                        ['Cena [PLN/MWh]', 'Błąd bezwzględny [PLN/MWh]'], 
                        'Dzień dostawy kontraktu', 
                        save_file=f'Documentation/SARIMA_forecast_pct_change_{file_name}_test')
