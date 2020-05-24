import pandas as pd
import os
global spot_df
spot_df = pd.read_pickle('data/tge_spot_preprocessed.p')


def diff_forecast(file_path, file_name):
    forecast_df = pd.read_pickle(file_path+file_name+'.p')
    if type(forecast_df) == type(pd.DataFrame()):
        forecast_df = forecast_df.iloc[:,0]
    df_joined = spot_df.join(forecast_df.rename('forecast'))
    df_joined['forecast'] = df_joined['forecast'] + df_joined['TGEgasDA'].shift()
    df_joined = df_joined[['forecast']].rename(columns={'forecast': file_name})
    return df_joined

def pct_change_forecast(file_path, file_name):
    forecast_df = pd.read_pickle(file_path+file_name+'.p')
    if type(forecast_df) == type(pd.DataFrame()):
        forecast_df = forecast_df.iloc[:,0]
    df_joined = spot_df.join(forecast_df.rename('forecast'))
    df_joined['forecast'] = (1 + df_joined['forecast']) * df_joined['TGEgasDA'].shift()
    df_joined = df_joined[['forecast']].rename(columns={'forecast': file_name})
    return df_joined


def joined_forecast(path_forecast):
    df_all = pd.DataFrame()

    for file_ in os.listdir(path_forecast):
        file_name = file_.rstrip('.p')
        df = diff_forecast(path_forecast, file_name)
        if df_all.empty:
            df_all = df.copy()
        else:
            df_all = df_all.join(df)    
    df_all.to_pickle(path_forecast + 'forecast_joined.p')




joined_forecast("forecast/pct_change/")
joined_forecast("forecast/diff/")









