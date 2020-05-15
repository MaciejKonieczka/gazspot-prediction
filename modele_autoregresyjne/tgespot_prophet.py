#%% import packeges
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import plotly.offline as py
import tqdm
import tempfile
from math import sqrt

from sklearn.metrics import mean_squared_error as mse

tempfile.tempdir = '/data/mkonieczka'
#%%

def _measure_rmse(actual, predicted):
    return sqrt(mse(actual, predicted))
#%% import data
df = pd.read_pickle(r"/data/PGE_SA_RynekGazu/RynekGazu_SieciNeuronowe/data/tge_spot_preporcessed.p")
df.reset_index(inplace=True)
#%%
# data = df[['delivery_date', 'Price']]
data = df.copy()

okres_testowy = 100 #dni

data.columns = ['ds', 'y']
data_test = data[-okres_testowy:]

forecast_all = pd.DataFrame()
for day in tqdm.tqdm(data_test['ds']):
    model = Prophet(daily_seasonality=False)
    current_dataset = data[data['ds'] < day]
    model.fit(current_dataset) # wybor zbioru treningowego
    ## predykcja na jeden dzien
    future = model.make_future_dataframe(periods=1, include_history=False)
    forecast = model.predict(future)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    forecast_all = forecast_all.append(forecast)
model = Prophet(daily_seasonality=False)
model.fit(data)



#%% 
# py.init_notebook_mode(connected=True)


_measure_rmse(data_test['y'], forecast_all['yhat'])

# fig = plot_plotly(model, forecast_all)  # This returns a plotly Figure
# py.plot(fig)



# %%
forecast_all.to_pickle('prophet_test100.p')



# %%
