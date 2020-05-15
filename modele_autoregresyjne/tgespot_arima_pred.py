#%% import packege
import pandas as pd
import matplotlib.pyplot as plt
from sarima_functions import sarima_forecast

DATA_PATH = '/data/PGE_SA_RynekGazu/RynekGazu_SieciNeuronowe/data/'
SARIMA_data = "tge_spot_preporcessedSARIMA.p"
spot_data = pd.read_pickle(DATA_PATH + SARIMA_data)


# %% SARIMA config
order = (2, 1, 1)
sorder = (1, 0, 1, 7)
trend = 't'
cfg = [order, sorder, trend]

# %% SARIMA forecast

sarima_forecast(spot_data, cfg)

# %%
