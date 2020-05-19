import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error as mse

from itertools import product

from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings


def check_stationary(df_series):
    # podziął na dwa zbiory
    df_series = df_series.copy()
    df_series.dropna(inplace=True)
    first_half = df_series.iloc[:len(df_series)//2]
    second_half = df_series.iloc[len(df_series)//2:]
    print(f"Means: 1st half: {first_half.mean()}, 2nd half: {second_half.mean()}")
    print(f"Std: 1st half: {first_half.std()}, 2nd half: {second_half.std()}")
    # ADFuller Test 
    result = adfuller(df_series)
    print("______ADFuller Test______")
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')


def plot_decomposition(df_series):
    decomposition = seasonal_decompose(df_series)
    return decomposition


def sarima_forecast(history_df, sarima_config, n_steps=1):
    order, sorder, trend = sarima_config
    # model = SARIMAX(history_df, order=order, seasonal_order=sorder, trend=trend)
    model = SARIMAX(history_df, order=order, seasonal_order=sorder, enforce_stationarity=False)
    model_fit = model.fit(disp=False)
    yhat = model_fit.predict(len(history_df), len(history_df) + n_steps - 1)
    return yhat[:n_steps]


def possible_config(
    p_params = [0, 1, 2],
    d_params = [0],
    q_params = [0, 1],
    t_params = ['c','t','ct'],
    P_params = [0, 1, 2],
    D_params = [0, 1],
    Q_params = [0, 1],  
    m_params = [0]):

    arima_params = list(product(p_params, d_params, q_params))
    sarima_params = list(product(P_params, D_params, Q_params, m_params))
    models = list(product(arima_params, sarima_params, t_params))
    return models


def grid_search(data, cfg_list, n_test, parallel=True, n_history=0):
    scores = None
    if parallel:
        executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
        tasks = (delayed(_score_model)(data, n_test, cfg, n_history) for cfg in cfg_list)
        scores = executor(tasks)
    else:
        scores = [_score_model(data, n_test, cfg, n_history) for cfg in cfg_list]

    scores = [r for r in scores if r[1] != None]
    scores.sort(key=lambda tup: tup[1])
    return scores


def _walk_forward_validation(data, n_test, cfg, n_history=0, recive_data=False):
    predictions = list()
    train, test = _train_test_split(data, n_test)
    history = [x for x in train]

    for i in range(len(test)):
        history = history[-n_history:]
        yhat = sarima_forecast(history, cfg)
        predictions.append(yhat)
        history.append(test[i])
    error = _measure_rmse(test, predictions)
    return error


def _measure_rmse(actual, predicted):
    return mse(actual, predicted)**0.5


def _train_test_split(df, n_test):
    return df[:-n_test], df[-n_test:]


def _score_model(data, n_test, cfg, n_history=0, debug=False):
    result = None
    # convert config to a key
    key = str(cfg)
    # show all warnings and fail on exception if debugging
    if debug:
        result = _walk_forward_validation(data, n_test, cfg, n_history)
    else:
        # one failure during model validation suggests an unstable config
        try:
            # never show warnings when grid searching, too noisy
            with catch_warnings():
                filterwarnings("ignore")
                result = _walk_forward_validation(data, n_test, cfg, n_history)
        except:
            print(f'{n_history}, {key}: RMSE= NaN')
    # check for an interesting result
    if result is not None:
        print(f'{n_history}, {key}: RMSE= {result}')
    return (key, result)