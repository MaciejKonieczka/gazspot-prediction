import pandas as pd
import matplotlib.pyplot as plt
try:
    from preprocessing_danych.dataset_config import test_index, train_index, val_index
except:
    from dataset_config import test_index, train_index, val_index


spot_df = pd.read_pickle('data/tge_spot_preprocessed.p')


train_ = spot_df.loc[train_index[0]:train_index[1]]
print("Zbior_treningowy:")
print(f"Od: {train_.index.min()} do: {train_.index.max()}")
print(f"Liczba obserwacji: {train_.shape[0]} ({round(100* train_.shape[0] / spot_df.shape[0], 2)}%)")

val_ = spot_df.loc[val_index[0]:val_index[1]]
print("")
print("Zbior_walidacyjny:")
print(f"Od: {val_.index.min()} do: {val_.index.max()}")
print(f"""Liczba obserwacji: {val_.shape[0]} \
({round(100* val_.shape[0] / spot_df.shape[0], 2)}% wszyskich obserwacji \
i {round(100* val_.shape[0] / train_.shape[0], 2)}% zbioru treningowego)""")

test_ = spot_df.loc[test_index[0]:test_index[1]]
print("")
print("Zbior_testowy:")
print(f"Od: {test_.index.min()} do: {test_.index.max()}")
print(f"Liczba obserwacji: {test_.shape[0]} ({round(100* test_.shape[0] / spot_df.shape[0], 2)}%)")

