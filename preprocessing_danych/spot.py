import pandas as pd


spot_filename = 'data/tge_spot.p'
spot_df = pd.read_pickle(spot_filename)[['Price']]
spot_df.rename(columns={'Price': 'TGEgasDA'}, inplace=True)

data_range = pd.date_range(spot_df.index.min(), spot_df.index.max())
missing_dates = set(data_range) - set(spot_df.index)
print("Brakujące dni:")
[print(i.strftime("%d-%m-%Y")) for i in missing_dates]


###### >>>> 1 uzupełnienie braku transakcji za pomocą metody fill forward
spot_ffill_df = spot_df.resample('1d').mean()
spot_ffill_df[spot_ffill_df.index.isin(missing_dates) \
    | spot_ffill_df.index.shift(1).isin(missing_dates) \
    | spot_ffill_df.index.shift(-1).isin(missing_dates) ]

spot_ffill_df['TGEgasDA'] = spot_ffill_df['TGEgasDA'].ffill()
spot_ffill_df[spot_ffill_df.index.isin(missing_dates) \
    | spot_ffill_df.index.shift(1).isin(missing_dates) \
    | spot_ffill_df.index.shift(-1).isin(missing_dates) ]

spot_ffill_df.to_pickle("data/tge_spot_fill.p")





spot_tge_arch = pd.read_csv("data/RDNG_INDEX_REPORT.csv", skiprows=1, sep=';')
spot_tge_arch.rename(columns={'data dostawy': 'TransactionDate', 'kurs (PLN/MWh)':'TGEgasDA'}, inplace=True)


spot_tge_arch = spot_tge_arch[ spot_tge_arch['indeks'] == 'TGEgasDA']
spot_tge_arch['TransactionDate'] = pd.to_datetime(spot_tge_arch['TransactionDate'])
spot_tge_arch.set_index('TransactionDate', inplace=True)

spot_tge_arch = spot_tge_arch[['TGEgasDA']]
spot_tge_arch['TGEgasDA'] = pd.to_numeric(spot_tge_arch['TGEgasDA'])

spot_df_full = spot_tge_arch.join(spot_ffill_df, how='outer', lsuffix='_arch')
spot_df_full['TGEgasDA'] = spot_df_full['TGEgasDA'].fillna(spot_df_full['TGEgasDA_arch'])
spot_df_full = spot_df_full[['TGEgasDA']]


spot_df_full.to_pickle('data/tge_spot_preprocessed.p')



















# Brakuące dni dostawy na RDN



