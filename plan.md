# Predykcja ceny gazu na TGE RDN

## TODO:
### BASIC:
1. Wybór metryki (RMSE lub MAE)
- RMSE jako główna metryka
- MAE jako metryka pomocnicza przy ewaluacji
- Dodatkowa metryka (jak się uda) - Trend (rosnący/malejący/ ze brak zmiany 5 procentyl zmian dzień do dnia (może procentowo?))   
2. Podział na zbiory treningowy i testowy
- Zbiór treningowy [2017-]
- Zbiór testowy [2019-07-01 - 2020-04-30]


### SARIMA:
1. Optymalizacja hiperparametrów SARIMA (bez trendu, z sezonowością 7 dni)
2. SARIMA dla kroczącego okna czasowego
3. Wybór najlepszego modelu SARIMA
4. Wykres najlepszego modelu
5. Dokumentacja Kodu

### LSTM:
1. Instalacja tensorflow 2.0
2. Przygotowanie danych do sieci
3. Wybór różnych okien czasowych do sieci (7, 14, 28, 90 dni)
4. Implemetnacja sieci
5. Trening sieci
6. Ewaluacja wyników


### SARIMAX:
1. Korelacja kontraktów tygodniowych, miesiecznych
2. Normalizacja? 
3. Uzupelnienie pustych wierszy (ffill)
4. Sarimax z dodaniem zmiennych, wszystkie kombinacje?
5. Ewaluacja

### LSTM 2:
1. Modyfikacja sieci
2. Trening
3. wyniki

