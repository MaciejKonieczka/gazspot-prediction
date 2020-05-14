import pandas as pd
import matplotlib.pyplot as plt

spot_df = pd.read_pickle('data/tge_spot.p')

# df = pd.read_csv('data/RTT_GS_CONTRACTS_REPORT.csv', sep=';', skiprows=2)
# df.sample()

spot_df[['Price']].plot()
plt.show()