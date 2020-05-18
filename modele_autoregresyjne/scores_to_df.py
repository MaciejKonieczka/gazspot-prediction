import pandas as pd

df_scores = pd.read_csv("Documentation/sarima_hypertuninglog.txt", header=None, sep=":", names=['key', 'score'])
df_scores['score'] = pd.to_numeric(df_scores['score'].str.replace('RMSE=',''), errors='coerce')

df_scores.sort_values('score').head()

p_params = []
d_params = []
q_params = []
t_params = []
P_params = []
D_params = []
Q_params = []  
m_params = []

for i, r in pd.DataFrame(df_scores['key'].str.replace('(','').str.replace(')','').str.split(',')).iterrows():
    p_params.append(r[0][0])
    d_params.append(r[0][1])
    q_params.append(r[0][2])
    P_params.append(r[0][3])
    D_params.append(r[0][4])
    Q_params.append(r[0][5])  
    m_params.append(r[0][6])
    t_params.append(r[0][7])


df_scores['p_params'] = p_params
df_scores['d_params'] = d_params
df_scores['q_params'] = q_params
df_scores['P_params'] = P_params
df_scores['D_params'] = D_params
df_scores['Q_params'] = Q_params
df_scores['m_params'] = m_params
df_scores['t_params'] = t_params


df_scores = df_scores.drop(columns=['key']).sort_values('score')

for col_ in ['p_params',
    'd_params',
    'q_params',
    'P_params',
    'D_params',
    'Q_params',
    'm_params']:
    df_scores[col_] = df_scores[col_].astype(int)

df_scores['t_params'] = None

df_scores.to_pickle('Documentation/sarima_hypertuning.p')

df = pd.read_pickle('Documentation/sarima_hypertuning.p')

df[ df['D_params'] == 0 ].head(10)

#