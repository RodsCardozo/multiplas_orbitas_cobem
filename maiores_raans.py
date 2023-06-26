import pandas as pd
from datetime import datetime, timedelta
import os, sys
dia = '01/01/2023 12:00:00'
data = datetime.strptime(dia, "%m/%d/%Y %H:%M:%S")
data = [data + timedelta(days=x) for x in range(0,365,30)]
inclinacao = 98.0
from tqdm import tqdm
maximo_beta = []
data_max_beta = []
for i in tqdm(range(0,12), colour='#3c32a8'):
    from vetor_solar import beta_angle
    df = pd.read_csv(f'./analise_raan/raan_mes_{i}/analise_beta_mes_{i}.csv', sep=',')
    date = data[i]
    time = [date + timedelta(seconds=x*10) for x in range(0,int(len(df)))]
    raan = list(df['raan'])
    from numpy import pi
    beta = [beta_angle(tempo, inclinacao, Raan)*(180/pi) for tempo,Raan in zip(time,raan)]
    df = pd.DataFrame()
    df['Beta'] = beta
    df['Data'] = time

    beta_maximo = df['Beta'].max()
    indice_maior = df['Beta'].idxmax()
    linha_maior = df.loc[indice_maior, :]
    maximo_beta.append(linha_maior[0])
    data_max_beta.append(linha_maior[1])

df = pd.DataFrame()
df['Maximo Beta'] = maximo_beta
df['Data'] = data_max_beta
from vetor_solar import beta_raan
beta2 = list(df['Maximo Beta'])
data2 = list(df['Data'])
df['raan'] = [beta_raan(beta, data, 98.0) for beta,data in zip(beta2, data2)]

import plotly.express as px
fig1 = px.line(df, y="Maximo Beta", text="Data")
fig1.update_traces(textposition="bottom right")
fig2 = px.line(df, y="Maximo Beta", text="Data")
fig2.update_traces(textposition="bottom right")
fig1.show()
fig2.show()
