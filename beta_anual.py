from vetor_solar import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# deterinação do beta mensal

# variacao do mes
data = '01/01/2023 00:00:00'
data_inicio = datetime.strptime(data, '%m/%d/%Y %H:%M:%S')
meses = [data_inicio + timedelta(days=x) for x in range(0, 365, 1)]
Meses = [nome_mes(mes.month) for mes in meses]

# inclinacao da orbita [graus]

inclinacao = 51.63

# taxa de precessao mensal [graus/s]

omega_p = taxa_precessao(0.0, 6871.0, inclinacao)*(180/np.pi)

# variacao do Raan

Raan0 = 0.0

Raan_mes = []

Raan_mes.append(Raan0)

for i in range(1,365):
    a = Raan_mes[i-1] + omega_p * (23*3600 + 56*60 + 4)
    Raan_mes.append(a)

# calculo do beta anual

beta = [beta_angle(dia, inclinacao, raan)*(180/np.pi) for dia,raan in zip(meses, Raan_mes)]

# plotagem do beta mensal

df = pd.DataFrame()
df['Beta'] = beta
df['Date'] = meses
df['temp_color'] = np.arange(len(df))
df['Raan'] = (Raan_mes)
df['Month'] = [f'{nome_mes(int(mes.month))}' + ' ' + f'{mes.day}' for mes in meses]
df.to_csv('beta_mes.csv', sep=',', index=False)
import plotly.express as px
fig = px.line(df, y="Beta", color_discrete_sequence=['#3CB371'])
#fig.update_traces(textposition="bottom left")
fig.update_layout(showlegend=True,
                  legend=dict(bgcolor='rgb(250, 250, 250)',
                                 bordercolor='rgb(120, 120, 120)',
                                 borderwidth=1.0,
                                 itemdoubleclick="toggleothers",
                                 title_text='Anual Beta Variation'
                                 ),
                     xaxis=dict(
                         showgrid=True,
                         gridcolor='rgba(100, 100, 100, 0.5)',
                         gridwidth=1,
                         title_font=dict(size=24),
                         tickfont=dict(size=16),

                     griddash='dot'
                     ),
                     yaxis=dict(
                         showgrid=True,
                         gridcolor='rgba(100, 100, 100, 0.5)',
                         gridwidth=1,
                         title_font=dict(size=24),
                         tickfont=dict(size=16),
                     griddash='dot'
                     ),
                     autosize=True,
                     plot_bgcolor='rgb(255, 255, 255)',
                     xaxis_title='Day',
                     yaxis_title='Beta')
fig.update_layout(
    plot_bgcolor='white',  # Cor de fundo do gráfico
    paper_bgcolor='white',  # Cor de fundo do papel
    xaxis=dict(
        linecolor='rgb(190, 190, 190)',  # Cor da borda do eixo x
    ),
    yaxis=dict(
        linecolor='rgb(190, 190, 190)',  # Cor da borda do eixo y
    ),
)
fig.update_layout(
    xaxis=dict(
        showline=True,  # Exibir borda do eixo x
        mirror=True,  # Refletir a linha de grade na borda do eixo x
    ),
    yaxis=dict(
        showline=True,  # Exibir borda do eixo y
        mirror=True,  # Refletir a linha de grade na borda do eixo y
    ),
)

fig.update_layout(
    xaxis=dict(
        linewidth=2,  # Espessura da borda do eixo x
    ),
    yaxis=dict(
        linewidth=2,  # Espessura da borda do eixo y
    ),
)
fig.update_layout(height=600, width=1000)
import plotly.io as pio
pio.write_image(fig, 'Beta_mensal.png')
