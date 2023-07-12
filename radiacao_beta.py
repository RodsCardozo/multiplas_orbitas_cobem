from propagador_orbital import propagador_orbital
from radiacao_incidente import calor_incidente
from vetor_solar import *
from periodo_orbital import periodo_orbital
from datetime import datetime, timedelta
import numpy as np
import os, sys
import pandas as pd
# variacao da data
raan = [-47.4274401335861, -227.65171264121352, -232.39445665457214,-241.87994468128937, -251.3654327080066, -260.85092073472384,
        -275.0791527747997, -554.9010495629576, -564.3865375896747, -578.6147696297504,-588.1002576564675,-872.6648984579812, -877.4076424713397]

beta = [1.3154168751436284, 6.950829064406996,11.283229540520566,19.685633053572484,27.37301517070359, 34.15023582791156,
        41.641791649097826, 47.1453914151243, 55.35820777219794, 65.00697296978785,67.73304757643822,70.15663485451473, 72.7266789331106]

Data = ['2023-01-11 00:00:00', '2023-02-18 00:00:00', '2023-02-19 00:00:00', '2023-02-21 00:00:00', '2023-02-23 00:00:00',
        '2023-02-25 00:00:00', '2023-02-28 00:00:00', '2023-04-28 00:00:00', '2023-04-30 00:00:00','2023-05-05 00:00:00',
        '2023-05-03 00:00:00','2023-07-04 00:00:00',
        '2023-07-05 00:00:00']

def raan_beta(dataframe, lista_componente, componente, plot):
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
    df = dataframe
    linhas = px.line(df2,x=df['Beta'], y=lista_componente)
    linhas.update_layout(showlegend = True,
                         legend = dict(bgcolor = 'rgb(250,250,250)',
                                       bordercolor = 'rgb(130, 130, 130)',
                                       borderwidth = 1.0,
                                       itemdoubleclick = "toggleothers",
                                       title_text = 'Value'
                                       ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.5)',
                             gridwidth=1,
                     griddash='dot',
                             tickfont=dict(size=16),
                             title_font=dict(size=18),
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.5)',
                             gridwidth=1,
                     griddash='dot',
                             tickfont=dict(size=16),
                             title_font=dict(size=18),
                         ),
                         plot_bgcolor = 'rgb(255, 255, 255)',
                         xaxis_title='Beta',
                         yaxis_title='Radiation [W/m^2]')
    linhas.update_layout(
        plot_bgcolor='white',  # Cor de fundo do gráfico
        paper_bgcolor='white',  # Cor de fundo do papel
        xaxis=dict(
            linecolor='rgb(190, 190, 190)',  # Cor da borda do eixo x
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            linecolor='rgb(190, 190, 190)',  # Cor da borda do eixo y
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
    )
    linhas.update_layout(
        xaxis=dict(
            showline=True,  # Exibir borda do eixo x
            mirror=True,  # Refletir a linha de grade na borda do eixo x
        ),
        yaxis=dict(
            showline=True,  # Exibir borda do eixo y
            mirror=True,  # Refletir a linha de grade na borda do eixo y
        ),
    )

    linhas.update_layout(
        xaxis=dict(
            linewidth=2,  # Espessura da borda do eixo x
        ),
        yaxis=dict(
            linewidth=2,  # Espessura da borda do eixo y
        ),
    )
    linhas.update_layout(
        height=600,  # Altura da figura em pixels
        width=800,  # Largura da figura em pixels
    )
    if plot == 1:
        linhas.show()
    elif plot == 2:
        pass
    return linhas
'''for i in range(0, len(raan)):
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    # cria o folder para cada caso
    createFolder(f'radiacao_beta/beta_{beta[i]}')
    createFolder(f'radiacao_beta/beta_{beta[i]}/propagador')
    createFolder(f'radiacao_beta/beta_{beta[i]}/calor')
    SMA = 6871.0
    ecc = 0.002
    Raan = raan[i] #float(df.iloc[2, 0])  # ascencao direita do nodo ascendente
    arg_per = 0.0  # argumento do perigeu
    true_anomaly = 0.0  # anomalia verdadeira
    inc = 51.63
    mu = 398600.0  # constante gravitacional da terra
    J2 = 1.08263e-3  # zona harmonica 2
    Raio_terra = 6371.0  # raio da terra
    num_orbita = 1  # numero de obitas
    rp = SMA * (1 - ecc)
    T_orbita = periodo_orbital(SMA)

    PSIP = 0.0
    TETAP = 0.0
    PHIP = (2 * np.pi) / T_orbita
    phi0 = raan[i] #float(df.iloc[14, 0])
    teta0 = inc #float(df.iloc[15, 0])
    psi0 = true_anomaly #float(df.iloc[16, 0])
    input_string = Data[i] # df.iloc[18, 0]
    data = datetime.strptime(input_string, "%Y-%m-%d %H:%M:%S")
    delt = 1.0

    massa = 3.0  # massa do cubesat
    largura = 0.1  # comprimento do sat
    comprimento = 0.1  # largura do sat
    altura = 0.2  # altura do sat

    # Intensidade radiante do sol e terra e valores de emissividade

    Is = 1367.0  # radiacao solar
    Ir = 267.0  # radiacao IR Terra
    e = 1.0  # emissividade Terra
    ai = 1.0  # absortividade do satelite
    gama = 0.3  # refletividade da Terra
    import os.path
    inicio = datetime.now()

    # simulacao orbital
    Propagacao_orbital = propagador_orbital(data, SMA, ecc, Raan, arg_per, true_anomaly, inc, num_orbita, delt, psi0,
                                            teta0,
                                            phi0, PSIP, TETAP, PHIP, massa, largura, comprimento, altura, 1, beta[i])
    Propagacao_orbital.to_csv((f'radiacao_beta/beta_{beta[i]}/propagador/dados_ECI_{i}.csv'), sep=',')
    # simulacao radiacao
    calor_total = calor_incidente(Propagacao_orbital, Is, Ir, e, ai, gama, data, 1, inc, beta[i])

    calor_total.to_csv((f'radiacao_beta/beta_{beta[i]}/calor/Calor_Incidente_{beta[i]}.csv'), sep=',')
    fim = datetime.now()
    var = fim - inicio

    # print dados entrada
    tempo = [inicio, fim, var]
    dados_orbita = [SMA, ecc, Raan, arg_per, true_anomaly, inc, num_orbita, delt]

    from vetor_solar import beta_angle

    print(
        '|----------------| ' \
        '\n' \
        '| Nova simulacao | ' \
        '\n' \
        '|----------------| ' \
        '\n' \
        f'Tempo de inicio: {tempo[0]} ' \
        f'\n' \
        f'Tempo final: {tempo[1]} ' \
        f'\n' \
        f'Tempo total de simulacao: {tempo[2]} ' \
        f'\n' \
        f'|------------------| ' \
        f'\n' \
        '| Dados da orbita: | ' \
        '\n' \
        '|------------------| ' \
        '\n' \
        f'Semi eixo = {dados_orbita[0]} ' \
        f'\n' \
        f'Excentricidade = {dados_orbita[1]} ' \
        f'\n' \
        f'Raan = {dados_orbita[2]} ' \
        f'\n' \
        f'Argumento do perigeu = {dados_orbita[3]} ' \
        f'\n' \
        f'Anomalia verdadeira = {dados_orbita[4]} ' \
        f'\n' \
        f'Inclinação = {dados_orbita[5]} ' \
        f'\n' \
        f'Numero de orbitas = {dados_orbita[6]} ' \
        f'\n' \
        f'Passo de integraçao = {dados_orbita[7]} ' \
        f'\n'
        f'Angulo Beta: {np.degrees(beta_angle(data, inc, Raan))}'
        f'\n'
        f'Data simulada de propagação: {data}'
        f'\n'
        f'Numero de Orbitas simuladas: {num_orbita}'
    )'''

maximo = []
minimo = []
media = []
for i in range(0, len(beta)):
    df = pd.read_csv(f'radiacao_beta/beta_{beta[i]}/calor/Calor_Incidente_{beta[i]}.csv', sep=',')
    df['Total'] = df['Total 1'] + df['Total 2'] + df['Total 3'] + df['Total 4'] + df['Total 5'] + df['Total 6']
    maximo.append(df['Total'].max())
    minimo.append(df['Total'].min())
    media.append((df['Total'].mean()))
df2 = pd.DataFrame()
df2['Maximum'] = maximo
df2['Minimum'] = minimo
df2['Mean'] = media
df2['Beta'] = beta
df2.to_csv('raan_beta.csv', sep=',', index=False)
beta = raan_beta(df2, ['Maximum', 'Minimum', 'Mean'], 'Radiacao',2)
import plotly.io as pio
pio.write_image(beta, 'raan_beta.png')