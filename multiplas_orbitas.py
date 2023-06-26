from vetor_solar import *
import pandas as pd
import os,sys
from datetime import datetime, timedelta
from periodo_orbital import periodo_orbital
import numpy as np
from plots import calor_total,calor_solar,calor_albedo,calor_IR_Terra
def beta_mes(inicio, inc):
    """
    :param inicio: Data de inicio para beta mensal
    :param inc: inclinacao da orbita
    :return: RAAN para que satisfaça o beta calculado
    """
    from numpy import degrees, pi
    import plotly.express as px
    dia_ini = inicio
    ini_date = datetime.strptime(dia_ini, "%m/%d/%Y %H:%M:%S")
    data = [ini_date + timedelta(days=x) for x in range(0, 365, 30)]
    inc = inc
    beta = [beta_angle(dia, inc, 0.0) for dia in data]
    data2 = [str(dado) for dado in data]
    df = pd.DataFrame(degrees(beta), columns=['Beta'])
    df['data'] = data2
    df['inc'] = inc
    df['mes'] = [nome_mes(dado.month) + ' ' + str(dado.day) for dado in data]
    raan = [float(beta_raan(beta, data, inc) * (180 / pi)) for beta, data in zip(beta, data)]
    df['raan'] = raan
    fig = px.scatter(df, y="Beta", text="mes")
    fig.update_traces(textposition="bottom left")
    fig.show()
    return raan, df, data

raan, df, data = beta_mes('01/01/2023 12:00:00', 98.0)

from propagador_orbital import propagador_orbital as po
from radiacao_incidente import calor_incidente as ci
from tqdm import tqdm
for i in tqdm(range(0, len(raan))):
    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    # cria o folder para cada caso
    createFolder(f'./results_{i}/')
    createFolder(f'./results_{i}/radiacao_{i}/graficos_{i}')
    def resource_path(relative_path):

        try:
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)

    df = pd.read_csv(resource_path("Condicoes Iniciais\dados_entrada.csv"), sep='=', engine='python',
                     on_bad_lines='skip')
    SMA = float(df.iloc[0, 0])
    a = float(df.iloc[1, 0])  # excentricidade da orbita
    if a < 0.002:
        ecc = 0.002
    else:
        ecc = a
    Raan = raan #float(df.iloc[2, 0])  # ascencao direita do nodo ascendente
    arg_per = (float(df.iloc[3, 0]))  # argumento do perigeu
    true_anomaly = (float(df.iloc[4, 0]))  # anomalia verdadeira
    b = (float(df.iloc[5, 0]))  # inclinacao
    if b < 0.01:
        inc = 0.1
    else:
        inc = b
    mu = float(df.iloc[6, 0])  # constante gravitacional da terra
    J2 = float(df.iloc[7, 0])  # zona harmonica 2
    Raio_terra = float(df.iloc[8, 0])  # raio da terra
    num_orbita = int(df.iloc[9, 0])  # numero de obitas
    rp = SMA * (1 - ecc)
    T_orbita = periodo_orbital(SMA)

    PSIP = float(df.iloc[11, 0])
    TETAP = float(df.iloc[12, 0])
    PHIP = (2 * np.pi) / T_orbita
    phi0 = Raan[i]#float(df.iloc[14, 0])
    teta0 = inc#float(df.iloc[15, 0])
    psi0 = true_anomaly #float(df.iloc[16, 0])
    input_string = df.iloc[18, 0]
    data = data #datetime.strptime(input_string, " %m/%d/%Y %H:%M:%S")
    delt = float(df.iloc[19, 0])

    massa = float(df.iloc[21, 0])  # massa do cubesat
    largura = float(df.iloc[22, 0])  # comprimento do sat
    comprimento = float(df.iloc[23, 0])  # largura do sat
    altura = float(df.iloc[24, 0])  # altura do sat

    # Intensidade radiante do sol e terra e valores de emissividade

    Is = float(df.iloc[26, 0])  # radiacao solar
    Ir = float(df.iloc[27, 0])  # radiacao IR Terra
    e = float(df.iloc[28, 0])  # emissividade Terra
    ai = float(df.iloc[29, 0])  # absortividade do satelite
    gama = float(df.iloc[30, 0])  # refletividade da Terra
    from time import time

    inicio = datetime.now()

    orbita = po(data[i], SMA, ecc, Raan[i], arg_per, true_anomaly, inc, num_orbita, delt, psi0, teta0,
                                        phi0, PSIP, TETAP, PHIP, massa, largura, comprimento, altura, i)

    radiacao = ci(orbita, Is, Ir, e, ai, gama, data[i], i)

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
        f'| Dados da orbita: n° {i} | ' \
        '\n' \
        '|------------------| ' \
        '\n' \
        f'Semi eixo = {dados_orbita[0]} ' \
        f'\n' \
        f'Excentricidade = {dados_orbita[1]} ' \
        f'\n' \
        f'Raan = {raan[i]} ' \
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
        f'Angulo Beta: {np.degrees(beta_angle(data[i], inc, Raan[i]))}'
        f'\n'
        f'Data simulada de propagação: {data[i]}'
        f'\n'
        f'Numero de Orbitas simuladas: {num_orbita}'
    )
    a = '|----------------| ' \
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
        f'| Dados da orbita: n° {i} | ' \
        '\n' \
        '|------------------| ' \
        '\n' \
        f'Semi eixo = {dados_orbita[0]} ' \
        f'\n' \
        f'Excentricidade = {dados_orbita[1]} ' \
        f'\n' \
        f'Raan = {raan[i]} ' \
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
        f'\n'\
        f'Angulo Beta: {np.degrees(beta_angle(data[i], inc, Raan[i]))}'\
        f'\n'\
        f'Data simulada de propagação: {data[i]}'\
        f'\n'\
        f'Numero de Orbitas simuladas: {num_orbita}'
    with open(f'./results_{i}/log_{i}.txt', "w") as arquivo:
    # Escreve a mensagem no arquivo
        arquivo.write(a)
    arquivo.close()
    import plotly.io as pio
    linhas1 = calor_solar(radiacao,2)
    pio.write_image(linhas1, f'./results_{i}/radiacao_{i}/graficos_{i}/radiacao_solar_{i}.png')

    linhas2 = calor_albedo(radiacao, 2)
    pio.write_image(linhas2, f'./results_{i}/radiacao_{i}/graficos_{i}/radiacao_albedo_{i}.png')

    linhas3 = calor_IR_Terra(radiacao, 2)
    pio.write_image(linhas3, f'./results_{i}/radiacao_{i}/graficos_{i}/radiacao_IR_Terra_{i}.png')

    linhas4 = calor_total(radiacao, 2)
    pio.write_image(linhas4, f'./results_{i}/radiacao_{i}/graficos_{i}/radiacao_total_{i}.png')


