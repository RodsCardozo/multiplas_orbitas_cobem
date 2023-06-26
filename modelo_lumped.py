"""primeira versáo de um modelo lumped para cubesats

Teste 1 - Criando um modelo de caixa com radiação interna

comparação com o artigo de Garzón et al.

dados do material:

      Alumínio Polido - tabela 12-3 Çengel

      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0 kg/m^3
      'condutividade térmica': 200 J/m.K
      'calor especifico': 920.0 J/kg.K

"""
import gc

"""
Primeira etapa: criação dos nós

"""
from datetime import datetime
inicio = datetime.now()
import fator_forma_interno as ff
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import pandas as pd
import gc
import tracemalloc

lump = pd.DataFrame()
A1 = {'nome': 1,
      'x': 5.0,
      'y': 0.0,
      'z': 0.0,
      'Lx': 0.02,
      'Ly': 0.10,
      'Lz': 0.30,
      'n': 'i',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump = pd.DataFrame(A1, index=[0])
A2 = {'nome': 2,
      'x': 0.0,
      'y': 5.0,
      'z': 0.0,
      'Lx': 0.10,
      'Ly': 0.02,
      'Lz': 0.30,
      'n': 'j',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump.loc[len(lump)] = A2
# print(lump)
A3 = {'nome': 3,
      'x': -5.0,
      'y': 0.0,
      'z': 0.0,
      'Lx': 0.02,
      'Ly': 0.10,
      'Lz': 0.30,
      'n': 'i',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump.loc[len(lump)] = A3
# print(lump)
A4 = {'nome': 4,
      'x': 0.0,
      'y': -5.0,
      'z': 0.0,
      'Lx': 0.10,
      'Ly': 0.02,
      'Lz': 0.30,
      'n': 'j',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump.loc[len(lump)] = A4
# print(lump)
A5 = {'nome': 5,
      'x': 0.0,
      'y': 0.0,
      'z': -15.0,
      'Lx': 0.100,
      'Ly': 0.100,
      'Lz': 0.02,
      'n': 'k',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump.loc[len(lump)] = A5
# print(lump)
A6 = {'nome': 6,
      'x': 0.0,
      'y': 0.0,
      'z': 15.0,
      'Lx': 0.10,
      'Ly': 0.10,
      'Lz': 0.02,
      'n': 'k',
      'emissividade': 0.84,
      'absortividade': 0.14,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0
      }
lump.loc[len(lump)] = A6

"""
Segunda etapa: junção das interfaces. Designar quem está tocando quem e quem está vendo quem. 
"""

""" Fator de forma interno para cavidades quadradas """


def fator_forma(A, B, tipo_fator_forma, **kwargs):
    import fator_forma_interno as ff
    tipo = str(tipo_fator_forma)
    if tipo.lower() == 'paralelo':
        Fij = ff.fator_paralelo_dict(A, B, kwargs['X'], kwargs['Y'])
    elif tipo.lower() == 'perpendicular':
        Fij = ff.fator_perpendicular_dict(A, B, kwargs['X'], kwargs['Y'], kwargs['Z'])
    return Fij


def ff_interno(df):
    """
      :param df: DataFrame with nodes
      :return: Array with all the Form Factors for a square cavity
      """
    lump = df
    N = len(df)  # número de nós
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # A = np.dot(lump.iloc[i,7], lump.iloc[j,7])
            # print(A)
            if i == j:
                M[i, j] = 0

            elif lump.iloc[i, 7] == lump.iloc[j, 7]:
                if lump.iloc[i, 7] == 'i':
                    M[i, j] = ff.fator_paralelo_dict(A=lump.iloc[i], B=lump.iloc[j], X=lump.iloc[i, 5],
                                                     Y=lump.iloc[i, 6])
                elif lump.iloc[i, 7] == 'j':
                    M[i, j] = ff.fator_paralelo_dict(A=lump.iloc[i], B=lump.iloc[j], X=lump.iloc[i, 4],
                                                     Y=lump.iloc[i, 6])
                else:
                    M[i, j] = ff.fator_paralelo_dict(A=lump.iloc[i], B=lump.iloc[j], X=lump.iloc[i, 4],
                                                     Y=lump.iloc[i, 5])
            else:
                if lump.iloc[i, 7] == 'i' and lump.iloc[j, 7] == 'k':
                    X = lump.iloc[i, 5]
                    Y = lump.iloc[i, 6]
                    Z = lump.iloc[j, 4]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

                elif lump.iloc[i, 7] == 'i' and lump.iloc[j, 7] == 'j':
                    X = lump.iloc[i, 6]
                    Y = lump.iloc[i, 5]
                    Z = lump.iloc[j, 4]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

                elif lump.iloc[i, 7] == 'j' and lump.iloc[j, 7] == 'i':
                    X = lump.iloc[i, 6]
                    Y = lump.iloc[i, 4]
                    Z = lump.iloc[j, 5]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

                elif lump.iloc[i, 7] == 'j' and lump.iloc[j, 7] == 'k':
                    X = lump.iloc[i, 4]
                    Y = lump.iloc[i, 6]
                    Z = lump.iloc[j, 5]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

                elif lump.iloc[i, 7] == 'k' and lump.iloc[j, 7] == 'i':
                    X = lump.iloc[i, 5]
                    Y = lump.iloc[i, 4]
                    Z = lump.iloc[j, 6]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

                elif lump.iloc[i, 7] == 'k' and lump.iloc[j, 7] == 'j':
                    X = lump.iloc[i, 4]
                    Y = lump.iloc[i, 5]
                    Z = lump.iloc[j, 6]
                    M[i, j] = ff.fator_perpendicular_dict(A=lump.iloc[i], B=lump.iloc[j], X=X, Y=Y, Z=Z)

    return M


""" Fator de Gebhart """


def fator_gebhart(df, F):
    """
      :param df: Dataframe with nodes
      :param F: Array with all Form Factors
      :return B: Array with Gebhart factors
      """
    F = ff_interno(lump)
    N = len(F)
    M = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i == k:
                d = 1
            else:
                d = 0
            M[i, k] = (1 - lump.iloc[k, 8]) * F[i, k] - d
    b = []
    for j in range(N):
        b.append(-lump.iloc[j, 8] * F[j])
    B = []
    for i in range(N):
        B.append(np.linalg.solve(M, b[i]))

    return np.array(B)


areas_rad = {'A1': 0.1 * 0.3,
             'A2': 0.1 * 0.3,
             'A3': 0.1 * 0.3,
             'A4': 0.1 * 0.3,
             'A5': 0.1 * 0.1,
             'A6': 0.1 * 0.1
             }

area_rad = list(areas_rad.values())
""" Conexão entre nós """

""" Matriz de condução """

conduc = {'1': [2, 4, 5, 6],
          '2': [1, 3, 5, 6],
          '3': [2, 4, 5, 6],
          '4': [1, 3, 5, 6],
          '5': [1, 2, 3, 4],
          '6': [1, 2, 3, 4]
          }
lista_nomes = list(conduc.keys())


K = pd.DataFrame(conduc)
'''print(len(conduc))'''
# Define a matriz inicial preenchida por zeros
K_m = np.zeros((len(conduc), len(conduc)))
'''print(K_m)
print(K.columns[0])
print(K.columns.get_loc('1'))
print(set(K.iloc[:,K.columns.get_loc('1')]))'''

# Lógica para preencher a matriz com 1 onde existe condução entre nós

for i in range(0, len(conduc)):

    A = K.columns[i]  # Recebe a posicao da coluna dada o valor de i
    B = set(K.iloc[:, K.columns.get_loc(lista_nomes[i])])  # Recebe o valor dentro do DF dada

    for j in range(0, len(conduc)):
        if i == j:
            K_m[i, j] = 0
        elif j + 1 in B:
            K_m[i, j] = 1

# Calculo das áreas de condução e comprimento

areas = {'1 2': ['Lx', 'Lz'],
         '1 4': ['Lx', 'Lz'],
         '1 5': ['Ly', 'Lx'],
         '1 6': ['Ly', 'Lx'],
         '2 1': ['Ly', 'Lz'],
         '2 3': ['Ly', 'Lz'],
         '2 5': ['Lx', 'Ly'],
         '2 6': ['Lx', 'Ly'],
         '3 2': ['Lx', 'Lz'],
         '3 4': ['Lx', 'Lz'],
         '3 5': ['Ly', 'Lx'],
         '3 6': ['Ly', 'Lx'],
         '4 1': ['Ly', 'Lz'],
         '4 3': ['Ly', 'Lz'],
         '4 5': ['Lx', 'Ly'],
         '4 6': ['Lx', 'Ly'],
         '5 1': 2.0,
         '5 2': 2.0,
         '5 3': 2.0,
         '5 4': 2.0,
         '6 1': 2.0,
         '6 2': 2.0,
         '6 3': 2.0,
         '6 4': 2.0,
         }

# Comprimento das distâncias entre os nós

L = {'1 2': 0.10 / 2.0,
     '1 4': 0.10 / 2.0,
     '1 5': 0.3 / 2,
     '1 6': 0.3 / 2,
     '2 1': 0.10 / 2.0,
     '2 3': 0.10 / 2.0,
     '2 5': 0.3 / 2,
     '2 6': 0.3 / 2,
     '3 2': 0.10 / 2.0,
     '3 4': 0.10 / 2.0,
     '3 5': 0.3 / 2,
     '3 6': 0.3 / 2,
     '4 1': 0.10 / 2.0,
     '4 3': 0.10 / 2.0,
     '4 5': 0.3 / 2,
     '4 6': 0.3 / 2,
     '5 1': 0.10 / 2.0,
     '5 2': 0.10 / 2.0,
     '5 3': 0.10 / 2.0,
     '5 4': 0.10 / 2.0,
     '6 1': 0.10 / 2.0,
     '6 2': 0.10 / 2.0,
     '6 3': 0.10 / 2.0,
     '6 4': 0.10 / 2.0,
     }

# Calculando a conductancia entre faces

lista_nomes = list(areas.keys())
chaves = list(areas.values())
comprimento = list(L.values())

K = 0
for nome in lista_nomes:
    A = nome.split()
    i, j = A
    i = int(i)
    j = int(j)
    b = chaves[K]
    if type(b) == list:
        k = (lump.iloc[i - 1, lump.columns.get_loc('condutividade térmica')] *
             lump.iloc[i - 1, lump.columns.get_loc(b[0])] *
             lump.iloc[i - 1, lump.columns.get_loc(b[1])]) / comprimento[K]
    else:
        k = (b * lump.iloc[i - 1, lump.columns.get_loc('condutividade térmica')]) / comprimento[K]
    K_m[i - 1, j - 1] = k
    K += 1
# matriz de gebhart

F = ff_interno(lump)
fg = fator_gebhart(lump, F)

# Calculando a variação de temperatura na órbita

# importar as temperaturas da orbita
calor = pd.read_csv("results/Calor_Incidente.csv", sep=',')
# calor_total_face1 = calor.iloc[:, calor.columns.get_loc('Total 1')].tolist()
calor_lump = pd.DataFrame()
'''calor_lump['Face 1'] = calor['Total 1']
calor_lump['Face 2'] = calor['Total 2']
calor_lump['Face 3'] = calor['Total 3']
calor_lump['Face 4'] = calor['Total 4']
calor_lump['Face 5'] = calor['Total 5']
calor_lump['Face 6'] = calor['Total 6']'''

calor_lump1 = np.array(calor['Total 1'])
calor_lump2 = np.array(calor['Total 2'])
calor_lump3 = np.array(calor['Total 3'])
calor_lump4 = np.array(calor['Total 4'])
calor_lump5 = np.array(calor['Total 5'])
calor_lump6 = np.array(calor['Total 6'])



calor = np.array([calor_lump1, calor_lump2, calor_lump3, calor_lump4, calor_lump5, calor_lump6])
print(len(calor[0]))
from memory_profiler import profile

def lumped(calor, fg, area_rad):
    # Temperaturas dos nós iniciais que serão alteradas a cada passo de integração
    Temp_ini = [273.0, 273.0, 273.0, 273.0, 273.0, 273.0]
    Temp_pos = [273.0, 273.0, 273.0, 273.0, 273.0, 273.0]
    Dt_termico = 0.01  # passo de integracao termica
    Dt_orbita = 1.0  # passo de integracao da orbita
    T_externa = 3.0  # Temperatura de fundo do espaço
    # constante de steffan boltzmann
    sig = 5.67e-8
    cores = ['#5F9EA0', '#66CDAA', '#7FFFD4', '#006400', '#556B2F', '#8FBC8F', '#2E8B57', '#3CB371', '#20B2AA', '#98FB98',
             '#00FF7F', '#7CFC00', '#00FF00', '#7FFF00', '#00FA9A', '#ADFF2F', '#32CD32', '#9ACD32', '#228B22', '#6B8E23',
             '#BDB76B','#FF6347','#FF4500','#FF0000','#FF69B4','#FF1493','#FFC0CB','#FFB6C1','#DB7093',
             '#B03060','#C71585','#D02090','#FF00FF','#EE82EE','#DDA0DD','#DA70D6','#BA55D3',
             '#9932CC','#9400D3','#8A2BE2','#A020F0','#9370DB','#D8BFD8','#FFFAFA']
    total = len(calor[0])
    cont = 0
    n = 0
    numero_orbitas = 26
    # calculo da temperaturas
    var_temp = []
    #tracemalloc.start()

    while n < numero_orbitas:
        for k in tqdm(range(0, total, 1), colour=cores[n]):
            while cont < Dt_orbita:
                # passo orbital
                Tcond = np.zeros(6)
                Trad = np.zeros(6)
                alfa = 0.8
                emi = 0.8
                for i in range(0, 6):
                    for j in range(0, 6):
                        # Preencher o vetor conducao
                        '''Tcond[j] = (K_m[i, j] * (Temp_ini[j] - Temp_ini[i]))

                        # Preencher o vetor radiacao
                        Trad[j] = (sig * lump.iloc[i, lump.columns.get_loc('emissividade')] * area_rad[i] * fg[i, j] * (
                                    Temp_ini[j] ** 4 - Temp_ini[i] ** 4))
                        C_i = lump.iloc[i, lump.columns.get_loc('densidade')] * \
                          lump.iloc[i, lump.columns.get_loc('calor especifico')] * \
                          lump.iloc[i, lump.columns.get_loc('Lx')] * \
                          lump.iloc[i, lump.columns.get_loc('Ly')] * \
                          lump.iloc[i, lump.columns.get_loc('Lz')]
                          
                          'emissividade': 0.05,
                          'absortividade': 0.09,
                          'densidade': 2700.0,
                          'condutividade térmica': 200,
                          'calor especifico': 920.0
                          '''

                        Tcond[j] = (K_m[i, j] * (Temp_ini[j] - Temp_ini[i]))

                        # Preencher o vetor radiacao
                        Trad[j] = (sig * emi * area_rad[i] * fg[i, j] * (
                                    Temp_ini[j] ** 4 - Temp_ini[i] ** 4))

                    # Capacitancia termica
                    rho = 2700.0
                    cp = 920.0
                    C_i = rho * cp * 0.10 * 0.10 * 0.02
                    Temp_pos[i] = Temp_ini[i] + (Dt_termico / C_i) * (sum(Tcond) + sum(Trad) +
                                                                      area_rad[i] * calor[i, k] * (alfa)
                                                                      - sig * area_rad[i] * emi * (Temp_ini[i] ** 4))
                    del C_i
                    del Tcond
                    del Trad
                    Tcond = np.zeros(6)
                    Trad = np.zeros(6)
                del Temp_ini
                Temp_ini = Temp_pos
                a = Temp_pos.copy()
                var_temp.append(a)
                del a
                del Temp_pos
                Temp_pos = [273.0, 273.0, 273.0, 273.0, 273.0, 273.0]
                cont += Dt_termico
            cont = 0
        gc.collect()


        '''df = pd.DataFrame(var_temp, columns=['Face1', 'Face2', 'Face3', 'Face4', 'Face5', 'Face6'])
        df.to_csv(f'temperaturas_transiente_2/resultado_temp_{n}.csv', sep=',', index=False)'''
        import csv

        lista_aninhada = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # Exemplo de lista aninhada

        with open(f'temperaturas_transiente_3/resultado_temp_{n}.csv', 'w', newline='') as arquivo_csv:
            writer = csv.writer(arquivo_csv)
            writer.writerow(['Face1', 'Face2', 'Face3', 'Face4', 'Face5', 'Face6'])  # Escreve o cabeçalho do CSV, se necessário

            for lista in var_temp:
                writer.writerow(lista)  # Escreve cada lista como uma linha no CSV

        arquivo_csv.close()  # Fecha o arquivo após a conclusão das operações de escrita

        del var_temp
        #del df
        '''snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for stat in top_stats:
            print(stat)
        tracemalloc.stop()'''
        gc.collect()
        var_temp = []
        print(f'Orbita numero: {n}')
        n += 1

lumped(calor, fg, area_rad)

fim = datetime.now()
var = fim - inicio
'''import plotly.io as pio
import plotly.express as px
fim = datetime.now()
var = fim - inicio

# print dados entrada
tempo = [inicio, fim, var]

Temp = df
linhas = px.line(Temp, y=['Face1','Face2','Face3','Face4','Face5','Face6'])
pio.write_image(linhas, 'imagens_resultado/temp_faces.png')'''

from bot_simulacao import bot_temperaturas

tempo = [inicio, fim, var]
bot_temperaturas(tempo)
