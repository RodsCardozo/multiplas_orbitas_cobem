"""primeira versáo de um modelo lumped para cubesats

Teste 1 - Criando um modelo de caixa com radiação interna

comparação com o artigo de Garzón et al.

dados do material:

      Alumínio Polido - tabela 12-3 Çengel

      'emissividade': 0.05,
      'absortividade': 0.09,
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
from tqdm import tqdm
import pandas as pd
import gc

import os, sys

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
nome_arquivo = input(f'Insira o nome da pasta: ')
createFolder(nome_arquivo)
"""
Segunda etapa: junção das interfaces. Designar quem está tocando quem e quem está vendo quem. 
"""

""" Fator de forma interno para cavidades quadradas """


'''def fator_forma(A, B, tipo_fator_forma, **kwargs):
    import fator_forma_interno as ff
    tipo = str(tipo_fator_forma)
    if tipo.lower() == 'paralelo':
        Fij = ff.fator_paralelo_dict(A, B, kwargs['X'], kwargs['Y'])
    elif tipo.lower() == 'perpendicular':
        Fij = ff.fator_perpendicular_dict(A, B, kwargs['X'], kwargs['Y'], kwargs['Z'])
    return Fij'''


def ff_interno(face):
    """
      :param df: DataFrame with nodes
      :return: Array with all the Form Factors for a square cavity
      """

    N = len(faces)  # número de nós
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # A = np.dot(lump.iloc[i,7], lump.iloc[j,7])
            # print(A)
            if i == j:
                M[i, j] = 0

            elif face[i].n == face[j].n:
                if face[i].n == 'i':
                    M[i, j] = ff.fator_paralelo(X=face[i].Ly, Y=face[i].Lz, L=(face[i].Lx + face[i].Lx))
                elif face[i].n == 'j':
                    M[i, j] = ff.fator_paralelo(X=face[i].Lx, Y=face[i].Lz, L=(face[i].Ly + face[i].Ly))
                else:
                    M[i, j] = ff.fator_paralelo(X=face[i].Lx, Y=face[i].Ly, L=(face[i].Lz + face[i].Lz))
            else:
                if face[i].n == 'i' and face[j].n == 'k':
                    X = face[i].Ly
                    Y = face[i].Lz
                    Z = face[j].Lx
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'i' and face[j].n == 'j':
                    X = face[i].Lz
                    Y = face[i].Ly
                    Z = face[j].Lx
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'i':
                    X = face[i].Lz
                    Y = face[i].Lx
                    Z = face[j].Ly
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'k':
                    X = face[i].Lx
                    Y = face[i].Lz
                    Z = face[j].Ly
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'i':
                    X = face[i].Ly
                    Y = face[i].Lx
                    Z = face[j].Lz
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'j':
                    X = face[i].Lx
                    Y = face[i].Ly
                    Z = face[j].Lz
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

    return M


""" Fator de Gebhart """


def fator_gebhart(face, F):
    """
      :param df: Dataframe with nodes
      :param F: Array with all Form Factors
      :return B: Array with Gebhart factors
      """
    F = ff_interno(face)
    N = len(F)
    M = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i == k:
                d = 1
            else:
                d = 0
            M[i, k] = (1 - face[k].e) * F[i, k] - d
    b = []
    for j in range(N):
        b.append(-face[j].e * F[j])
    B = []
    for i in range(N):
        B.append(np.linalg.solve(M, b[i]))
    return np.array(B)

class Face:
    def __init__(self, nome, x, y, z, Lx, Ly, Lz, n, e, a, rho, k, cp):
        self.nome = nome
        self.x = x
        self.y = y
        self.z = z
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.n = n
        self.e = e
        self.a = a
        self.rho = rho
        self.k = k
        self.cp = cp

face1 = Face(nome = 'Face1', x = 5.0, y = 0.0, z = 0.0, Lx = 0.02, Ly = 0.10, Lz = 0.30, n =  'i' , e = 0.52, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

face2 = Face(nome = 'Face2', x = 0.0, y = 5.0, z = 0.0, Lx = 0.10, Ly = 0.02, Lz = 0.30, n =  'j' , e = 0.32, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

face3 = Face(nome = 'Face3', x = -5.0, y = 0.0, z = 0.0, Lx = 0.02, Ly = 0.10, Lz = 0.30, n =  'i' ,  e = 0.82, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

face4 = Face(nome = 'Face4', x = 0.0, y = -5.0, z = 0.0, Lx = 0.1, Ly = 0.02, Lz = 0.30, n =  'j' ,  e = 0.32, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

face5 = Face(nome = 'Face5', x = 0.0, y = 0.0, z = -15.0, Lx = 0.1, Ly = 0.10, Lz = 0.02, n =  'k' ,  e = 0.32, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

face6 = Face(nome = 'Face6', x = 0.0, y = 0.0, z = 15.0, Lx = 0.1, Ly = 0.10, Lz = 0.02, n =  'k' ,  e = 0.02, a = 0.85,
             rho = 2700.0, k = 200, cp = 920.0)

faces = [face1, face2, face3, face4, face5, face6]


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
print(K)
# Define a matriz inicial preenchida por zeros

K_m = np.zeros((len(conduc), len(conduc)))

# Lógica para preencher a matriz com 1 onde existe condução entre nós

for i in range(0, len(conduc)):

    B = set(K.iloc[:, K.columns.get_loc(lista_nomes[i])])  # Recebe o valor dentro do DF dada
    print(B)
    for j in range(0, len(conduc)):
        if i == j:
            K_m[i, j] = 0
        elif j + 1 in B:
            K_m[i, j] = 1

# Calculo das áreas de condução e comprimento

areas = {'1 2': 0.02*0.3,
         '1 4': 0.02*0.3,
         '1 5': 0.02*0.1,
         '1 6': 0.02*0.1,
         '2 1': 0.02*0.3,
         '2 3': 0.02*0.3,
         '2 5': 0.02*0.1,
         '2 6': 0.02*0.1,
         '3 2': 0.02*0.3,
         '3 4': 0.02*0.3,
         '3 5': 0.02*0.1,
         '3 6': 0.02*0.1,
         '4 1': 0.02*0.3,
         '4 3': 0.02*0.3,
         '4 5': 0.02*0.1,
         '4 6': 0.02*0.1,
         '5 1': 0.02*0.1,
         '5 2': 0.02*0.1,
         '5 3': 0.02*0.1,
         '5 4': 0.02*0.1,
         '6 1': 0.02*0.1,
         '6 2': 0.02*0.1,
         '6 3': 0.02*0.1,
         '6 4': 0.02*0.1
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
     '6 4': 0.10 / 2.0
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
    k = (b * faces[i-1].k) / comprimento[K]
    K_m[i - 1, j - 1] = k
    K += 1

# matriz de gebhart

F = ff_interno(faces)
fg = fator_gebhart(faces, F)
print(sum(fg[0]))

# Calculando a variação de temperatura na órbita

# importar as temperaturas da orbita
calor = pd.read_csv("results/Calor_Incidente.csv", sep=',')
# calor_total_face1 = calor.iloc[:, calor.columns.get_loc('Total 1')].tolist()
calor_lump1 = np.array(calor['Total 1'])
calor_lump2 = np.array(calor['Total 2'])
calor_lump3 = np.array(calor['Total 3'])
calor_lump4 = np.array(calor['Total 4'])
calor_lump5 = np.array(calor['Total 5'])
calor_lump6 = np.array(calor['Total 6'])

calor_lump1 = np.array(calor['Total 1'])
calor_lump2 = np.zeros(len(calor))
calor_lump3 = np.zeros(len(calor))
calor_lump4 = np.zeros(len(calor))
calor_lump5 = np.zeros(len(calor))
calor_lump6 = np.zeros(len(calor))
calor = np.array([calor_lump1, calor_lump2, calor_lump3, calor_lump4, calor_lump5, calor_lump6])

def lumped(calor, fg, area_rad, faces):
    # Temperaturas dos nós iniciais que serão alteradas a cada passo de integração
    Temp_ini = [273.0, 273.0, 273.0, 273.0, 273.0, 273.0]
    Temp_pos = [273.0, 273.0, 273.0, 273.0, 273.0, 273.0]

    # passo de integracao termica
    Dt_termico = 0.01

    # passo de integracao da orbita
    Dt_orbita = 1.0

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
    # tracemalloc.start()

    while n < numero_orbitas:
        for k in tqdm(range(0, total, 1), colour=cores[n]):
            while cont < Dt_orbita:
                # passo orbital
                Tcond = np.zeros(6)
                Trad = np.zeros(6)
                alfa = 0.09
                emi = 0.05
                for i in range(0, 6):
                    for j in range(0, 6):

                        # Preencher o vetor conducao
                        Tcond[j] = (K_m[i, j] * (Temp_ini[j] - Temp_ini[i]))

                        # Preencher o vetor radiacao
                        Trad[j] = (sig * faces[i].e * area_rad[i] * fg[i, j] * (
                                Temp_ini[j] ** 4 - Temp_ini[i] ** 4))

                    # Capacitancia termica
                    rho = faces[i].rho
                    cp = faces[i].cp
                    V = faces[i].Lx * faces[i].Ly * faces[i].Lz
                    C_i = rho * cp * V

                    # Equacao transiente para cada nó
                    Temp_pos[i] = Temp_ini[i] + (Dt_termico / C_i) * (sum(Tcond) + sum(Trad) +
                                                                      area_rad[i] * calor[i, k] * (faces[i].a)
                                                                      - sig * area_rad[i] * faces[i].e * (Temp_ini[i] ** 4))

                    # Delta os dados da memória
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

        # Elimina os dados da RAM
        gc.collect()

        import csv
        with open(f'{nome_arquivo}/resultado_temp_{n}.csv', 'w', newline='') as arquivo_csv:
            writer = csv.writer(arquivo_csv)
            writer.writerow(
                ['Face1', 'Face2', 'Face3', 'Face4', 'Face5', 'Face6'])  # Escreve o cabeçalho do CSV

            for lista in var_temp:
                writer.writerow(lista)  # Escreve cada lista como uma linha no CSV

        arquivo_csv.close()  # Fecha o arquivo após a conclusão das operações de escrita
        del var_temp
        gc.collect()
        var_temp = []
        n += 1


lumped(calor, fg, area_rad, faces)

fim = datetime.now()
var = fim - inicio

