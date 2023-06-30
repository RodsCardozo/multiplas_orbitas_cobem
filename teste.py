'''# Importa a biblioteca pandas
import pandas as pd

# Cria um DataFrame de exemplo
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})

# Extrai os valores das colunas para uma lista de tuplas
tuplas = list(zip(df['col1'], df['col2'], df['col3']))

# Converte as tuplas em listas
lista = [list(tupla) for tupla in tuplas]

# Exibe a lista gerada
print(lista)'''
import numpy as np

'''from random import random
from memory_profiler import profile

@profile
def meu_loop():
    minha_lista = []
    for i in range(1000000):
        minha_lista.append(random())
    return minha_lista

if __name__ == '__main__':
    meu_loop()'''
'''import psutil
import matplotlib.pyplot as plt

plt.ion()  # ativa o modo interativo

fig = plt.figure()
ax = fig.add_subplot(111)

while True:
    ax.clear()
    mem = psutil.virtual_memory()
    ax.bar(['Used', 'Available', 'Free'], [mem.used, mem.available, mem.free])
    ax.set_title('Uso de memória')
    ax.set_ylabel('Bytes')
    plt.draw()
    plt.pause(0.1)  # pausa a cada atualização para dar tempo de desenhar o gráfico'''
'''
import pyproj

# Define as coordenadas do vetor posição do satélite
x = -2.5809539349100987e-13 * 1000 # coordenada x em metros # 	-629.104152290459	-1256.2919624901176
y = -629.104152290459 * 1000 # coordenada y em metros
z = -1256.2919624901176 * 1000 # coordenada z em metros
# Define o sistema de coordenadas de entrada (geocêntrico)
input_proj = pyproj.CRS.from_epsg(4328)

# Define o sistema de coordenadas de saída (geográfico)
output_proj = pyproj.CRS.from_epsg(4326)

# Cria um objeto Transformer para realizar a transformação de coordenadas
transformer = pyproj.transformer.Transformer.from_crs(input_proj, output_proj)

# Converte as coordenadas do sistema de coordenadas geocêntricas para o sistema de coordenadas geográficas
lon, lat, alt = transformer.transform(x, y, z, radians=True)

# Exibe as coordenadas de latitude e longitude em graus
print(f"Latitude: {np.degrees(lat):.4f} graus")
print(f"Longitude: {np.degrees(lon):.4f} graus")'''
'''from datetime import datetime, timedelta
from pyorbital.orbital import Orbital
import matplotlib.pyplot as plt

norad_id = 25544 # ISS NORAD ID
satellite = Orbital('N', norad_id)
print(satellite)
start_time = datetime.utcnow()
end_time = start_time + timedelta(days=1)
time_step = timedelta(minutes=5)

lons, lats = [], []

while start_time < end_time:
    lon, lat, alt = satellite.get_lonlatalt(start_time)
    lons.append(lon)
    lats.append(lat)
    start_time += time_step

fig, ax = plt.subplots()
ax.plot(lons, lats)
ax.set_aspect('equal')
plt.show()'''
'''import pygame
import math
import sys

pygame.init()
screen = pygame.display.set_mode((1000, 1000))
pygame.display.set_caption("Rotating Object")

x = 200
y = 200
angle = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    screen.fill((255, 255, 255))
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        angle += 1
    if keys[pygame.K_RIGHT]:
        angle -= 1
    rad_angle = math.radians(angle)
    new_x = x * math.cos(rad_angle) - y * math.sin(rad_angle)
    new_y = y * math.cos(rad_angle) + x * math.sin(rad_angle)
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(new_x, new_y, 50, 50))
    pygame.display.flip()
    pygame.time.Clock().tick(60)'''

'''import tkinter as tk

def calculate_volume():
    width = float(width_entry.get())
    height = float(height_entry.get())
    depth = float(depth_entry.get())
    volume = width * height * depth
    volume_label.config(text="Volume: " + str(volume))

# Criando a janela principal
root = tk.Tk()
root.title("Cubo Interativo")

# Criando os campos de entrada
width_label = tk.Label(root, text="Largura:")
width_label.grid(row=0, column=0)
width_entry = tk.Entry(root)
width_entry.grid(row=0, column=1)

height_label = tk.Label(root, text="Altura:")
height_label.grid(row=1, column=0)
height_entry = tk.Entry(root)
height_entry.grid(row=1, column=1)

depth_label = tk.Label(root, text="Profundidade:")
depth_label.grid(row=2, column=0)
depth_entry = tk.Entry(root)
depth_entry.grid(row=2, column=1)

# Criando o botão de cálculo
calculate_button = tk.Button(root, text="Calcular", command=calculate_volume)
calculate_button.grid(row=3, column=0)

# Criando o rótulo de exibição do resultado
volume_label = tk.Label(root, text="Volume:")
volume_label.grid(row=3, column=1)

# Iniciando a janela principal
root.mainloop()'''
''''import numpy as np
import pymesh

# Define as dimensões do cubo
cube_width = 50
cube_height = 50
cube_depth = 50

# Cria um objeto Mesh vazio
mesh = pymesh.obj.Mesh()

# Adiciona os vértices do cubo
vertices = np.array([
    [0, 0, 0],  # 0
    [cube_width, 0, 0],  # 1
    [cube_width, cube_height, 0],  # 2
    [0, cube_height, 0],  # 3
    [0, 0, cube_depth],  # 4
    [cube_width, 0, cube_depth],  # 5
    [cube_width, cube_height, cube_depth],  # 6
    [0, cube_height, cube_depth],  # 7
])

for vertex in vertices:
    mesh.add_vertex(vertex)

# Adiciona as faces do cubo
faces = np.array([
    [0, 1, 2, 3],  # face frontal
    [1, 5, 6, 2],  # face direita
    [5, 4, 7, 6],  # face traseira
    [4, 0, 3, 7],  # face esquerda
    [3, 2, 6, 7],  # face superior
    [4, 5, 1, 0],  # face inferior
])

for face in faces:
    mesh.add_face(face)

# Salva o cubo em um arquivo OBJ
pymesh.save_mesh("cubo.obj", mesh)'''

'''var = []

for i in range(0,10):
    for j in range(0,100):
        var.append(j**2)
    import pandas as pd

    df = pd.DataFrame(var, columns=['Face1'])
    df.to_csv(f'teste/resultado_temp{i}.csv', sep=',', index=False)
    var = []
'''

''''nome': 5,
      'x': 0.0,
      'y': 0.0,
      'z': -15.0,
      'Lx': 0.100,
      'Ly': 0.100,
      'Lz': 0.02,
      'n': 'k',
      'emissividade': 0.05,
      'absortividade': 0.09,
      'densidade': 2700.0,
      'condutividade térmica': 200,
      'calor especifico': 920.0'''
'''
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

face1 = Face('Face1', 0.0, 0.0, -15.0, 0.1, 0.1, 0.02, 'k', e=0.05, a=0.09,
             k=200.0, cp=920.0, rho=2700.0)

face2 = Face('Face2', 0.0, 0.0, -15.0, 0.1, 0.1, 0.02, 'k', e=0.05, a=0.09,
             k=200.0, cp=920.0, rho=2700.0)

faces = [face1, face2]
for i in faces:
    print(i.cp)'''
'''
import tkinter as tk
from tkinter import ttk

# Função para executar a simulação com base nos parâmetros inseridos
def executar_simulacao():
    # Obtenha os parâmetros inseridos nas abas
    parametro1 = aba1_entry.get()
    parametro2 = aba2_entry.get()
    # Realize a simulação com base nos parâmetros
    # (substitua esta parte com a lógica da sua simulação)

# Crie a janela principal
janela = tk.Tk()
janela.title("Simulação Numérica")

# Crie as abas
abas = ttk.Notebook(janela)

# Aba 1
aba1 = ttk.Frame(abas)
abas.add(aba1, text="Aba 1")

# Adicione widgets à Aba 1
aba1_label = ttk.Label(aba1, text="Parâmetro 1:")
aba1_label.pack()
aba1_entry = ttk.Entry(aba1)
aba1_entry.pack()

# Aba 2
aba2 = ttk.Frame(abas)
abas.add(aba2, text="Aba 2")

# Adicione widgets à Aba 2
aba2_label = ttk.Label(aba2, text="Parâmetro 2:")
aba2_label.pack()
aba2_entry = ttk.Entry(aba2)
aba2_entry.pack()

# Botão para executar a simulação
botao_executar = ttk.Button(janela, text="Executar Simulação", command=executar_simulacao)
botao_executar.pack()

# Empacote as abas
abas.pack()

# Inicie o loop principal da interface gráfica
janela.mainloop()'''
import plotly.graph_objects as go

# Dados de exemplo
x = [100 * i for i in range(1, 13)]  # Valores de x a cada 100 segundos
y = [i for i in range(1, 13)]  # Valores de y

# Configuração das datas
tickvals = [100 * i for i in range(1, 13)]  # Posições dos ticks no eixo-x
ticktext = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']  # Texto dos ticks no eixo-x

# Criação do gráfico
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines'))

# Atualização do layout com as configurações de legenda
fig.update_layout(
    xaxis=dict(
        tickmode='array',
        tickvals=tickvals,
        ticktext=ticktext,
        title='Tempo'
    ),
    yaxis=dict(title='Valores'),
    title='Análise Anual',
    showlegend=True
)

# Exibição do gráfico
fig.show()
