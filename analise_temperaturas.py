
import pandas as pd
import os, sys

nome_pasta = input(f'Insira o nome da pasta: ')
nome_arquivo = input(f'Insira o nome do arquivo: ')
formato = input('Insira o formato: ')
# Lista todos os arquivos CSV em um diretório específico
arquivos_csv = []

arquivo_inicio = 24
arquivo_final = 25
for i in range(arquivo_inicio,arquivo_final+1):
    nome = (f"{nome_pasta}/resultado_temp_{i}.csv")
    arquivos_csv.append(nome)


# Cria uma lista vazia para armazenar os DataFrames de cada arquivo CSV
dataframes = []

# Loop pelos arquivos CSV e carrega cada um em um DataFrame separado
for arquivo in arquivos_csv:
    df = pd.read_csv(arquivo)
    dataframes.append(df)

# Concatena os DataFrames em um único DataFrame
df_final = pd.concat(dataframes)

# Salva o DataFrame final em um arquivo CSV
df_final.to_csv(f'{nome_pasta}/{nome_arquivo}.{formato}', index=False)


