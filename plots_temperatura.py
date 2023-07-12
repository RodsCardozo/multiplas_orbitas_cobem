
def temp_comp2(dataframe, lista_componente, componente, plot):
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
    df = dataframe
    df2 = df.iloc[::100, :].reset_index(drop=True)
    linhas = px.line(df2, y=lista_componente)
    linhas.update_layout(showlegend = True,
                         legend = dict(bgcolor = 'rgb(250,250,250)',
                                       bordercolor = 'rgb(130, 130, 130)',
                                       borderwidth = 1.0,
                                       itemdoubleclick = "toggleothers",
                                       title_text = 'Componente'
                                       ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.5)',
                             gridwidth=1,
                             dtick=1000,
                     griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.5)',
                             gridwidth=1,
                     griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor = 'rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Temperature [K]')
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
def createFolder(directory):
    import os, sys
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)



# Plot dos resultados
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio

beta_lista = [0.0, 72.0]
for beta in beta_lista:
    temperaturas = pd.read_csv(f'Transiente10/beta{beta}/resultado_temp_14.csv')
    temperaturas = temperaturas.rename(columns={'substrato 1': 'Substrate 1','substrato 2': 'Substrate 2','substrato 3': 'Substrate 3',
                                        'substrato 4': 'Substrate 4','substrato 5': 'Substrate 5','substrato 6': 'Substrate 6',
                                        'substrato 7': 'Substrate 7','substrato 8': 'Substrate 8'})
    temperaturas = temperaturas.rename(columns={'Base painel 1': 'Face 1','Base painel 2': 'Face 2','Base painel 3': 'Face 3', 'Base painel 4': 'Face 4',
                                        'Tampa 1': 'Botton', 'Tampa 2': 'Top'})

    print(max(list(temperaturas.max() - temperaturas.min())))

    # cria a pasta para salvar as imagens
    createFolder(f'Transiente10/beta{beta}/graficos')

    # PCBS
    pcbs = temp_comp2(temperaturas, ['PCB 1', 'PCB 2', 'PCB 3', 'PCB 4'], 'PCBs', 2)
    pio.write_image(pcbs, f'Transiente10/beta{beta}/graficos/Temperatura_PCB_beta_{beta}.png')

    # substratos
    substratos = temp_comp2(temperaturas, ['Substrate 1', 'Substrate 3', 'Substrate 5', 'Substrate 7'], 'Substrates', 2)
    pio.write_image(substratos, f'Transiente10/beta{beta}/graficos/Temperatura_substrato_beta_{beta}.png')

    # base painel
    faces_externas = temp_comp2(temperaturas, ['Face 1', 'Face 2', 'Face 3', 'Face 4', 'Botton', 'Top'],
                                'External Faces', 2)
    pio.write_image(faces_externas, f'Transiente10/beta{beta}/graficos/Temperatura_faces_externas_beta_{beta}.png')

    base_painel2 = temp_comp2(temperaturas, ['base cobre 1', 'base cobre 2', 'base cobre 3', 'base cobre 4'], 'Base dos Paineis', 2)

    pio.write_image(base_painel2, f'Transiente10/beta{beta}/graficos/Temperatura_base_cobre_beta_{beta}.png')

    estrutura_vertical = temp_comp2(temperaturas, ['Estrutura Vertical 1','Estrutura Vertical 2','Estrutura Vertical 3',
                                                  'Estrutura Vertical 4','Estrutura Vertical 5','Estrutura Vertical 6',
                                                  'Estrutura Vertical 7','Estrutura Vertical 8'], 'Estruturas Verticais', 2)
    pio.write_image(estrutura_vertical, f'Transiente10/beta{beta}/graficos/Temperatura_estrutura_vertical_beta_{beta}.png')

    estrutura_horizontal = temp_comp2(temperaturas, ['Estrutura Horizontal 1', 'Estrutura Horizontal 2', 'Estrutura Horizontal 3', 'Estrutura Horizontal 4',
                                                    'Estrutura Horizontal 5', 'Estrutura Horizontal 6', 'Estrutura Horizontal 7', 'Estrutura Horizontal 8',
                                                    'Estrutura Horizontal 9', 'Estrutura Horizontal 10', 'Estrutura Horizontal 11',
                                                    'Estrutura Horizontal 12', 'Estrutura Horizontal 13', 'Estrutura Horizontal 14',
                                                    'Estrutura Horizontal 15', 'Estrutura Horizontal 16'], 'Estrutura Horizontal', 2)
    pio.write_image(estrutura_horizontal, f'Transiente10/beta{beta}/graficos/Temperatura_estrutura_horizontal_beta_{beta}.png')