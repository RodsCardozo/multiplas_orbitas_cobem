import sys, os
import pandas as pd
import plotly.io as pio
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
def temp_comp2(dataframe, lista_componente, componente, plot):
    import plotly.express as px
    import plotly.io as pio
    import plotly.graph_objects as go
    df = dataframe
    df2 = df.iloc[::100, :].reset_index(drop=True)
    linhas = px.line(df2, y=lista_componente)
    linhas.update_layout(title=dict(font=dict(color = 'black'),
                                    text = f'{componente}',
                                    x = 0.5),
                         showlegend = True,
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
        ),
        yaxis=dict(
            linecolor='rgb(190, 190, 190)',  # Cor da borda do eixo y
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

    if plot == 1:
        linhas.show()
    elif plot == 2:
        pass
    return linhas

# cria o folder para cada caso
beta=[0.0,72.0]

for I in range(0,2):
    createFolder(f'Transiente8/beta{beta[I]}/regime_permanente')
    for j in range(0,11):
        df = pd.read_csv(f'Transiente8/beta{beta[I]}/resultado_temp_{j}.csv')
        linhas = temp_comp2(df, ['PCB 1'], 'Base Painel', 2)
        pio.write_image(linhas, f'Transiente8/beta{beta[I]}/regime_permanente/PCB1_{j}_{beta[I]}.png')