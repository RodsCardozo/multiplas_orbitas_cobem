#plots.py>
# plotagens orbitais #
def plot_animacao_orbita(dataframe, size):
    import plotly.express as px
    import plotly.graph_objs as go
    import plotly.io as pio
    df = dataframe
    size = size
    fig = px.scatter_3d(df, x="X_ECI", y="Y_ECI", z="Z_ECI",
                        range_x=[-size, size],
                        range_y=[-size, size],
                        range_z=[-size, size],
                        animation_frame="Tempo",
                        title="Spaceflight",
                        size_max=100, opacity=0.7

                        )

    import numpy as np

    # Raio da Terra
    Re = 6371.0

    # Define as coordenadas x, y e z para a superfície esférica da Terra
    phi, theta = np.linspace(0, np.pi, 100), np.linspace(0, 2 * np.pi, 100)
    x = Re*np.outer(np.sin(theta), np.cos(phi))
    y = Re*np.outer(np.sin(theta), np.sin(phi))
    z = Re*np.outer(np.cos(theta), np.ones_like(phi))

    # Adiciona a superfície esférica da Terra
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale='earth', showscale=False))

    # Adiciona um traçado da orbita
    x1 = df['X_ECI'].to_list()
    y1 = df['Y_ECI'].to_list()
    z1 = df['Z_ECI'].to_list()
    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', marker=dict(size=5, color='red'), name='Sat 1'))

    # Define o layout da figura
    fig.update_layout(scene=dict(
        xaxis=dict(title='X', autorange=True),
        yaxis=dict(title='Y', autorange=True),
        zaxis=dict(title='Z', autorange=True)),
        updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(method='animate',
                                                                          args=[None, dict(
                                                                              frame=dict(duration=0.1, redraw=True),
                                                                              fromcurrent=True,
                                                                              transition=dict(duration=0,
                                                                                              easing='linear'),
                                                                              mode='immediate')])])])
    fig.show()
    pio.write_image(fig, 'imagens_resultado/animacao_3d.png')
    return
def plot_groundtrack_3D(dataframe, plot):
    import plotly.graph_objects as go
    import plotly.io as pio
    df = dataframe

    scl = ['rgb(213,62,79)', 'rgb(244,109,67)', 'rgb(253,174,97)', \
           'rgb(254,224,139)', 'rgb(255,255,191)', 'rgb(230,245,152)', \
           'rgb(171,221,164)', 'rgb(102,194,165)', 'rgb(50,136,189)'
           ]
    n_colors = len(scl)

    fig = go.Figure()


    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        mode='lines',
        line=dict(width=2, color='rgb(213,62,79)'
                      ),
        connectgaps=False)
        )

    fig.update_layout(
        title_text='Contour lines over globe<br>(Click and drag to rotate)',
        showlegend=False,
        geo=dict(
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(34,139,34)',
            lakecolor='rgb(0, 255, 255)',
            oceancolor='rgb(0,0,139)',
            projection=dict(
                type='orthographic',
                rotation=dict(
                    lon=-100,
                    lat=40,
                    roll=0
                ),

            ),
            lonaxis=dict(
                showgrid=True,
                gridcolor='rgb(0,0,0)',
                gridwidth=0.5
            ),
            lataxis=dict(
                showgrid=True,
                gridcolor='rgb(0,0,0)',
                gridwidth=0.5
            )
        )
    )
    pio.write_image(fig, 'imagens_resultado/groundtrack_3D.png')
    if plot == 1:
        fig.show()
    elif plot == 2:
        pass
    return
def plot_groundtrack_2D(dataframe, plot):
    import plotly.io as pio
    import plotly.graph_objects as go
    import plotly.io as pio
    from PIL import Image
    df = dataframe
    fig = go.Figure(data=go.Scattergeo(
        lat=df['latitude'],
        lon=df['longitude'],
        mode='lines',
        line=dict(width=2, color='red'),
        name='A1',
        showlegend=True
    ))

    fig.update_layout(
        title= dict(font_color = 'red',
                    text = 'Groundtrack 2D',
                    x = 0.5,
                    ),
        showlegend=True,
        geo=dict(
            showland=True,
            showcountries=True,
            showocean=True,
            countrywidth=0.5,
            landcolor='rgb(255,255,255)',
            lakecolor='rgb(240,248,255)',
            oceancolor='rgb(240,248,255)',
            projection_type="equirectangular",
            coastlinewidth=1,
            lataxis=dict(
                dtick=30,
                gridcolor='rgb(0, 0, 0)',
                griddash = "dash",
                gridwidth=0.5,
                range=[-90, 90],
                showgrid=True,
                tick0 = -90

            ),
            lonaxis=dict(
                range=[-180, 180],
                showgrid=True,
                gridcolor='rgb(0, 0, 0)',
                griddash="dash",
                gridwidth=0.5,
                dtick=60,
                tick0 = -180
            )

        )
    )
    pio.write_image(fig, 'imagens_resultado/groundtrack_2D.png')
    if plot == 1:
        fig.show()
    elif plot == 2:
        pass
    return
def plot_ground2d_novo(dataframe, plot):
    import plotly.graph_objects as go
    import plotly.io as pio
    from PIL import Image

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    df = dataframe
    f = plt.figure(figsize=(10, 7.5))
    m = Basemap(projection="mill", lon_0=0);
    m.drawcoastlines()
    m.drawparallels(np.arange(-90, 91, 30), labels=[1, 0, 0, 0])
    m.drawmeridians(np.arange(-180, 181, 60), labels=[0, 0, 0, 1])

    plt.savefig('groundtrack.png')
    # Create figure
    fig = go.Figure()

    pyLogo = Image.open("groundtrack.png")

    fig = go.Figure()

    # Constants
    img_width = 1600
    img_height = 900
    scale_factor = 0.5

    # Add invisible scatter trace.
    # This trace is added to help the autoresize logic work.


    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=pyLogo)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )
    fig.add_trace(
        go.Scatter(
            x=df['longitude'],
            y=df['latitude'],
            mode="markers",
            marker_opacity=0
        )
    )
    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    if plot == 1:
        fig.show(config={'doubleClick': 'reset'})
    elif plot == 2:
        pass

# Plotagem termica #

def calor_solar(dataframe, plot):
    import plotly.express as px
    import plotly.io as pio
    Q_sol = dataframe
    linhas = px.line(Q_sol, y=['Solar 1', 'Solar 2', 'Solar 3', 'Solar 4', 'Solar 5', 'Solar 6'])

    linhas.update_layout(showlegend=True,
                         legend=dict(bgcolor='rgb(250, 250, 250)',
                                     bordercolor='rgb(120, 120, 120)',
                                     borderwidth=1.0,
                                     itemdoubleclick="toggleothers",
                                     title_text='Face'
                                     ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=1000,
                     griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=150,
                     griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor='rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Radiation [W/m^2]')
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
    #pio.write_image(linhas, 'imagens_resultado/radiacao_solar.png')
    return linhas

def calor_albedo(dataframe,plot):
    import plotly.express as px
    import plotly.io as pio
    Q_alb = dataframe
    linhas = px.line(Q_alb, y=['Albedo 1', 'Albedo 2', 'Albedo 3', 'Albedo 4', 'Albedo 5', 'Albedo 6'])
    linhas.update_layout(showlegend=True,
                         legend=dict(bgcolor='rgb(250, 250, 250)',
                                     bordercolor='rgb(120, 120, 120)',
                                     borderwidth=1.0,
                                     itemdoubleclick="toggleothers",
                                     title_text='Face'
                                     ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=1000,
                     griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                     griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor='rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Radiation [W/m^2]')
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
    #pio.write_image(linhas, 'imagens_resultado/radiacao_albedo.png')
    return linhas

def calor_IR_Terra(dataframe,plot):
    import plotly.io as pio
    import plotly.express as px
    Q_ir = dataframe
    linhas = px.line(Q_ir, y = ['IR Terra 1', 'IR Terra 2', 'IR Terra 3', 'IR Terra 4', 'IR Terra 5', 'IR Terra 6'])
    linhas.update_layout(showlegend=True,
                         legend=dict(bgcolor='rgb(250, 250, 250)',
                                     bordercolor='rgb(120, 120, 120)',
                                     borderwidth=1.0,
                                     itemdoubleclick="toggleothers",
                                     title_text='Face'
                                     ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=1000,
                     griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=50,
                     griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor='rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Radiation [W/m^2]')
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
    #pio.write_image(linhas, 'imagens_resultado/radiacao_IR.png')
    return linhas

def calor_total(dataframe, plot):
    import plotly.io as pio
    import plotly.express as px
    Q_total = dataframe
    linhas = px.line(Q_total, y=['Total 1', 'Total 2', 'Total 3', 'Total 4', 'Total 5', 'Total 6'])
    linhas.update_layout(showlegend=True,
                         legend=dict(bgcolor='rgb(250, 250, 250)',
                                     bordercolor='rgb(120, 120, 120)',
                                     borderwidth=1.0,
                                     itemdoubleclick="toggleothers",
                                     title_text='Face'
                                     ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=1000,
                     griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=150,
                     griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor='rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Radiation [W/m^2]')
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
    #pio.write_image(linhas, 'imagens_resultado/radiacao_total.png')
    return linhas

def soma_radiaco(dataframe, plot):
    import plotly.io as pio
    import plotly.express as px
    Q_total = dataframe
    Q_total['Nadir'] = Q_total['Total 1'] + Q_total['Total 2'] + Q_total['Total 3'] + Q_total['Total 4'] + Q_total['Total 5'] + Q_total['Total 6']
    y_dtick = int((-Q_total['Nadir'].min()+Q_total['Nadir'].max())/5)
    linhas = px.line(Q_total, y=['Nadir'])
    linhas.update_layout(showlegend=True,
                         legend=dict(bgcolor='rgb(250, 250, 250)',
                                     bordercolor='rgb(120, 120, 120)',
                                     borderwidth=1.0,
                                     itemdoubleclick="toggleothers",
                                     title_text='Attitude'
                                     ),
                         xaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             dtick=1000,
                             griddash='dot'
                         ),
                         yaxis=dict(
                             showgrid=True,
                             gridcolor='rgba(100, 100, 100, 0.3)',
                             gridwidth=1,
                             griddash='dot'
                         ),
                         autosize=True,
                         plot_bgcolor='rgb(255, 255, 255)',
                         xaxis_title='Time [s]',
                         yaxis_title='Radiation [W/m^2]')
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