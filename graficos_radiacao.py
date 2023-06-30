
from plots import *
import pandas as pd
import plotly.io as pio
beta = [0.0, 72.0]
inc = [51.63, 51.63]
for i in range(len(beta)):

    radiacao = pd.read_csv(f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/Calor_incidente.csv')

    linhas1 = calor_solar(radiacao, 2)
    pio.write_image(linhas1, f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/graficos_{i}/radiacao_solar_beta_{beta[i]}.png')

    linhas2 = calor_albedo(radiacao, 2)
    pio.write_image(linhas2, f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/graficos_{i}/radiacao_albedo_beta_{beta[i]}.png')

    linhas3 = calor_IR_Terra(radiacao, 2)
    pio.write_image(linhas3, f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/graficos_{i}/radiacao_IR_Terra_beta_{beta[i]}.png')

    linhas4 = calor_total(radiacao, 2)
    pio.write_image(linhas4, f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/graficos_{i}/radiacao_total_beta_{beta[i]}.png')

    linhas5 = soma_radiaco(radiacao, 2)
    pio.write_image(linhas5,
                    f'beta{beta[i]}/analise_{inc[i]}/results_{i}/radiacao_{i}/graficos_{i}/soma_radiacao_beta_{beta[i]}.png')