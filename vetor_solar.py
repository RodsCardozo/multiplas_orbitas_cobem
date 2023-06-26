def nome_mes(mes):
    if mes == 1:
        mes = 'Jan'
    if mes == 2:
        mes = 'Feb'
    if mes == 3:
        mes = 'Mar'
    if mes == 4:
        mes = 'Apr'
    if mes == 5:
        mes = 'May'
    if mes == 6:
        mes = 'Jun'
    if mes == 7:
        mes = 'Jun'
    if mes == 8:
        mes = 'Aug'
    if mes == 9:
        mes = 'Sep'
    if mes == 10:
        mes = 'Oct'
    if mes == 11:
        mes = 'Nov'
    if mes == 12:
        mes = 'Dec'
    return mes


def posicao_sol(date):
    """
    :param date: date of interest
    :return: solar vector
    """
    from datetime import datetime
    import numpy as np
    def utc_to_jd(date):
        # separa a string data
        dt = date

        # converte datetime para tempo juliano
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        jd = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

        # adiciona fração do dia
        frac_day = (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        jd += frac_day

        return jd

    # calcula a data juliana para hoje
    data = datetime(month= date.month, day= date.day, year=date.year, hour=date.hour, minute=date.minute)

    # calcula o século juliano
    T_uti = (utc_to_jd(data) - 2451545.0) / 36525

    T_tdb = T_uti

    # calcula a longitude média do sol
    lamb_M = (280.46 + 36000.771 * T_tdb) % 360

    # anomalia media para o sol
    M_sol = (357.5291092 + 35999.05034*T_tdb)% 360


    # calcula a longitude da ecliptica
    lamb_ecl = lamb_M + 1.914666471 * np.sin(np.radians(M_sol)) + 0.019994643 * np.sin(np.radians(2 * M_sol))

    # calcula a obliquidade da ecliptica
    e = 23.439291 - 0.0130042*T_tdb

    # magnitude da distancia do sol
    r_sol = 1.000140612 - 0.016708617 * np.cos(np.radians(M_sol)) - 0.000139589 * np.cos(np.radians(2 * M_sol))

    # vetor posicao solar
    r_sol_vet = [r_sol * np.cos(np.radians(lamb_ecl)), r_sol * np.cos(np.radians(e)) * np.sin(np.radians(lamb_ecl)),
                 r_sol * np.sin(np.radians(e)) * np.sin(np.radians(lamb_ecl))]
    vet_sol = [x * 149597870.7 for x in r_sol_vet]
    return vet_sol


def beta_angle(date, inc, raan):

    import numpy as np
    import ephem
    import math
    from datetime import datetime
    def utc_to_jd(date):
        # separa a string data
        dt = date

        # converte datetime para tempo juliano
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        jd = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

        # adiciona fração do dia
        frac_day = (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        jd += frac_day

        return jd

    if type(date) == str:
        input_string = date
        date = datetime.strptime(input_string, '%m/%d/%Y %H:%M:%S')
    else:
        date = date
    # calcula a data juliana para hoje

    date = datetime(month=date.month, day=date.day, year=date.year, hour=date.hour, minute=date.minute)

    # calcula o século juliano
    T_uti = (utc_to_jd(date) - 2451545.0) / 36525

    T_tdb = T_uti

    # calcula a longitude média do sol
    lamb_M = (280.46 + 36000.771 * T_tdb) % 360

    # anomalida média do sol
    M_sol = 357.5291092 + 35999.05034*T_tdb

    # Criando um objeto para a data desejada
    data = ephem.Date(date)

    # Criando um objeto para o Sol
    sol = ephem.Sun()

    # Atualizando as coordenadas do Sol para a data desejada
    sol.compute(data)

    # Obtendo a declinação em radianos
    dec = sol.dec

    # inclincaçao da orbita terrestre
    e = np.radians(23.4)

    # longitude da ecliptica
    lamb_ecl = lamb_M + 1.914666471 * np.sin(np.radians(M_sol)) + 0.019994643*np.sin(2*M_sol)

    # vetor do plano da eclipse
    s = [np.cos(np.radians(lamb_ecl)), np.sin(np.radians(lamb_ecl))*np.cos(e),
         np.sin(np.radians(lamb_ecl))*np.sin(e)]

    # vetor do plano do satelite
    raan = np.radians(raan)
    inc = np.radians(inc)
    n = [np.sin(raan) * np.sin(inc), -np.cos(raan) * np.sin(inc), np.cos(inc)]

    beta_angle = np.arcsin(np.dot(s,n))

    return beta_angle

def taxa_precessao(ecc, semi_eixo_maior, inc):
    import math
    mu = 398600
    R_E = 6371.0
    j2 = 0.0010826
    omega_pre = -((3 * math.sqrt(mu) * j2 * R_E**2) / (2 * (1 - ecc**2)**2 * semi_eixo_maior**(7/2))) * math.cos(math.radians(inc))
    return omega_pre

def beta_raan(beta, date, inc):
    import numpy as np
    import ephem
    import math
    from datetime import datetime
    def utc_to_jd(date):
        # separa a string data
        dt = date

        # converte datetime para tempo juliano
        a = (14 - dt.month) // 12
        y = dt.year + 4800 - a
        m = dt.month + 12 * a - 3
        jd = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

        # adiciona fração do dia
        frac_day = (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0
        jd += frac_day

        return jd

    if type(date) == str:
        input_string = date
        date = datetime.strptime(input_string, '%m/%d/%Y %H:%M:%S')
    else:
        date = date
    # calcula a data juliana para hoje

    data = datetime(month=date.month, day=date.day, year=date.year, hour=date.hour, minute=date.minute)

    # calcula o século juliano
    T_uti = (utc_to_jd(data) - 2451545.0) / 36525

    T_tdb = T_uti

    # calcula a longitude média do sol
    lamb_M = (280.46 + 36000.771 * T_tdb) % 360

    # anomalida média do sol
    M_sol = 357.5291092 + 35999.05034*T_tdb

    # Criando um objeto para a data desejada
    data = ephem.Date(date)

    # Criando um objeto para o Sol
    sol = ephem.Sun()

    # Atualizando as coordenadas do Sol para a data desejada
    sol.compute(data)

    # Obtendo a declinação em radianos
    dec = sol.dec

    # inclincaçao da orbita terrestre
    e = np.radians(23.4)

    # longitude da ecliptica
    lamb_ecl = lamb_M + 1.914666471 * np.sin(np.radians(M_sol)) + 0.019994643*np.sin(2*M_sol)

    # vetor do plano do satelite
    inc = np.radians(inc)
    from sympy import nsolve, Symbol, cos, sin, asin
    raan = Symbol('raan')
    eq = - beta + asin(np.dot([np.cos(np.radians(lamb_ecl)), np.sin(np.radians(lamb_ecl))*np.cos(e),
         np.sin(np.radians(lamb_ecl))*np.sin(e)], [sin(raan) * np.sin(inc), -cos(raan) * np.sin(inc), np.cos(inc)]))

    return nsolve(eq, 1.0)
'''a = beta_raan(0.0, '01/01/2015 00:00:00', 80.0)
import numpy as np
print(f'valor de raan para que seja beta 0: {a * 180/np.pi}')'''


if __name__ == '__main__':
    import numpy as np

    beta_iss = beta_angle('01/01/2015 00:00:00', 80.0, 96.8277534533855)
    print(np.degrees(beta_iss))
    import pandas as pd
    from datetime import datetime, timedelta
    dia_ini = '01/01/2015 00:00:00'
    ini_date = datetime.strptime(dia_ini, "%m/%d/%Y %H:%M:%S")
    data = [ini_date + timedelta(days=x) for x in range(0,365,30)]
    print(data)
    inc = 98.0
    raan0 = 0.0
    raan = []
    for i in range(0, 365):
        raan_var = raan0 + taxa_precessao(0.0, 6771.0, inc) * 24 * 60 * 60

        raan.append(np.degrees(raan_var))
        raan0 = raan_var

    beta = [beta_angle(dia, inc, 0.0) for dia in data]
    data2 = [str(dado) for dado in data]
    df = pd.DataFrame(np.degrees(beta), columns=['Beta'])
    df['data'] = data2
    df['inc'] = inc
    import plotly.express as px
    df['mes'] = [nome_mes(dado.month) + ' ' + str(dado.day) for dado in data]
    raan = [beta_raan(beta, data, inc)*(180/np.pi) for beta, data in zip(beta,data)]
    df['raan'] = raan
    fig = px.scatter(df, y="Beta", text="mes")
    fig.update_traces(textposition="bottom left")
    fig.show()

'''from datetime import datetime, timedelta
inicio = datetime(day=20, month=3, year=2023, hour=12, minute=0)
dias = [inicio + timedelta(days=x) for x in range(0,365)]
sol_dia = [posicao_sol(dia) for dia in dias]

import pandas as pd
import numpy as np

lat = []

for i in range(0, 365):

    if np.sqrt((sol_dia[i][1] / np.linalg.norm(sol_dia[i]))**2) > 1.0:
        lat.append(np.degrees(np.arcsin(1.0)))
    else:
        lat.append(np.degrees(np.arcsin(sol_dia[i][2] / np.linalg.norm(sol_dia[i]))))

long = []
for i in range(0, len(sol_dia)):
    long.append(np.degrees(np.arctan2(sol_dia[i][1], sol_dia[i][0])))




df = pd.DataFrame(sol_dia, columns=['x', 'y', 'z'])
df['latitude'] = lat
df['longitude'] = long
df['dia'] = dias


import plotly.express as px

fig = px.line(df, x="longitude", y="latitude", text="dia")
fig.update_traces(textposition="bottom right")
fig.show()'''