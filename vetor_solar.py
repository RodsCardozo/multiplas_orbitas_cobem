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
    beta = np.radians(beta)
    eq = - beta + asin(np.dot([np.cos(np.radians(lamb_ecl)), np.sin(np.radians(lamb_ecl))*np.cos(e),
         np.sin(np.radians(lamb_ecl))*np.sin(e)], [sin(raan) * np.sin(inc), -cos(raan) * np.sin(inc), np.cos(inc)]))

    return nsolve(eq, 1.0)
