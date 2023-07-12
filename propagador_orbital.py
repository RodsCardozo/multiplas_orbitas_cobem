# -*- coding: utf-8 -*-
# propagador_orbital_mk4.py>
"""
    Universidade Federal de Santa Catarina
    Laboratory of Applications and Research in Space - LARS
    Orbital Mechanics Division

    Título do Algoritmo = Codigo principal de propagacao orbital e analise termica
    Autor= Rodrigo S. Cardozo
    Versão = 0.1.0
    Data = 05/04/2023

"""


def propagador_orbital(data: str, semi_eixo: float, excentricidade: float, raan, argumento_perigeu: float,
                       anomalia_verdadeira: float, inclinacao: float, num_orbitas: int, delt: float, psi: float,
                       teta: float, phi: float, psip: float, tetap: float, phip: float, massa: float, largura: float,
                       comprimento: float, altura: float, N, beta):

    """
    :param data = inicio da simulacao
    :param semi_eixo = altitude no periapse da orbita
    :param excentricidade = e
    :param raan= Angulo da posicao do nodo ascendente
    :param argumento_perigeu = Angulo da orientacao da linha dos apses
    :param anomalia_verdadeira = algulo do vetor posicao e a linha dos apses com origem no foco
    :param inclinacao = inclinacao da orbita
    :param num_orbitas = numero de orbitas a serem simuladas
    :param delt = Time step for the integration
    :param psi = primeiro angulo de Euler
    :param teta = segundo angulo de Euler
    :param phi = terceiro angulo de Euler
    :param psip = velocidade angular do primeiro angulo de Euler
    :param tetap = velocidade angular do segundo angulo de Euler
    :param phip = velocidade angular do terceiro angulo de Euler
    :param massa = massa do cubesat
    :param largura = largura do cubsat
    :param comprimento = comprimento do cubesat
    :param altura = altura do cubesat
    :return df = Dataframe with informations about orbit and attitude
    :param n = numero de orbitas
    """
    print("Propagando o movimento")
    import numpy as np
    import pandas as pd
    from scipy.integrate import odeint
    from datetime import timedelta
    from nrlmsise00 import msise_model
    from periodo_orbital import periodo_orbital
    import os, sys

    '''def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    createFolder(f'beta{beta}/analise_{inclinacao}/results_{N}/propagador_orbital_{N}')'''
    def propagador(q: list,
                   t: float,
                   rho: float,
                   velocidade: float,
                   massa: float,
                   largura: float,
                   comprimento: float,
                   altura: float,
                   CD: float,
                   posicao: float,
                   Area_transversal: float):  # funcao para integrar

        import numpy as np

        # parametro gravitacional mu = GM
        mu = 398600.0

        # Parametro harmonico J2
        J2 = 1.08263e-3

        # Raio da Terra
        R_terra = 6371.0

        # vetor posicao inicial

        r = posicao
        m = massa  # massa do cubesat
        a = largura  # comprimento do sat
        b = comprimento  # largura do sat
        c = altura # altura do sat
        Ix3 = (m / 12) * (b ** 2 + c ** 2)  # momento de inercia na direcao x
        Iy3 = (m / 12) * (a ** 2 + c ** 2)  # momento de inercia na direcao y
        Iz3 = (m / 12) * (a ** 2 + b ** 2)  # momento de inercia na direcao z

        # Condicoes inicial do propagador
        h, ecc, anomalia_verdadeira, raan, inc, arg_per, q0, q1, q2, q3, wx3, wy3, wz3 = q

        # Equacoes diferenciais ordinarias
        dMdt = [r*((-1/(2*r))*h*rho*velocidade*((CD*Area_transversal)/m) - 1.5 * ((J2*mu*R_terra**2)/r**4) * np.sin(inc)**2*np.sin(2*(arg_per + anomalia_verdadeira))),

                (h/mu)*np.sin(anomalia_verdadeira)*((-1/(2*h))*mu*ecc*rho*velocidade*((CD*Area_transversal)/m)*np.sin(anomalia_verdadeira)
                - 1.5*((J2*mu*R_terra**2)/r**4)*(1 - 3*np.sin(inc)**2*np.sin(arg_per + anomalia_verdadeira)**2))
                + (((-1/(2*r))*h*rho*velocidade*((CD*Area_transversal)/m) - 1.5*((J2*mu*R_terra**2)/r**4)*np.sin(inc)**2*np.sin(2*(arg_per
                + anomalia_verdadeira)))/(mu*h))*((h**2 + mu*r)*np.cos(anomalia_verdadeira) + mu*ecc*r),

                (h/r**2 + ((h**2*np.cos(anomalia_verdadeira))/(mu*ecc*h))*((-1/(2*h))*mu*ecc*rho*velocidade*((CD*Area_transversal)/m)*np.sin(anomalia_verdadeira)
                - 1.5*((J2*mu*R_terra**2)/r**4)*(1 - 3*np.sin(inc)**2*np.sin(arg_per + anomalia_verdadeira)**2))
                - (r + h**2/mu)*(np.sin(anomalia_verdadeira)/(ecc*h))*((-1/(2*r))*h*rho*velocidade*((CD*Area_transversal)/m)
                - (1.5 * (J2*mu*R_terra**2)/r**4) * np.sin(inc)**2 * np.sin(2*(arg_per + anomalia_verdadeira)))),

                (r/(h*np.sin(inc)))*np.sin(arg_per + anomalia_verdadeira)*(- 1.5*((J2*mu*R_terra**2)/r**4)*np.sin(2*inc)*np.sin(arg_per + anomalia_verdadeira)),

                (r / (h)) * np.cos(arg_per + anomalia_verdadeira) * (- 1.5 * ((J2 * mu * R_terra ** 2) / r ** 4) * np.sin(2 * inc) * np.sin(arg_per + anomalia_verdadeira)),

                (-1/(ecc*h))*((h**2/mu)*np.cos(anomalia_verdadeira)*((-1/(2*h))*mu*ecc*rho*velocidade*((CD*Area_transversal)/m)*np.sin(anomalia_verdadeira)
                - 1.5*((J2*mu*R_terra**2)/r**4)*(1 - 3*np.sin(inc)**2*np.sin(arg_per + anomalia_verdadeira)**2))
                - (r + h**2/mu)*np.sin(anomalia_verdadeira)*((-1/(2*r))*h*rho*velocidade*((CD*Area_transversal)/m)
                - 1.5 * ((J2*mu*R_terra**2)/r**4) * np.sin(inc)**2 * np.sin(2*(arg_per + anomalia_verdadeira))))
                - ((r*np.sin(arg_per + anomalia_verdadeira))/(h*np.tan(inc)))*(- 1.5 * (J2*mu*R_terra**2)/r**4 * np.sin(2*inc) * np.sin(arg_per + anomalia_verdadeira)),

                0.5 * (- q1 * wx3 - q2 * wy3 - q3 * wz3),
                0.5 * (+ q0 * wx3 + q3 * wy3 - q2 * wz3),
                0.5 * (- q3 * wx3 + q0 * wy3 + q1 * wz3),
                0.5 * ( q2 * wx3 - q1 * wy3 + q0 * wz3),
                ((Iy3 - Iz3) / Ix3) * wy3 * wz3,
                ((Iz3 - Ix3) / Iy3) * wz3 * wx3,
                ((Ix3 - Iy3) / Iz3) * wx3 * wy3]
        return dMdt

    # Funcao para calcular a densidade atmosferica ao longo da orbita
    def densidade(data, altitude, latitude, longitude):
        """

        :param data: mes/dia/ano hora:minu:sec
        :param altitude: altitude da orbita
        :param latitude: latitude
        :param longitude: longitude
        :return: densidade da atmosfera em kg/m**3
        """
        densidade = msise_model(data, altitude, latitude, longitude, 150, 150, 4, lst=16)
        rho = densidade[0][5] * 1000
        return rho

    # condicoes iniciais

    SMA = float(semi_eixo)  # semi eixo maior
    ecc0 = float(excentricidade)  # ecentricidade da orbita
    raan0 = np.radians(float(raan))  # ascencao direita do nodo ascendente
    arg_per0 = np.radians(float(argumento_perigeu))  # argumento do perigeu
    true_anomaly0 = np.radians(float(anomalia_verdadeira))  # anomalia verdadeira
    inc0 = np.radians(float(inclinacao))  # inclinacao
    rp0 = SMA*(1-excentricidade) # semi eixo maior
    T_orb = periodo_orbital(SMA) # periodo medio da orbita
    mu = 398600 # parametro gravitacional mu = GM
    J2 = 1.08263e-3 # zona harmonica j2
    R_terra = 6371.0 # raio da terra
    h0 = np.sqrt(semi_eixo*mu*(1 - excentricidade**2)) # momento linear do satelite
    psi = float(np.radians(psi))  # angulo inicial de PSI
    teta = float(np.radians(teta))  # angulo inicial de TETA
    phi = float(np.radians(phi))  # angulo inicial de PHI
    psip = float(psip)  # velocidade angular do angulo PSI
    tetap = float(tetap)  # velocidade angular do angulo TETA
    phip = float(phip)  # velocidade angular do angulo PHI

    # parametros para integracao dos quaternions

    wx3_i = float(-psip * np.sin(teta) * np.cos(phi) + tetap * np.sin(phi))     # velocidade angular do corpo em x
    wy3_i = float(psip * np.sin(teta) * np.sin(phi) + tetap * np.cos(phi))      # velocidade angular do corpo em y
    wz3_i = float(psip * np.cos(teta) + phip)                                   # velocidade angular do corpo em z

    q0 = float((np.cos(psi / 2) * np.cos(teta / 2) * np.cos(phi / 2) - np.sin(psi / 2) * np.cos(teta / 2) * np.sin(
        phi / 2)))  # quaternion q0
    q1 = float((np.cos(psi / 2) * np.sin(teta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.sin(teta / 2) * np.sin(
        phi / 2)))  # quaternion q1
    q2 = float((np.sin(psi / 2) * np.sin(teta / 2) * np.cos(phi / 2) - np.cos(psi / 2) * np.sin(teta / 2) * np.sin(
        phi / 2)))  # quaternion q2
    q3 = float((np.cos(psi / 2) * np.cos(teta / 2) * np.sin(phi / 2) + np.sin(psi / 2) * np.cos(teta / 2) * np.cos(
        phi / 2)))  # quaternion q3

    # Matriz de rotacao
    '''x_rot = np.cos(np.radians(argumento_perigeu)) * np.cos(np.radians(raan)) - np.cos(np.radians(inclinacao)) \
            * np.sin(np.radians(argumento_perigeu)) * np.sin(np.radians(raan))

    y_rot = np.cos(np.radians(argumento_perigeu)) * np.sin(np.radians(raan)) + np.cos(np.radians(inclinacao))\
            * np.sin(np.radians(argumento_perigeu)) * np.cos(np.radians(raan))

    z_rot = np.sin(np.radians(inclinacao)) * np.sin(np.radians(argumento_perigeu))'''

    raan = np.radians(raan)
    inclinacao2 = np.radians(inclinacao)
    arg_per = np.radians(argumento_perigeu)


    Q_rot = np.array([[np.cos(arg_per) * np.cos(raan) - np.sin(arg_per) * np.sin(raan) * np.cos(inclinacao2),
                       np.cos(arg_per) * np.sin(raan) + np.sin(arg_per) * np.cos(inclinacao2) * np.cos(raan),
                       np.sin(arg_per) * np.sin(inclinacao2)],
                      [-np.sin(arg_per) * np.cos(raan) - np.cos(arg_per) * np.sin(raan) * np.cos(inclinacao2),
                       -np.sin(arg_per) * np.sin(raan) + np.cos(arg_per) * np.cos(inclinacao2) * np.cos(raan),
                       np.cos(arg_per) * np.sin(inclinacao2)],
                      [np.sin(inclinacao2) * np.sin(raan), -np.sin(inclinacao2) * np.cos(raan), np.cos(inclinacao2)]])
    h1 = np.sqrt(semi_eixo*mu*(1 - excentricidade**2))

    ano = [np.cos(np.radians(anomalia_verdadeira)), np.sin(np.radians(anomalia_verdadeira)), 0]
    r = [((h1**2/mu)*1/(1 + excentricidade*np.cos(np.radians(anomalia_verdadeira)))) * a for a in ano]

    Posi_ini = np.dot(np.transpose(Q_rot), r)

    lamb_e = raan0 # (np.arctan2(Posi_ini[1], Posi_ini[0]))
    latitude0 = np.degrees((np.arcsin(Posi_ini[2] / np.linalg.norm(Posi_ini))))
    longitude0 = np.degrees((np.arctan2(Posi_ini[1], Posi_ini[0])))

    '''import pyproj
    input_proj = pyproj.CRS.from_epsg(4328)

    # Define o sistema de coordenadas de saída (geográfico)
    output_proj = pyproj.CRS.from_epsg(4326)

    # Cria um objeto Transformer para realizar a transformação de coordenadas
    transformer = pyproj.transformer.Transformer.from_crs(input_proj, output_proj)

    # Converte as coordenadas do sistema de coordenadas geocêntricas para o sistema de coordenadas geográficas
    longitude0, latitude0, alt = transformer.transform(b[0] * 1000.0, b[1] * 1000.0, b[2] * 1000.0, radians=True)'''

    lat = [(latitude0)]
    long = [(longitude0)]

    # comeco da integracao

    DELTAT = delt
    mu = 398600
    J2 = 1.08263e-3
    R_terra = 6371
    Time_step = delt
    passo = 10000
    ini_date = data
    n = num_orbitas
    T = T_orb*n
    t = np.linspace(0, Time_step, passo)
    data = [ini_date]
    solution = [[h0, ecc0, true_anomaly0, raan0, inc0, arg_per0, q0, q1, q2, q3, wx3_i, wy3_i, wz3_i]]
    time_simu = [0]
    cont = 0
    r = []
    v = []
    #while cont < T:

    for i in (range(0,int(T)+1, int(delt))):
        qi = [h0, ecc0, true_anomaly0, raan0, inc0, arg_per0, q0, q1, q2, q3, wx3_i, wy3_i, wz3_i]
        altitude = rp0 - R_terra
        latitude = lat[-1]
        longitude = long[-1]
        posicao = (h0**2/mu)*(1/(1-ecc0*np.cos(true_anomaly0)))
        air_density = densidade(ini_date, altitude, latitude, longitude)
        velocidade = (mu/h0)*np.sqrt(np.sin(true_anomaly0)**2 + (ecc0 + np.cos(true_anomaly0))**2)*1000.0
        v.append(velocidade/1000.0)
        massa = massa
        CD = 2.2
        Area_transversal = 0.1*0.1
        largura = largura
        comprimento = comprimento
        altura = altura
        sol = odeint(propagador, qi, t, args=(air_density, velocidade, massa, largura, comprimento, altura, CD, posicao, Area_transversal))
        solution.append(sol[-1])
        h0 = sol[-1][0]
        ecc0 = sol[-1][1]
        true_anomaly0 = sol[-1][2]
        raan0 = sol[-1][3]
        inc0 = sol[-1][4]
        arg_per0 = sol[-1][5]
        SMA = (h0**2/mu) * (1 / (1 - ecc0**2))
        rp0 = SMA*(1-ecc0)
        q0 = sol[-1][6]
        q1 = sol[-1][7]
        q2 = sol[-1][8]
        q3 = sol[-1][9]
        wx3_i = sol[-1][10]
        wy3_i = sol[-1][11]
        wz3_i = sol[-1][12]
        cont = cont + Time_step
        time_simu.append(cont)
        final_date = timedelta(seconds=Time_step)
        ini_date = ini_date + final_date
        data.append(ini_date)

        # Calculo da longitude e latitude no ECEF

        xp = (h0 ** 2 / mu) * (1 / (1 + ecc0 * np.cos(true_anomaly0))) * np.cos(true_anomaly0)
        yp = (h0 ** 2 / mu) * (1 / (1 + ecc0 * np.cos(true_anomaly0))) * np.sin(true_anomaly0)
        zp = 0
        r_p = [xp, yp, zp]

        lamb_e = lamb_e - ((2*np.pi)/(23*3600 + 56*60 + 4))*DELTAT

        '''X_ECEF = ((np.cos(lamb_e) * np.cos(arg_per0) - np.sin(lamb_e) * np.sin(arg_per0) * np.cos(inc0)) * xp
                 + (-np.cos(lamb_e) * np.sin(arg_per0) - np.sin(lamb_e) * np.cos(inc0) * np.cos(arg_per0)) * yp
                 + np.sin(lamb_e) * np.sin(inc0) * zp)

        Y_ECEF = ((np.sin(lamb_e) * np.cos(arg_per0) + np.cos(lamb_e) * np.cos(inc0) * np.sin(arg_per0)) * xp
                 + (-np.sin(lamb_e) * np.sin(arg_per0) + np.cos(lamb_e) * np.cos(inc0) * np.cos(arg_per0)) * yp
                 - np.cos(lamb_e) * np.sin(inc0) * zp)

        Z_ECEF = (np.sin(inc0) * np.sin(arg_per0) * xp
                 + np.sin(inc0) * np.cos(arg_per0) * yp
                 + np.cos(inc0) * zp)'''

        phi = lamb_e
        teta = inc0
        psi = arg_per0

        Q_rot = np.array([[np.cos(psi) * np.cos(phi) - np.sin(psi) * np.sin(phi) * np.cos(teta),
                           np.cos(psi) * np.sin(phi) + np.sin(psi) * np.cos(teta) * np.cos(phi),
                           np.sin(psi) * np.sin(teta)],
                          [-np.sin(psi) * np.cos(phi) - np.cos(psi) * np.sin(phi) * np.cos(teta),
                           -np.sin(psi) * np.sin(phi) + np.cos(psi) * np.cos(teta) * np.cos(phi),
                           np.cos(psi) * np.sin(teta)],
                          [np.sin(teta) * np.sin(phi), -np.sin(teta) * np.cos(phi), np.cos(teta)]])
        R_ECEF = np.dot(np.transpose(Q_rot), r_p)
        r.append(np.array(R_ECEF))

        latitude = ((np.arcsin(R_ECEF[2]/np.linalg.norm(R_ECEF))))

        longitude = ((np.arctan2(R_ECEF[1],R_ECEF[0])))

        '''import pyproj
        input_proj = pyproj.CRS.from_epsg(4328)

        # Define o sistema de coordenadas de saída (geográfico)
        output_proj = pyproj.CRS.from_epsg(4326)

        # Cria um objeto Transformer para realizar a transformação de coordenadas
        transformer = pyproj.transformer.Transformer.from_crs(input_proj, output_proj)

        # Converte as coordenadas do sistema de coordenadas geocêntricas para o sistema de coordenadas geográficas
        longitude, latitude, alt = transformer.transform(X_ECEF*1000.0, Y_ECEF*1000.0, Z_ECEF*1000.0, radians=True)'''

        lat.append(np.degrees(latitude))
        long.append(np.degrees(longitude))



    solucao = pd.DataFrame(solution, columns=['h', 'ecc', 'anomalia_verdadeira', 'raan', 'inc', 'arg_per', 'q0', 'q1', 'q2', 'q3', 'wx3', 'wy3', 'wz3'])
    solucao['X_perifocal'] = (solucao['h']**2/mu)*(1/(1 + solucao['ecc']*np.cos(solucao['anomalia_verdadeira'])))*np.cos(solucao['anomalia_verdadeira'])
    solucao['Y_perifocal'] = (solucao['h']**2/mu)*(1/(1 + solucao['ecc']*np.cos(solucao['anomalia_verdadeira'])))*np.sin(solucao['anomalia_verdadeira'])
    solucao['Z_perifocal'] = 0
    solucao['distancia'] = np.sqrt(solucao['X_perifocal']**2 + solucao['Y_perifocal']**2)

    df = pd.DataFrame()
    psi = []
    for i in range(0, len(solution), 1):
        q0 = solucao.iloc[i, 6]
        q1 = solucao.iloc[i, 7]
        q2 = solucao.iloc[i, 8]
        q3 = solucao.iloc[i, 9]
        a = np.arctan2(2*(q1*q3 + q0*q2), 2*(-q2*q3 + q0*q1))
        if np.linalg.norm(a) < 0.000000001:
            psi.append(0.0)
        elif np.linalg.norm(np.linalg.norm(a) - np.pi) < 0.000001:
            psi.append(0.0)
        else:
            psi.append(a)

    dfpsi = pd.DataFrame(np.unwrap(psi), columns=['PHI'])
    df = pd.concat([df, dfpsi], axis=1)

    teta = []
    for i in range(0, len(solution), 1):
        q0 = solucao.iloc[i, 6]
        q1 = solucao.iloc[i, 7]
        q2 = solucao.iloc[i, 8]
        q3 = solucao.iloc[i, 9]
        a = (2*(q0**2 + q3**2) - 1)
        if a >= 1:
            teta.append(np.arccos(1))
        elif a <= -1:
            teta.append(np.arccos(-1))
        else:
            teta.append(np.arccos(a))
    dfteta = pd.DataFrame(teta, columns=['TETA'])
    df = pd.concat([df, dfteta], axis=1)

    phi = []
    for i in range(0, len(solution), 1):
        q0 = solucao.iloc[i, 6]
        q1 = solucao.iloc[i, 7]
        q2 = solucao.iloc[i, 8]
        q3 = solucao.iloc[i, 9]
        a = np.arctan2((2*q1*q3 - 2*q0*q2), (2*q2*q3 + 2*q0*q1))

        if np.linalg.norm(a) < 0.000000001:
            phi.append(0.0)
        elif np.linalg.norm(np.linalg.norm(a) - np.pi) < 0.000001:
            phi.append(0.0)
        else:
            phi.append(a)

    dfphi = pd.DataFrame(np.unwrap(phi), columns=['PSI'])
    df = pd.concat([df, dfphi], axis=1)

    df['X_ECI'] = ((np.cos(solucao['raan'])*np.cos(solucao['arg_per']) - np.sin(solucao['raan'])*np.sin(solucao['arg_per'])*np.cos(solucao['inc']))*solucao['X_perifocal']

                        + (-np.cos(solucao['raan'])*np.sin(solucao['arg_per']) - np.sin(solucao['raan'])*np.cos(solucao['inc'])*np.cos(solucao['arg_per']))*solucao['Y_perifocal']

                        + np.sin(solucao['raan'])*np.sin(solucao['inc'])*solucao['Z_perifocal'])

    df['Y_ECI'] = ((np.sin(solucao['raan'])*np.cos(solucao['arg_per']) + np.cos(solucao['raan'])*np.cos(solucao['inc'])*np.sin(solucao['arg_per']))*solucao['X_perifocal']

                        + (-np.sin(solucao['raan'])*np.sin(solucao['arg_per']) + np.cos(solucao['raan'])*np.cos(solucao['inc'])*np.cos(solucao['arg_per']))*solucao['Y_perifocal']

                        - np.cos(solucao['raan'])*np.sin(solucao['inc'])*solucao['Z_perifocal'])

    df['Z_ECI'] = (np.sin(solucao['inc'])*np.sin(solucao['arg_per'])*solucao['X_perifocal']
                        + np.sin(solucao['inc'])*np.cos(solucao['arg_per'])*solucao['Y_perifocal']
                        + np.cos(solucao['inc'])*solucao['Z_perifocal'])


    df1 = pd.DataFrame(lat, columns=['latitude'])
    df = pd.concat([df, df1], axis=1)

    df2 = pd.DataFrame(long, columns=['longitude'])
    df = pd.concat([df, df2], axis=1)

    df3 = pd.DataFrame(data, columns=['Data'])
    df = pd.concat([df, df3], axis=1)

    df4 = pd.DataFrame(time_simu, columns=['Tempo'])
    df = pd.concat([df, df4], axis=1)

    '''from vetor_solar import beta_angle
    beta = [np.degrees(beta_angle(x, np.degrees(inc0), y)) for x,y in zip(df['Data'].to_list(), solucao['raan'].to_list())]
    df5 = pd.DataFrame(beta, columns=['Beta'])

    df = pd.concat([df,df5], axis=1)'''
    '''    df['r'] = np.sqrt((df['X_ECI'] ** 2 + df['Y_ECI'] ** 2 + df['Z_ECI'] ** 2))
    df['end'] = 'end'''
    import os.path
    #df.to_csv(os.path.join(f'beta{beta}/analise_{inclinacao}/results_{N}/propagador_orbital_{N}', f'dados_ECI_{N}.csv'), sep=',')

    #solucao.to_csv(os.path.join(f'beta{beta}/analise_{inclinacao}/results_{N}/propagador_orbital_{N}', f'solver_{N}.csv'), sep=',')
    r = pd.DataFrame(r, columns=['rx', 'ry', 'rz'])

    r['latitude'] = np.degrees(-np.arcsin(r['rz'] / (r['rx']**2 + r['ry']**2 + r['rz']**2)**0.5))
    r['longitude'] = np.degrees(np.arctan2(r['ry'], r['rx']))
    r['r'] = np.sqrt(r['rx']**2 + r['ry']**2 + r['rz']**2)
    r['vel'] = v
    r['vel_anali'] = np.sqrt(((mu/h0) * (1 + excentricidade*np.cos(solucao['anomalia_verdadeira'])))**2 + ((mu/h0) * excentricidade*np.sin(solucao['anomalia_verdadeira'])))
    r['end'] = 'end'
    #r.to_csv(os.path.join(f'beta{beta}/analise_{inclinacao}/results_{N}/propagador_orbital_{N}', f'ECEF_R_{N}.csv'), sep=',')

    return df


