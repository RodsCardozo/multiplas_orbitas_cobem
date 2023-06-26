
"""
    Universidade Federal de Santa Catarina
    Laboratory of Applications and Research in Space - LARS
    Orbital Mechanics Division

    Título do Algoritmo= Dados de entrada para simulação orbital e radiação incidente
    Autor= Rodrigo S. Cardozo
    Versão= 0.0.1
    Data= 24/10/2022

"""
def calor_incidente(posicao_orientacao, radiacao_solar, radiacao_terra, emissividade_terra, absortividade_satelite,
                    refletividade_terra, data,n):

    """
    :param posicao_orientacao = Dataframe com a orientacao do cubesat e a sua posicao
    :param radiacao_solar = Intensidade da irradiacao solar
    :param radiacao_terra = Intensidade da IR da Terra
    :param emissividade_terra = valor da emissidade médio da Terra
    :param absortividade_satelite = valor média da absortividade de cada face do satelite
    :param refletividade_terra = valor da refletividade médio da Terra
    :param numero_divisoes = divisao da terra em n elementos de area
    :param n = numero de orbitas
    """
    print("Calculando calor")
    from fator_forma import fator_forma_classico as FS
    import numpy as np
    import pandas as pd
    import os, sys

    def createFolder(directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print('Error: Creating directory. ' + directory)
    createFolder(f'./results_{n}/radiacao_{n}')
    ''' Dados iniciais da orbita '''

    # propagador orbital

    prop_orb = posicao_orientacao  #propagador_orbital(data, SMA, ecc, Raan, arg_per, true_anomaly, inc, 1, delt, psi0, teta0, phi0, PSIP, TETAP,
                                   #PHIP, massa, largura, comprimento, altura, omegap)

    # Intensidade radiante do sol e terra e valores de emissividade

    Is = radiacao_solar          # radiacao solar
    Ir = radiacao_terra          # radiacao IR Terra
    e = emissividade_terra       # emissividade Terra
    ai = absortividade_satelite  # absortividade do satelite
    gama = refletividade_terra   # refletividade da Terra
    Raio_terra = 6371.0
    from vetor_solar import posicao_sol
    Vs = np.array(posicao_sol(data)) # vetor solar
    print(f'Vetor solar: {Vs/np.linalg.norm(Vs)}')
    Ni = [[1, 0, 0],
          [0, 1, 0],
          [-1, 0, 0],
          [0, -1, 0],
          [0, 0, -1],
          [0, 0, 1]]
    df1 = pd.DataFrame(Ni, columns=['x', 'y', 'z'])
    Posicao_orientacao = pd.concat([prop_orb, df1], axis=1)

    # determinacao da orientacao de cada face

    names = [['N1_X', 'N1_Y', 'N1_Z'],
             ['N2_X', 'N2_Y', 'N2_Z'],
             ['N3_X', 'N3_Y', 'N3_Z'],
             ['N4_X', 'N4_Y', 'N4_Z'],
             ['N5_X', 'N5_Y', 'N5_Z'],
             ['N6_X', 'N6_Y', 'N6_Z']]
    R = []
    phi = np.array(Posicao_orientacao['PHI'])
    teta = np.array(Posicao_orientacao['TETA'])
    psi = np.array(Posicao_orientacao['PSI'])

    for j in range(0, len(Ni), 1):
        for i in range(0, len(Posicao_orientacao), 1):
            A = Ni[j]

            PHI = phi[i]
            TETA = teta[i]
            PSI = psi[i]

            vetor = A


            Q_rot = np.array([[np.cos(PHI) * np.cos(PSI) - np.sin(PHI) * np.sin(PSI) * np.cos(TETA),
                               np.cos(PHI) * np.sin(PSI) + np.sin(PHI) * np.cos(TETA) * np.cos(PSI),
                               np.sin(PHI) * np.sin(TETA)],
                              [-np.sin(PHI) * np.cos(PSI) - np.cos(PHI) * np.sin(PSI) * np.cos(TETA),
                               -np.sin(PHI) * np.sin(PSI) + np.cos(PHI) * np.cos(TETA) * np.cos(PSI),
                               np.cos(PHI) * np.sin(TETA)],
                              [np.sin(TETA) * np.sin(PSI), -np.sin(TETA) * np.cos(PSI), np.cos(TETA)]])

            Q = np.dot(np.transpose(Q_rot), vetor)
            R1 = Q[0]
            R2 = Q[1]
            R3 = Q[2]
            R.append([np.array(R1), np.array(R2), np.array(R3)])

        df2 = pd.DataFrame(R, columns=names[j])
        R = []
        Posicao_orientacao = pd.concat([Posicao_orientacao, df2], axis=1)

    import os.path
    Posicao_orientacao.to_csv(os.path.join(f'./results_{n}/radiacao_{n}', f'Posicao_orientacao_{n}.csv'), sep=',')

    tupla1 = list(zip(Posicao_orientacao['X_ECI'], Posicao_orientacao['Y_ECI'], Posicao_orientacao['Z_ECI']))
    vetor_posicao = [np.array(tupla) for tupla in tupla1]

    '''Inicio do calculo de radiacao'''

    print('Calculando radiacao solar')
    Qs1 = []
    Qs2 = []
    Qs3 = []
    Qs4 = []
    Qs5 = []
    Qs6 = []
    for i in (range(0, len(vetor_posicao), 1)):

        PSI = np.arccos(np.dot(vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), Vs / np.linalg.norm(Vs)))
        QSI = np.arcsin(Raio_terra / np.linalg.norm((vetor_posicao[i])))

        if PSI + QSI < np.pi:

            A1 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Z')]])

            A2 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Z')]])

            A3 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Z')]])

            A4 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Z')]])

            A5 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Z')]])

            A6 = np.array(
                [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_X')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Y')],
                 Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Z')]])

            k1 = np.dot(A1/np.linalg.norm(A1), Vs/np.linalg.norm(Vs))
            k2 = np.dot(A2/np.linalg.norm(A2), Vs/np.linalg.norm(Vs))
            k3 = np.dot(A3/np.linalg.norm(A3), Vs/np.linalg.norm(Vs))
            k4 = np.dot(A4/np.linalg.norm(A4), Vs/np.linalg.norm(Vs))
            k5 = np.dot(A5/np.linalg.norm(A5), Vs/np.linalg.norm(Vs))
            k6 = np.dot(A6/np.linalg.norm(A6), Vs/np.linalg.norm(Vs))

            if k1 > 0:
                qs1 = ai * Is * k1
                Qs1.append(qs1)
            else:
                Qs1.append(0)
            if k2 > 0:
                qs2 = ai * Is * k2
                Qs2.append(qs2)
            else:
                Qs2.append(0)

            if k3 > 0:
                qs3 = ai * Is * k3
                Qs3.append(qs3)
            else:
                Qs3.append(0)
            if k4 > 0:
                qs4 = ai * Is * k4
                Qs4.append(qs4)
            else:
                Qs4.append(0)
            if k5 > 0:
                qs5 = ai * Is * k5
                Qs5.append(qs5)
            else:
                Qs5.append(0)
            if k6 > 0:
                qs6 = ai * Is * k6
                Qs6.append(qs6)
            else:
                Qs6.append(0)

        else:
            Qs1.append(0)
            Qs2.append(0)
            Qs3.append(0)
            Qs4.append(0)
            Qs5.append(0)
            Qs6.append(0)

    '''Radiacao de albedo incidente'''

    print('Calculando radiacao de albedo')

    Qalb1 = []
    Qalb2 = []
    Qalb3 = []
    Qalb4 = []
    Qalb5 = []
    Qalb6 = []

    for i in (range(0, len(vetor_posicao), 1)):

        A1 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Z')]])

        A2 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Z')]])

        A3 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Z')]])

        A4 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Z')]])

        A5 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Z')]])

        A6 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Z')]])

        phi = (np.dot(Vs/np.linalg.norm(Vs), vetor_posicao[i] / np.linalg.norm(vetor_posicao[i])))
        PSI = np.arccos(np.dot(vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), Vs / np.linalg.norm(Vs)))
        QSI = np.arcsin(Raio_terra / np.linalg.norm((vetor_posicao[i])))

        if PSI < np.pi/2:

            d1 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A1 / np.linalg.norm(A1)))
            d2 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A2 / np.linalg.norm(A2)))
            d3 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A3 / np.linalg.norm(A3)))
            d4 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A4 / np.linalg.norm(A4)))
            d5 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A5 / np.linalg.norm(A5)))
            d6 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A6 / np.linalg.norm(A6)))
            VP = (np.linalg.norm(-vetor_posicao[i]))

            FS1 = FS(VP, d1)
            FS2 = FS(VP, d2)
            FS3 = FS(VP, d3)
            FS4 = FS(VP, d4)
            FS5 = FS(VP, d5)
            FS6 = FS(VP, d6)

            Qalb1.append(ai * gama * Is * FS1 * phi) #ai * gama * Is *

            Qalb2.append(ai * gama * Is * FS2 * phi)

            Qalb3.append(ai * gama * Is * FS3 * phi)

            Qalb4.append(ai * gama * Is * FS4 * phi)

            Qalb5.append(ai * gama * Is * FS5 * phi)

            Qalb6.append(ai * gama * Is * FS6 * phi)
        else:

            Qalb1.append(0.0)

            Qalb2.append(0.0)

            Qalb3.append(0.0)

            Qalb4.append(0.0)

            Qalb5.append(0.0)

            Qalb6.append(0.0)

    ''' Radiacao da terra '''
    print('Calculando radiacao da terra')

    Qrad1 = []
    Qrad2 = []
    Qrad3 = []
    Qrad4 = []
    Qrad5 = []
    Qrad6 = []

    for i in (range(0, len(vetor_posicao), 1)):
        A1 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N1_Z')]])

        A2 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N2_Z')]])

        A3 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N3_Z')]])

        A4 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N4_Z')]])

        A5 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N5_Z')]])

        A6 = np.array(
            [Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_X')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Y')],
             Posicao_orientacao.iloc[i, Posicao_orientacao.columns.get_loc('N6_Z')]])
        #
        D1 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A1/np.linalg.norm(A1))
        if D1 > 1.0:
            d1 = np.arccos(1.0)
        elif D1 < -1.0:
            d1 = np.arccos(-1.0)
        else:
            d1 = np.arccos(D1)

        D2 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A2/np.linalg.norm(A2))
        if D2 > 1.0:
            d2 = np.arccos(1.0)
        elif D2 < -1.0:
            d2 = np.arccos(-1.0)
        else:
            d2 = np.arccos(D2)

        D3 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A3/np.linalg.norm(A3))
        if D3 > 1.0:
            d3 = np.arccos(1.0)
        elif D3 < -1.0:
            d3 = np.arccos(-1.0)
        else:
            d3 = np.arccos(D3)

        D4 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A4/np.linalg.norm(A4))
        if D4 > 1.0:
            d4 = np.arccos(1.0)
        elif D4 < -1.0:
            d4 = np.arccos(-1.0)
        else:
            d4 = np.arccos(D4)

        D5 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A5/np.linalg.norm(A5))
        if D5 > 1.0:
            d5 = np.arccos(1.0)
        elif D5 < -1.0:
            d5 = np.arccos(-1.0)
        else:
            d5 = np.arccos(D5)

        D6 = np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A6/np.linalg.norm(A6))
        if D6 > 1.0:
            d6 = np.arccos(1.0)
        elif D6 < -1.0:
            d6 = np.arccos(-1.0)
        else:
            d6 = np.arccos(D6)
        '''d2 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A2/np.linalg.norm(A2)))
        d3 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A3/np.linalg.norm(A3)))
        d4 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A4/np.linalg.norm(A4)))
        d5 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A5/np.linalg.norm(A5)))
        d6 = np.arccos(np.dot(-vetor_posicao[i] / np.linalg.norm(vetor_posicao[i]), A6/np.linalg.norm(A6)))'''
        VP = np.linalg.norm(vetor_posicao[i])

        FS1 = FS(VP, d1)
        FS2 = FS(VP, d2)
        FS3 = FS(VP, d3)
        FS4 = FS(VP, d4)
        FS5 = FS(VP, d5)
        FS6 = FS(VP, d6)

        Qrad1.append(e * Ir *(FS1)) #e * Ir *

        Qrad2.append(e * Ir *(FS2))

        Qrad3.append(e * Ir *(FS3))

        Qrad4.append(e * Ir *(FS4))

        Qrad5.append(e * Ir *(FS5))

        Qrad6.append(e * Ir *(FS6))

    rad_sol = []
    for i in range(0, len(Qs1), 1):
        rad_sol.append([Qs1[i], Qs2[i], Qs3[i], Qs4[i], Qs5[i], Qs6[i]])

    Q_sol = pd.DataFrame(rad_sol, columns=['Solar 1', 'Solar 2', 'Solar 3', 'Solar 4', 'Solar 5', 'Solar 6'])

    rad_alb = []
    for i in range(0, len(Qalb1), 1):
        rad_alb.append([Qalb1[i], Qalb2[i], Qalb3[i], Qalb4[i], Qalb5[i], Qalb6[i]])
    Q_alb = pd.DataFrame(rad_alb, columns=['Albedo 1', 'Albedo 2', 'Albedo 3', 'Albedo 4', 'Albedo 5', 'Albedo 6'])

    rad_terra = []
    for i in range(0, len(Qrad1), 1):
        rad_terra.append([Qrad1[i], Qrad2[i], Qrad3[i], Qrad4[i], Qrad5[i], Qrad6[i]])
    Q_terra = pd.DataFrame(rad_terra, columns=['IR Terra 1', 'IR Terra 2', 'IR Terra 3', 'IR Terra 4', 'IR Terra 5', 'IR Terra 6'])

    QT = pd.concat([Q_sol, Q_alb], axis=1)
    QT = pd.concat([QT, Q_terra], axis=1)
    QT['Total 1'] = QT['Solar 1'] + QT['Albedo 1'] + QT['IR Terra 1']
    QT['Total 2'] = QT['Solar 2'] + QT['Albedo 2'] + QT['IR Terra 2']
    QT['Total 3'] = QT['Solar 3'] + QT['Albedo 3'] + QT['IR Terra 3']
    QT['Total 4'] = QT['Solar 4'] + QT['Albedo 4'] + QT['IR Terra 4']
    QT['Total 5'] = QT['Solar 5'] + QT['Albedo 5'] + QT['IR Terra 5']
    QT['Total 6'] = QT['Solar 6'] + QT['Albedo 6'] + QT['IR Terra 6']

    import os.path
    QT.to_csv(os.path.join(f'./results_{n}/radiacao_{n}', 'Calor_Incidente.csv'), sep=',')
    print('Fim')

    return QT