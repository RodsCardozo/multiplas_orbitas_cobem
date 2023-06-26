
import numpy as np
import fator_forma_interno as ff
def ff_interno(face):
    """
      :param df: DataFrame with nodes
      :return: Array with all the Form Factors for a square cavity
      """
    N = len(face)  # número de nós
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # A = np.dot(lump.iloc[i,7], lump.iloc[j,7])
            # print(A)
            if i == j:
                M[i, j] = 0

            elif face[i].n == face[j].n:
                if face[i].n == 'i':
                    M[i, j] = ff.fator_paralelo(X=face[i].Ly, Y=face[i].Lz, L=(face[i].Lx + face[i].Lx))
                elif face[i].n == 'j':
                    M[i, j] = ff.fator_paralelo(X=face[i].Lx, Y=face[i].Lz, L=(face[i].Ly + face[i].Ly))
                else:
                    M[i, j] = ff.fator_paralelo(X=face[i].Lx, Y=face[i].Ly, L=(face[i].Lz + face[i].Lz))
            else:
                if face[i].n == 'i' and face[j].n == 'k':
                    X = face[i].Ly
                    Y = face[i].Lz
                    Z = face[j].Lx
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'i' and face[j].n == 'j':
                    X = face[i].Lz
                    Y = face[i].Ly
                    Z = face[j].Lx
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'i':
                    X = face[i].Lz
                    Y = face[i].Lx
                    Z = face[j].Ly
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'k':
                    X = face[i].Lx
                    Y = face[i].Lz
                    Z = face[j].Ly
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'i':
                    X = face[i].Ly
                    Y = face[i].Lx
                    Z = face[j].Lz
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'j':
                    X = face[i].Lx
                    Y = face[i].Ly
                    Z = face[j].Lz
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

    return M


""" Fator de Gebhart """


def fator_gebhart(face, F):
    """
      :param df: Dataframe with nodes
      :param F: Array with all Form Factors
      :return B: Array with Gebhart factors
      """
    F = ff_interno(face)
    N = len(F)
    M = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i == k:
                d = 1
            else:
                d = 0
            M[i, k] = (1 - face[k].e) * F[i, k] - d
    b = []
    for j in range(N):
        b.append(-face[j].e * F[j])
    B = []
    for i in range(N):
        B.append(np.linalg.solve(M, b[i]))
    return B
