
import numpy as np
import fator_forma_interno as ff
def ff_interno(face, cavidade):
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
                    M[i, j] = ff.fator_paralelo(X=face[i].X, Y=face[i].Y, L=(cavidade.Lx))
                elif face[i].n == 'j':
                    M[i, j] = ff.fator_paralelo(X=face[i].X, Y=face[i].Y, L=(cavidade.Ly))
                else:
                    M[i, j] = ff.fator_paralelo(X=face[i].X, Y=face[i].Y, L=(cavidade.Lz))
            else:
                if face[i].n == 'i' and face[j].n == 'k':
                    X = face[i].X
                    Y = face[i].Y
                    Z = face[j].X
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'i' and face[j].n == 'j':
                    X = face[i].Y
                    Y = face[i].X
                    Z = face[j].X
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'i':
                    X = face[i].Y
                    Y = face[i].X
                    Z = face[j].X
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'j' and face[j].n == 'k':
                    X = face[i].X
                    Y = face[i].Y
                    Z = face[j].X
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'i':
                    X = face[i].X
                    Y = face[i].Y
                    Z = face[j].Y
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)

                elif face[i].n == 'k' and face[j].n == 'j':
                    X = face[i].X
                    Y = face[i].Y
                    Z = face[j].Y
                    M[i, j] = ff.fator_perpendicular(X=X, Y=Y, Z=Z)
    return M


""" Fator de Gebhart """


def fator_gebhart2(face, cavidade):
    """
      :param df: Dataframe with nodes
      :param F: Array with all Form Factors
      :return B: Array with Gebhart factors
      """
    F = ff_interno(face, cavidade)
    print(F)
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
        print(face[j].e)
        print(F[j])
        b.append(-face[j].e * F[j])
        print('')
        print(b[j])
    B = []
    for i in range(N):
        B.append(np.linalg.solve(M, b[i]))
    return np.array(B)
