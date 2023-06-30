import numpy as np
def vetor(A):
    if A['n'] == 'i':
        A = [1,0,0]
    elif A['n'] == 'j':
        A = [0,1,0]
    else:
        A = [0,0,1]
    return A
def fator_paralelo(X, Y, L):
    x = X/L
    y = Y/L
    Fij = (2/(np.pi*x*y))*(np.log((((1 + x**2)*(1 + y**2))/(1 + x**2 + y**2))**(1/2))
                           + x*(1 + y**2)**(1/2)*np.arctan(x/(1+y**2)**(1/2))
                           + y*(1 + x**2)**(1/2)*np.arctan(y/(1+x**2)**(1/2))
                           - x*np.arctan(x) - y*np.arctan(y))
    return Fij

def fator_perpendicular(X, Y, Z):
    h = Z/X
    w = Y/X

    Fij = 1/(np.pi*w) * (w*np.arctan(1/w) + h*np.arctan(1/h) - (h**2 + w**2)**(1/2)*np.arctan(1/(h**2 + w**2)**(1/2))
                      + 0.25*np.log((((1 + w**2)*(1 + h**2))/(1 + w**2 + h**2))*((w**2*(1 + w**2 + h**2))/((1 + w**2)*(w**2 + h**2)))**(w**2) *
                                    ((h**2*(1 + h**2 + w**2))/((1 + h**2)*(h**2 + w**2)))**(h**2)))
    return Fij





