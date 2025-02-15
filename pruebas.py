# codigo de pruebas generales
from common_settings import*

def calculadoraRLC(R,L,C):
    w0 = 1/np.sqrt(L*C)
    dw = R/L
    Q = np.sqrt(L/C)/R
    return w0, dw, Q

r=1200
c=0.000001
l=0.010
print(calculadoraRLC(r,l,c))