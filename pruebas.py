# codigo de pruebas generales
from common_settings import*

def calculadoraRLC(R,L,C):
    w0 = 1/np.sqrt(L*C)
    dw = R/L
    Q = np.sqrt(L/C)/R
    f = 2*np.pi*w0
    df = 2*np.pi*dw
    return f , df

r=600
c=104*10**-9
l=0.010

print(np.mean((31476.975345034574,32967.421168783185,33087.17805098989)), "±", np.std((31476.975345034574,32967.421168783185,33087.17805098989)))