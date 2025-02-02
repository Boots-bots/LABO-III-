import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

##################################################################################
def Minimizer(f,x_data,y_data,std,parametros_iniciales, metodo = None, opciones = None):
    def error(parametros):
        y_mod = f(x_data, *parametros)
        return np.sum(((y_data - y_mod)/std)**2)

    def jacobiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        return np.array([(error(parametros + epsilon * np.eye(1, len(parametros), k)[0]) - error(parametros)) / epsilon for k in range(len(parametros))], dtype = float)

    def hessiano(parametros):
        epsilon = np.sqrt(np.finfo(float).eps)
        n = len(parametros)
        hess = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                ei = np.eye(1, n, i)[0] * epsilon
                ej = np.eye(1, n, j)[0] * epsilon
                hess[i, j] = (error(parametros + ei + ej) - error(parametros + ei) - error(parametros + ej) + error(parametros)) / (epsilon ** 2)
        return np.array([hess], dtype = float)

    jac = jacobiano if metodo in ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-krylov', 'trust-exact'] else None
    hess = hessiano if metodo in ['trust-ncg', 'trust-krylov', 'trust-exact'] else None
    
    resultado = minimize(error, parametros_iniciales, method=metodo, jac=jac, hess=hess, options=opciones)
    
    return resultado.x
#################################################################################################


ampere = (181, 160.9, 122, 98.5, 82.5, 71, 62.1, 55.3, 50, 25, 16.7, 12.5, 10, 8.3, 7.1, 6.2, 5.5, 4.9) #mA
ampere = [a/1000 for a in ampere]
oms = (20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)

stdA = 0.005*np.ones(len(ampere))
# std = (0.0005, 0.05, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0005, 0.0005, 0.005, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05)

potencia = [(ampere[i]**2)*oms[i] for i in range(len(ampere))]
def pipupipu(i,R):
    return R*i**2

def modelo(R,r):
    return ((5/(r + R))**2)*R

std = [propagación(pipupipu, (ampere[i], oms[i]), (stdA[i],stdA[i])) for i in range(len(ampere))]

pop = Minimizer(modelo, oms, potencia, std, (30), metodo = "Powell")
# pop,cov = curve_fit(modelo, oms, potencia, (30), std, absolute_sigma = True)

ymod = [modelo(R, pop[0]) for R in oms]
chi,pv,nu = chi2_pvalor(potencia, std, ymod, ("r"))

print(chi,pv)
print(pop)
# print(pop, np.sqrt(np.diag(cov)))

ejex = np.linspace(0,1000,10000)

plt.figure()
plt.title("Medición de maxima Potencia")
plt.xlabel("Resistencia de carga [Ω]")
plt.ylabel("Potencia [W]")
plt.axvline(30, linestyle = "-.", color = "brown") 
plt.errorbar(oms, potencia, yerr = std, fmt = ".", color = "b")
plt.plot(ejex, modelo(ejex, pop[0]), "r")


plt.figure()
plt.title("Medición de maxima Potencia")
plt.xlabel("Resistencia de carga [Ω]")
plt.ylabel("Corriente [A]")
plt.errorbar(oms, ampere, yerr = std, fmt = ".", color = "b")

plt.show(block = True)