import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*


def errorMulti(L, x, n, Dms):
    return x*L + n*Dms

ampere = [181, 160.9, 122, 98.5, 82.5, 71, 62.1, 55.3, 50, 25, 16.7, 12.5, 10, 8.3, 7.1, 6.2, 5.5, 4.9] #mA #181, 
ampere = [a/1000 for a in ampere]
oms = [20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]     # 20, 

stdA = [errorMulti(A, (1.2/100), 1, 0.0001) for A in ampere]
# stdA= [0.005, 0.0022719999999999997, 0.0020308, 0.001564, 0.0012820000000000002, 0.00109, 0.0009519999999999999, 0.0008452, 0.0007636, 0.0007000000000000001, 0.0004, 0.0003004, 0.00025, 0.00022, 0.00019960000000000003, 0.0001852, 0.0001744, 0.000166, 0.0001588]
# stdA = 0.00000001*np.ones(len(ampere))

def pot(i,R):
    return R*i**2
potencia = [pot(ampere[i], oms[i]) for i in range(len(ampere))]

def modelo(R,r,V):
    return ((V/(r + R))**2)*R

std = np.array([propagación(pot, (ampere[i], oms[i]), (stdA[i],0.1)) for i in range(len(ampere))], dtype = float)

# pop = Minimizer(modelo, oms, potencia, std, (7,5), metodo = "trust-constr")
pop,cov = curve_fit(modelo, oms, potencia, (6, 5), std, absolute_sigma = True)

ymod = [modelo(R, pop[0], pop[1]) for R in oms]
chi,pv,nu = chi2_pvalor(potencia, std, ymod, ("r","V"))

print(chi/nu,pv)
# print(pop)
print(pop, np.sqrt(np.diag(cov)))

ejex = np.linspace(0,1000,10000)
ejex2 = np.linspace(20,1000,10000)

def om(R,V):
    return V/R

pop2,cov2 = curve_fit(om, oms, ampere, (5), stdA, absolute_sigma = True)
print("V ", pop2[0])

oms.pop(0)
ampere.pop(0)
potencia.pop(0)
std2 = std.tolist()
std2.pop(0)
stdA.pop(0)

plt.figure()
plt.title("Medición de máxima Potencia")
plt.xlabel("Resistencia de carga [Ω]")
plt.ylabel("Potencia [W], Corriente [A]")
plt.errorbar(oms, potencia, yerr = std2, fmt = ".", color = "b", label = "Potencia transferida")
plt.axvline(pop[0], linestyle = "-.", color = "brown", label = "Resistencia interna 6.4 ± 0.3 Ω" ) 
plt.plot(ejex, modelo(ejex, pop[0], pop[1]), "r", label = "Ajuste") 
# plt.plot(ejex2, om(ejex, pop2[0]), "g", label = "Ajuste") 
plt.errorbar(oms, ampere, yerr = stdA, fmt = ".", color = "g", label = "Medición de corriente" )
plt.legend()

plt.show(block = True)