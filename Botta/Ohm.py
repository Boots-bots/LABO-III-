import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

carpeta = "H:\Mi unidad\Clase 2 osciloscopio CA ohm"
archivos = [f for f in os.listdir(carpeta) if f.endswith(".csv")]
med = []
 
for archivo in archivos:                                                                              # son 4 mediciones, 10Hz, 50kHz, 100Hz, 500 kHz
    ruta_csv = os.path.join(carpeta, archivo)                                                         #columna D (t) E (V) es CH1, columna J (t) K (V) CH2, (de una misma medicion D == J)
    datos = np.genfromtxt(ruta_csv, delimiter=",", skip_header=8, usecols=(3, 4, 9, 10))              # CH1 corresponde con Vg y CH2 con VR
    columna_D, columna_E, columna_J, columna_K = datos[:, 0], datos[:, 1], datos[:, 2], datos[:, 3]
    medn = columna_D, columna_E, columna_J, columna_K
    med.append(medn)

#Datos

# for i in range(4):
#     plt.figure()
#     plt.title("medicion: " + str(archivos[i]))
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Voltaje [V]")
#     plt.plot(med[i][0], med[i][1],".", label = "CH1")
#     plt.plot(med[i][2], med[i][3],".", label = "CH2")
#     plt.legend()
# plt.show(block=True)

raux = 1000 # Ω 
R = 5000 # Ω

def modelo(i,R):
    i = np.array(i)
    return i*R

#DC

voltaje = [0.995, 1.495, 1.98, 2.48, 2.98, 3.48, 3.98, 4.48, 4.97] #V
corriente = [0.000192, 0.000291, 0.000388, 0.000485, 0.000583, 0.000682, 0.000777, 0.000875, 0.000971] #A
std = 0.005 * np.ones(len(voltaje))

pop,cov = curve_fit(modelo, corriente, voltaje, (R), std, absolute_sigma = True)
# pop = Minimizer(modelo, corriente, voltaje, std, (R), metodo = "Newton-CG")

ymod = [modelo(i, pop[0]) for i in corriente]
chi,pv,nu = chi2_pvalor(voltaje, std, ymod, ("R"))
print("--------------------------")
print("R =", pop[0], "±", np.sqrt(np.diag(cov))[0])
print("chi2N: ", chi/nu, "±", np.sqrt(2/nu))
print("chi2: ", chi, "±", np.sqrt(2*nu))
print("pvalor: ", pv)

s = np.sqrt(2/nu)
ejex = np.linspace(np.min(corriente), np.max(corriente), 10000)
plt.figure()
plt.title("Medición Ley Ohm (DC)")
plt.ylabel("V [V]")
plt.xlabel("I [A]")
plt.plot(ejex, modelo(ejex, pop[0]), "r", label = "Ajuste")
plt.errorbar(corriente, voltaje, yerr = std, fmt =".", label = "Mediciones")
plt.text(0.8, 0.1, f"$\chi^2$ = {chi/nu:.1f} ± {s:.1f}", 
         transform=plt.gca().transAxes, fontsize=10, color="black")
plt.legend()
plt.show(block=True)
exit()
#AC
def iraux(Vgen,VR,raux):
    return (Vgen - VR)/raux

for m in range(4):
    iR = [iraux(med[m][1][j],med[m][3][j],raux) for j in range(len(med[m][0]))]
    VR = med[m][3]
    std = 0.005*np.ones(len(med[m][0]))
    pop = 0
    pop,cov = curve_fit(modelo, iR, VR, (R), std, absolute_sigma = True)
    pop = Minimizer(modelo, iR, VR, std, (R), metodo = "CG")
    print("----------------")
    print("med" + str(m))
    print("R =", pop[0], "±", np.sqrt(np.diag(cov))[0])
    ymod = [modelo(i, pop[0]) for i in iR]
    chi,pv,nu = chi2_pvalor(VR, std, ymod, ("R"))
    print("chi2N: ", chi/nu, "±", np.sqrt(2/nu))
    print("chi2: ", chi, "±", np.sqrt(2*nu))
    print("pvalor: ", pv)
    ejex = np.linspace(np.min(iR), np.max(iR), 1000000)
    plt.figure()
    plt.title("medicion: " + str(archivos[m]))
    plt.ylabel("Vr [V]")
    plt.xlabel("Ir [A]")
    plt.plot(ejex, modelo(ejex, pop[0]), "r", label = "Ajuste")
    plt.errorbar(iR,VR,yerr = std, fmt =".", label = "Mediciiones")
    plt.legend()
plt.show(block=True)

exit()
