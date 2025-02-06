import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*


import pandas as pd
carpeta = "H:\Mi unidad\Clase 2 osciloscopio CA ohm"
archivos = [f for f in os.listdir(carpeta) if f.endswith(".csv")]
med = []
 
for archivo in archivos:                                                                              # son 4 mediciones, 10Hz, 50kHz, 100Hz, 500 kHz
    ruta_csv = os.path.join(carpeta, archivo)                                                         #columna D (t) E (V) es CH1, columna J (t) K (V) CH2, (de una misma medicion D == J)
    datos = np.genfromtxt(ruta_csv, delimiter=",", skip_header=8, usecols=(3, 4, 9, 10))              # CH1 corresponde con Vg y CH2 con VR
    columna_D, columna_E, columna_J, columna_K = datos[:, 0], datos[:, 1], datos[:, 2], datos[:, 3]
    medn = columna_D, columna_E, columna_J, columna_K
    med.append(medn)

raux = 1000 # Ω (corregir)
R = 5000 # Ω

def i(Vgen,VR,raux):
    return (Vgen - VR)/raux

def modelo(it,R):
    return it*R




for m in range(4):
    ir = [i(med[m][1][j],med[m][3][j],raux) for j in range(len(med[m][0]))]
    Vr = med[m][3]
    std = 0.005*np.ones(len(med[m][0]))
    pop,cov = curve_fit(modelo, ir, Vr, (5000), std, absolute_sigma = True)
    print("----------------")
    print("med" + str(m))
    print("R =", pop[0], "±", np.sqrt(np.diag(cov))[0])
    ejex = np.linspace(np.min(ir), np.max(ir), 10000)
    plt.figure()
    plt.title("medicion: " + str(archivos[m]))
    plt.ylabel("Vr [V]")
    plt.xlabel("Ir [A]")
    plt.plot(ejex, modelo(ejex, pop[0]), "r")
    plt.errorbar(ir,Vr,yerr = std, fmt =".")
plt.show(block=True)

exit()
for i in range(4):
    plt.figure()
    plt.title("medicion: " + str(archivos[i]))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    plt.plot(med[i][0], med[i][1])
    plt.plot(med[i][2], med[i][3])
plt.show(block=True)