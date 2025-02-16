import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

def std(df, f):
    subset = df.loc[f]
    resolucion_ch1 = subset["ResolucionVCH1"].values
    resolucion_ch2 = subset["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 / 2
    std_ch2 = resolucion_ch2 / 2
    return std_ch1, std_ch2

def gen(t,V,f,φ):
    return V*np.sin(2*np.pi*f*t + φ)

# Clase 6 Med 2

#Transferencia 
 #C 104nF /F 0.010 H /R 1200 ohm y 600 ohms
 # w0 = 10000, f = 1591.5, dw = 120000, Q = 0.08)

ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Resonancia/"

df1200 = pd.read_csv(ruta + "RLC_Potencia(104u)(1200).csv", index_col=["Frecuencias"])
df600 = pd.read_csv(ruta + "RLC_Potencia(104u)(600).csv", index_col=["Frecuencias"])

frecuencias1200 = np.unique(df1200.index)
frecuencias600 = np.unique(df600.index)

ERRE = (1200, 600)

Chis1 = []
pvals1 = []
Chis2 = []
pvals2 = []

dif_fase = []
dif_fase_std = []

Vin = []
dif_amp = []
dif_amp_std = []

for frec in frecuencias600:
    tiempo = df600.loc[frec]["Tiempo"].values      ############################     # LUGARES DONDE HAY QUE CAMBIAR DE 600 A 1200 SI SE QUIERE EJECUTAR LOS OTROS DATOS
    vch1 = df600.loc[frec]["VoltajeCH1"].values
    vch2 = df600.loc[frec]["VoltajeCH2"].values

    amplitud_ch1 = (np.max(vch1)-np.min(vch1))
    amplitud_ch2 = (np.max(vch2)-np.min(vch2))
    amplitud_ch1 = np.array(amplitud_ch1)
    amplitud_ch2 = np.array(amplitud_ch2)

    stdIn, stdOut = std(df600, frec)         ######################

#in  (CH1)
    popIn, pcovIn = curve_fit(gen, tiempo, vch1, sigma = stdIn, p0 = (amplitud_ch1/2, frec, 0), absolute_sigma=True)
    popstdIn = np.sqrt(np.diag(pcovIn))
#out (CH2)
    popOut, pcovOut = curve_fit(gen, tiempo, vch2, sigma = stdOut, p0 = (amplitud_ch2/2, frec, 0), absolute_sigma=True)
    popstdOut = np.sqrt(np.diag(pcovOut))

# chequeo
    plt.figure()
    plt.title(f"frecuencia: {frec:.1f} Hz")
    plt.xlabel("tiempo [s]")
    plt.ylabel("V")
    plt.errorbar(tiempo,vch1,stdIn, fmt=".", color="b", label = "CH1 Vin")
    plt.errorbar(tiempo,vch2,stdOut, fmt=".", color="g", label = "CH2 Vout")
    plt.plot(tiempo, gen(tiempo,*popIn), "r")
    plt.plot(tiempo, gen(tiempo,*popOut), "r")
    plt.legend()
    plt.show(block=True)

#Bondad
    Aj1 = chi2_pvalor(vch1, stdIn, gen(tiempo, *popIn), ("V", "f", "φ"))
    Aj2 = chi2_pvalor(vch2, stdOut, gen(tiempo, *popOut), ("V", "f", "φ"))
    chi1 = Aj1[0]/Aj1[2]
    pval1 = Aj1[1]
    chi2 = Aj2[0]/Aj2[2]
    pval2 = Aj2[1]
    Chis1.append(chi1)  
    pvals1.append(pval1)  
    Chis2.append(chi2)  
    pvals2.append(pval2)  

#Fase
    dif_fase.append(popOut[2] - popIn[2])
    dif_fase_std.append(np.sqrt((popstdOut[2]*chi2)**2 + (popstdIn[2]*chi1)**2))

#Amplitud
    Vin.append(popIn[0])
    dif_amp.append(popOut[0]/popIn[0])
    dif_amp_std.append(np.sqrt(((popstdOut[0]*chi2)/popIn[0])**2 + (popOut[0]*(popstdIn[0]*chi1)/popIn[0]**2)**2))

dif_fase = np.array(dif_fase)
dif_fase_std = np.array(dif_fase_std)
dif_amp = np.array(dif_amp)
dif_amp_std = np.array(dif_amp_std)

w = frecuencias600*2*np.pi                                     ###########
ejex = np.linspace(np.min(w), np.max(w), 10000)

def I(w,V,R,L,C):
    return modulo(V)/np.sqrt(R**2 + (w*L - 1/(w*C))**2)
def P(w,V,R,L,C):
    return (R*(V/np.sqrt(2))**2)/(R**2 + (w*L - 1/(w*C))**2)

def Zab(w,R,L,C):
    return np.sqrt(R**2 + (w*L - 1/(w*C))**2)

corr = [I(w[i],Vin[i],1200,0.010,0.000001) for i in range(len(Vin))]
corr = [i*1000 for i in corr] #mA

pot = [1200*i**2 for i in corr]

plt.figure()
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0") 
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw") 
plt.plot(frecuencias600, corr ,'.b', label="Mediciones") #prop                    #################
plt.ylabel("I [mA]")
plt.xlabel("Frecuencias [Hz]")
plt.legend()

plt.figure()
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0") 
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw") 
plt.plot(frecuencias600, pot, '.b', label="Mediciones") #prop                       ######################
# plt.plot(ejex, P(ejex,0.5,1200,0.010,0.000001), "r", label ="modelo")
plt.ylabel("Potencia [W]")
plt.xlabel("Frecuencias [Hz]")
plt.legend()

plt.show(block=True)

plt.figure()
plt.axhline(y=0.7, color='g', linestyle='--', label = "70%") 
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0") 
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw") 
plt.errorbar(frecuencias600, dif_amp, dif_amp_std, fmt='.', color="b", label="Mediciones")         #############################
plt.ylabel("Transferencia")
plt.xlabel("Frecuencias [Hz]")
plt.legend()

plt.show(block=True)

exit()
plt.figure()
plt.title(r"Bondad de ajuste $(\chi^2)$")
plt.xlabel("Chi/nu")
plt.hist(Chis2, bins=int(np.sqrt((len(Chis2)))), alpha=0.5, label=f"CH2: {np.mean(Chis2)}")
plt.hist(Chis1, bins=int(np.sqrt(len(Chis1))), alpha=0.5, label=f"CH1: {np.mean(Chis1)}")
plt.legend()

plt.figure()
plt.title("Bondad de Ajuste (P-valor)")
plt.xlabel("P-valor")
plt.hist(pvals1, bins=int(np.sqrt(len(pvals1))), alpha=0.5, label=f"p.val1: {np.mean(pvals1)}")
plt.hist(pvals2, bins=int(np.sqrt(len(pvals2))), alpha=0.5, label=f"p.val2: {np.mean(pvals2)}")
plt.legend()
plt.show(block=True)