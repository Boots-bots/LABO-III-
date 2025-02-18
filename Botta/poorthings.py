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

save_folder = ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Resonancia/"
Datos600 = pd.read_csv(save_folder+"RLC_Potencia(104u)(600).csv", index_col=["Frecuencias"])
Datos1200 = pd.read_csv(save_folder+"RLC_Potencia(104u)(1200).csv", index_col=["Frecuencias"])

frecuencias12 = np.unique(Datos1200.index)
frecuencias6 = np.unique(Datos600.index)

R6 = 600
R12 = 1200 #OHM
L = 0.01 #10mH
C = 104*10**(-9) #100nF

w6 = frecuencias6*2*np.pi
w12 = frecuencias12*2*np.pi

w0 = 1/(2*np.pi*np.sqrt(L*C))
dw6 = R6/(2*np.pi*L)
dw12 = R12/(2*np.pi*L)
Q6 = (1/R6)*np.sqrt(L/C)
Q12 = (1/R12)*np.sqrt(L/C)

amplitud_ch1 = []
amplitud_ch2 = []

amplitud_ch1_600 = []
amplitud_ch2_600 = []

for freq in frecuencias6:
    vch1_600 = Datos600.loc[freq]["VoltajeCH1"].values
    vch2_600 = Datos600.loc[freq]["VoltajeCH2"].values
    amplitud_ch1_600.append(np.max(vch1_600)-np.min(vch1_600))
    amplitud_ch2_600.append(np.max(vch2_600)-np.min(vch2_600))

amplitud_ch1_600 = np.array(amplitud_ch1_600)
amplitud_ch2_600 = np.array(amplitud_ch2_600)

for freq in frecuencias12:
    vch1 = Datos1200.loc[freq]["VoltajeCH1"].values
    vch2 = Datos1200.loc[freq]["VoltajeCH2"].values
    amplitud_ch1.append(np.max(vch1)-np.min(vch1))
    amplitud_ch2.append(np.max(vch2)-np.min(vch2))

amplitud_ch1 = np.array(amplitud_ch1)
amplitud_ch2 = np.array(amplitud_ch2)

Vrms1 = amplitud_ch1/np.sqrt(2)
# Vrms2 = amplitud_ch2/np.sqrt(2)
Vrms1_600 = amplitud_ch1_600/np.sqrt(2) 

corriente6 = Vrms1_600/np.sqrt((R6**2) + (w6*L - 1/(w6 * C))**2)
corriente12 = Vrms1/np.sqrt((R12**2) + (w12*L - 1/(w12 * C))**2) 

corr600m = corriente6*1000
corr1200m = corriente12*1000

stdI6 = [np.mean(std(Datos600, f)[1]) for f in frecuencias6]
stdI12 = [np.mean(std(Datos1200, f)[1]) for f in frecuencias12]

def I(w,r,l,c,V):
  return  V/np.sqrt((r**2) + (w*l - 1/(w * c))**2)

init_guess6 = [600, 0.01, 104*10**(-9), np.mean(Vrms1_600)]
init_guess12 = [1200, 0.01, 104*10**(-9), np.mean(Vrms1)]

pipi6, cici6 = curve_fit(I, w6, corriente6, sigma = stdI6, p0=init_guess6, absolute_sigma=True)
pipierr6 = np.sqrt(np.diag(cici6))
pipi12, cici12 = curve_fit(I, w12, corriente12, sigma = stdI12, p0=init_guess12, absolute_sigma=True)
pipierr12 = np.sqrt(np.diag(cici12))

ejex = np.linspace(np.min(w12), np.max(w12), 1000)

latex_corr = r"$|I| = \frac{|V|}{\sqrt{R^2 + \left( \omega L - \frac{1}{\omega C} \right)^2 }}$"

plt.figure()
plt.plot(ejex, I(ejex, *pipi6)*1000, "r", label = "Ajustes")
plt.plot(ejex, I(ejex, *pipi12)*1000, "r")
plt.errorbar(w6, corr600m, yerr=stdI6, fmt=".", label = f"Q = {Q6:.1f}")   
plt.errorbar(w12, corr1200m, yerr=stdI12, fmt=".", label = F"Q = {Q12:.1f}") 
plt.text(0.1, 0.85, latex_corr,
         transform=plt.gca().transAxes, fontsize=14, color="black")
plt.grid(True)
plt.title("Corriente en Resonancia para diferentes Q")
plt.ylabel("I [mA]")
plt.xlabel("w [rad/s]")
plt.legend()
plt.xscale("log")

Potencia6 = R6 * (corriente6**2) 
Potencia12 = R12 * (corriente12**2) 

def P(i,r):
    return r*i**2

stdP6 = [propagación(P, (corriente6[i],600), (stdI6[i],0.01)) for i in range(len(corriente6))]
std0P6 = [s/600 for s in stdP6]
stdP12 = [propagación(P, (corriente12[i],1200), (stdI12[i],0.01)) for i in range(len(corriente12))]
std1P2 = [s/1200 for s in stdP12]
n = np.mean(std0P6)
m = np.mean(std1P2)

std1 = float(n)*np.ones(len(w6))
std2 = float(m)*np.ones(len(w12))

def Potencia(w1, R1, L1, C1, V1):          
    return ((((R1 * (V1**2))/ (R1**2 + (w1*L1 - 1/(w1*C1))**2))))

popt6, pcov6 = curve_fit(Potencia, w6, Potencia6, sigma=std1, p0=init_guess6, absolute_sigma=True) 
perr6 = np.sqrt(np.diag(pcov6))
popt12, pcov12 = curve_fit(Potencia, w12, Potencia12, sigma=std2, p0=init_guess12, absolute_sigma=True) 
perr12 = np.sqrt(np.diag(pcov12))


PotenciaMM_600 = Potencia6*1000
PotenciaMM_1200 = Potencia12*1000   #mW


std11 = [d*1000 for d in std1]
std22 = [d*1000 for d in std2]

latex_pot = r"$P(\omega) = \frac{V_{ef}^2 R}{R^2 + \left( \omega L - \frac{1}{\omega C} \right)^2 }$"


plt.figure()
plt.plot(ejex, Potencia(ejex, *popt6)*1000, color="r", label = "Ajustes") #linestyle = "--
plt.plot(ejex, Potencia(ejex, *popt12)*1000, color="r")
plt.errorbar(w6, PotenciaMM_600, yerr=std11, fmt= "b.", label = f"Q = {Q6:.1f}")
plt.errorbar(w12, PotenciaMM_1200, yerr=std22, fmt= "g.", label = f"Q = {Q12:.1f}")
plt.text(0.1, 0.85, latex_pot,
         transform=plt.gca().transAxes, fontsize=14, color="black")
plt.title("Potencia en resonancia para diferentes Q")
plt.ylabel("Potencia [mW]")
plt.xlabel("w [rad/s]")
plt.xscale('log')
plt.legend()
plt.grid(True)

# # bondad
# Aj11 = chi2_pvalor(corriente6, stdI6, I(w6, *pipi6), ("R", "L", "C", "V"))         
# Aj21 = chi2_pvalor(corriente12, stdI12, I(w12, *pipi12), ("R", "L", "C", "V"))
# Aj12 = chi2_pvalor(Potencia6, std1, Potencia(w6, *popt6), ("R", "L", "C", "V"))         
# Aj22 = chi2_pvalor(Potencia12, std2, Potencia(w12, *popt12), ("R", "L", "C", "V"))      
# chi11 = Aj11[0]/Aj11[2]   
# pval11 = Aj11[1]              
# chi21 = Aj21[0]/Aj21[2]
# pval21 = Aj21[1]
# chi12 = Aj12[0]/Aj12[2]   
# pval12 = Aj12[1]              
# chi22 = Aj22[0]/Aj22[2]
# pval22 = Aj22[1]

# print(chi11, pval11)
# print(chi21, pval21)
# print(chi12, pval12)
# print(chi22, pval22)


print('Resultados del ajuste Corriente:')
print("600")
print('R = ' + str(pipi6[0]) + " ± " + str(pipierr6[0])) 
print('L = ' + str(pipi6[1]) + " ± " + str(pipierr6[1]))
print('C = ' + str(pipi6[2]) + " ± " + str(pipierr6[2]))
print('Vrms = ' + str(pipi6[3]) + " ± " + str(pipierr6[3]))
print("1200")
print('R = ' + str(pipi12[0]) + " ± " + str(pipierr12[0])) 
print('L = ' + str(pipi12[1]) + " ± " + str(pipierr12[1]))
print('C = ' + str(pipi12[2]) + " ± " + str(pipierr12[2]))
print('Vrms = ' + str(pipi12[3]) + " ± " + str(pipierr12[3]))
print('Resultados del ajuste Potencia:')
print("600")
print('R = ' + str(popt6[0]) + " ± " + str(perr6[0])) 
print('L = ' + str(popt6[1]) + " ± " + str(perr6[1]))
print('C = ' + str(popt6[2]) + " ± " + str(perr6[2]))
print('Vrms = ' + str(popt6[3]) + " ± " + str(perr6[3]))
print("1200")
print('R = ' + str(popt12[0]) + " ± " + str(perr12[0])) 
print('L = ' + str(popt12[1]) + " ± " + str(perr12[1]))
print('C = ' + str(popt12[2]) + " ± " + str(perr12[2]))
print('Vrms = ' + str(popt12[3]) + " ± " + str(perr12[3]))


plt.show(block=True)