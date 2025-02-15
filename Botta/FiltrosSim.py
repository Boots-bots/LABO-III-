import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

R = 10      # Ω
C = 0.00001 # F  (10μF)

ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Sim/PasaBajos(RC)3.txt" # , 2, 3

# Inicializar variables
data_dict = {}

# Leer el archivo
with open(ruta, "r") as file:
    current_block = []
    current_freq = None

    for line in file:
        line = line.strip()
        if line.startswith("Step Information:"):
            # Guardar el bloque anterior en el diccionario
            if current_block and current_freq is not None:
                data_dict[current_freq] = np.array(current_block, dtype=float)
                current_block = []
            
            # Extraer la frecuencia
            parts = line.split("Frec=")
            if len(parts) > 1:
                current_freq = parts[1].split()[0].strip()  # Asegurar que no tenga espacios
        elif line and not line.startswith("time"):  # Ignorar encabezados
            values = list(map(float, line.split()))
            current_block.append(values)

# Guardar el último bloque
if current_block and current_freq is not None:
    data_dict[current_freq] = np.array(current_block, dtype=float)

# Crear listas para DataFrame
df_list = []

for freq, data in data_dict.items():
    temp_df = pd.DataFrame(data, columns=["time", "V(a)", "V(c)"])
    temp_df["Frequency"] = freq  # Agregar la frecuencia como columna
    df_list.append(temp_df)

df = pd.concat(df_list).set_index(["Frequency", "time"])

def convert_frequency(freq_str):
    if freq_str.endswith('K'):
        return float(freq_str[:-1]) * 1e3
    elif freq_str.endswith('M'):
        return float(freq_str[:-1]) * 1e6
    else:
        return float(freq_str)

frecuencias = df.index.get_level_values(0).unique()
frecuencias_num = np.array([convert_frequency(freq) for freq in frecuencias])
frecuencias_num = np.array([2*np.pi*f for f in frecuencias_num])


XmxmIn, YmxmIn = [], []
XminIn, YminIn = [], []
XmxmOut,YmxmOut = [], []
XminOut, YminOut = [], []

for frec in frecuencias:
    subset = df.loc[frec]
    x = subset.index.to_numpy()  # Tiempo
    yout = subset["V(a)"].to_numpy()  # Voltaje out
    yin = subset["V(c)"].to_numpy()  # Voltaje in
    xpin, ypin = máximos(x, yin)
    XmxmIn.append(xpin)
    YmxmIn.append(ypin)
    xmin, ymin = mínimos(x, yin)
    XminIn.append(xmin)
    YminIn.append(ymin)
    xpout, ypout = máximos(x, yout)
    XmxmOut.append(xpout)
    YmxmOut.append(ypout)
    xmout, ymout = mínimos(x, yout)
    XminOut.append(xmout)
    YminOut.append(ymout)


# for i in range(len(frecuencias)):
#     subset = df.loc[frecuencias[i]]
#     plt.figure()
#     plt.title(f"V(a) para {frecuencias[i]}Hz")
#     plt.xlabel("Tiempo (s)")
#     plt.ylabel("Voltaje (V)")
#     plt.plot(subset.index, subset["V(a)"], ".-b", label=f"Vout(a)")
#     plt.plot(subset.index, subset["V(c)"], ".-r", label=f"Vin(c)")
#     plt.plot(XmxmIn[i], YmxmIn[i], "og", label="Máximo")
#     plt.plot(XminIn[i], YminIn[i], "og", label="Minimo")
#     plt.plot(XmxmOut[i], YmxmOut[i], "om")
#     plt.plot(XminOut[i], YminOut[i], "om")
#     plt.legend()
#     plt.show(block=True)

diferenciasF = []
diferenciasV = []
for i in range(len(frecuencias)):
    #desfasajes
    DifFrec = []
    if len(XmxmIn[i]) == len(XmxmOut[i]):
        for j in range(len(XmxmIn[i])):
            DifFrec.append(XmxmIn[i][j] - XmxmOut[i][j])
    else:
        min_len = min(len(XmxmIn[i]), len(XmxmOut[i]))
        for j in range(min_len):
            DifFrec.append(XmxmIn[i][j] - XmxmOut[i][j])
    if len(XminIn[i]) == len(XminOut[i]):
        for j in range(len(XminIn[i])):
            DifFrec.append(XminIn[i][j] - XminOut[i][j])
    else:
        min_len = min(len(XminIn[i]), len(XminOut[i]))
        for j in range(min_len):
            DifFrec.append(XminIn[i][j] - XminOut[i][j])
    diferenciasF.append(DifFrec)

    #transferencias
    DifV = []
    if len(YmxmIn[i]) == len(YmxmOut[i]):
        for j in range(len(YmxmIn[i])):
            DifV.append(YmxmOut[i][j]/YmxmIn[i][j])
    else:
        min_len = min(len(YmxmIn[i]), len(YmxmOut[i]))
        for j in range(min_len):
            DifV.append(YmxmOut[i][j]/YmxmIn[i][j])
    if len(YminIn[i]) == len(YminOut[i]):
        for j in range(len(YminIn[i])):
            DifV.append(YminOut[i][j]/YminIn[i][j])
    else:
        min_len = min(len(YminIn[i]), len(YminOut[i]))
        for j in range(min_len):
            DifV.append(YminOut[i][j]/YminIn[i][j])
    diferenciasV.append(DifV)

#corrección
diferenciasF.pop(0)
diferenciasV.pop(0)

DiferenciaFase = [np.mean(diferenciasF[i]) for i in range(len(diferenciasF))]
stdDifFase = [np.std(diferenciasF[i]) for i in range(len(diferenciasF))]

DiferenciaVol = [np.mean(diferenciasV[i]) for i in range(len(diferenciasV))]
stdDifVol = [np.std(diferenciasV[i]) for i in range(len(diferenciasV))]


def fase(w,w0):
    return -np.arctan(w/w0)

def Transferencia(w,w0):
    return 1/np.sqrt(1+(w/w0)**2)

def Atenuación(w,w0):
    return 20*np.log10(Transferencia(w,w0)) 
def Atenuación2(T):
    return 20*sp.log(T,10) 


def ang(θ):
    "Angulos en radianes a grados"
    return θ*180/np.pi

w0 = 1/(R*C)
w = frecuencias_num 
# corr
w = w[1:]              
wred = w/w0

stdDifFaseRad = np.array([propagación(ang, DiferenciaFase[i], stdDifFase[i]) for i in range(len(DiferenciaFase))])

Atenuation = [Atenuación2(DiferenciaVol[i]) for i in range(len(DiferenciaVol))] 
AttStd = [propagación(Atenuación2, DiferenciaVol[i], stdDifVol[i]) for i in range(len(DiferenciaVol))]

n = 10000
ejex = np.linspace(np.min(w), np.max(w), int(n))
ejexred = np.linspace(np.min(wred), np.max(wred), int(n))

#lineal

plt.figure()
plt.title("Desfasaje en Filtro Pasabajos RC")
plt.xlabel("ω/ω0")
plt.ylabel("Φ [grados]")
plt.axhline(y=-45, color='g', linestyle='--', label = "-45°") 
plt.plot(ejexred, ang(fase(ejex,w0)), "r", label ="Modelo")
plt.errorbar(wred, ang(w*DiferenciaFase), ang(stdDifFaseRad), fmt=".", label = "Mediciones")
plt.legend()


plt.figure()
plt.title("Transferencia en Filtro Pasabajos RC")
plt.xlabel("ω/ω0")
plt.ylabel("T")
plt.axhline(y=0.7, color='g', linestyle='--', label = "-45°") 
plt.errorbar(wred, DiferenciaVol, stdDifVol, fmt=".", label = "Mediciones")
plt.plot(ejexred, Transferencia(ejex,w0), "r", label = "Modelo")
plt.legend()

#logaritmica

plt.figure()
plt.title("Desfasaje en Filtro Pasabajos RC")
plt.xlabel("ω/ω0")
plt.ylabel("Φ [grados]")
plt.xscale('log')
plt.axhline(y=-45, color='g', linestyle='--', label = "-45°") 
plt.plot(ejexred, ang(fase(ejex,w0)), "r", label = "Modelo")
plt.errorbar(wred, ang(w*DiferenciaFase), ang(stdDifFaseRad) ,fmt =".", label = "Mediciones")
plt.legend()

plt.figure()
plt.title("Atenuación en Filtro Pasabajos RC")
plt.xlabel("ω/ω0")
plt.ylabel("A [dB]")
plt.xscale('log') 
plt.axhline(y=-3, color='g', linestyle='--', label = "-3dB")
plt.plot(ejexred, Atenuación(ejex,w0), "r", label = "Modelo")
plt.errorbar(wred, Atenuation, AttStd, fmt =".", label = "Mediciones")
plt.legend()

plt.show(block=True)



