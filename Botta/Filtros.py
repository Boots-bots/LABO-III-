import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*


ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/PasaBajos(RC).txt"

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


frecuencias = df.index.get_level_values(0).unique()

# frecuencias = ["90K"]

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

diferencias = []
for i in range(len(XmxmIn)):
    DifFrec = []
    for j in range(len(XmxmIn[i])):
        DifFrec.append(XmxmIn[i][j] - XmxmOut[i][j])
    # for j in range(len(XminOut[i])):
    #     DifFrec.append(XminIn[i][j] - XminOut[i][j])
    diferencias.append(DifFrec)

Diferencias = [np.mean(diferencias[i]) for i in range(len(diferencias))]

def fase(w,w0):
    return -np.arctan(w/w0)

w0 = 1/(10*0.00001)
w = np.linspace(0,100000,101)

plt.figure()
plt.plot(w, fase(w,w0), ".")
plt.plot(w, w*Diferencias,".")
plt.show(block=True)


exit()
subset = df.loc[frec]
plt.figure()
plt.plot(subset.index, subset["V(a)"], ".b", label=f"Vout(a)")
plt.plot(subset.index, subset["V(c)"], ".r", label=f"Vin(c)")
plt.plot(XmxmIn, YmxmIn, ".g", label="Máximo")
plt.plot(XminIn, YminIn, ".g", label="Minimo")
plt.plot(XmxmOut, YmxmOut, ".g", label="Máximo")
plt.plot(XminOut, YminOut, ".g", label="Minimo")

plt.show(block=True)
exit()
for frec in frecuencias:
    subset = df.loc[frec]
    plt.figure()   
    plt.plot(subset.index, subset["V(a)"], ".b", label=f"Vout(a)")
    plt.plot(subset.index, subset["V(c)"], ".r", label=f"Vin(c)")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Voltaje (V)")
    plt.title(f"V(a) para {frec}Hz")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)


