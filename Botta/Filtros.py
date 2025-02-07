import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/data_test1.csv"

#usar arrays d1

exit()
import pandas as pd

escalalog = np.logspace(1, 4, 30)

ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/"
df = pd.read_csv(ruta + "data_test1.csv", index_col=["Frequency"])   

for col in ["Time1", "Time2", "V1", "V2"]:
    df[col] = df[col].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ") if isinstance(x, str) else x)


time1 = df["Time1"].values
v1 =  df["V1"].values

plt.figure()
for i in range(30):
    plt.plot(time1[i], v1[i], ".")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señales por frecuencia")

plt.show(block=True)


