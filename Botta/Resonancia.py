# imports de cosas, por como esta armado el repositorio el archivo common_settings donde estan todas los imports y cosas
# generales se carga de esta manera (hay un codigo para cargar cosas de forma similar de drive y en jupyter colab)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

# funciones a necesitar
def std(df, f):
    "calcula un error de los datos a partir de la escala de los datos"
    subset = df.loc[f]                                                    # identifica los datos de una frecuencia f
    resolucion_ch1 = subset["ResolucionVCH1"].values                      # extrae los valores de las columnas de escala de CH1 y CH2 de esa frecuencia
    resolucion_ch2 = subset["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 / 2                                          # calcula el error como la mitad de la escala
    std_ch2 = resolucion_ch2 / 2
    return std_ch1, std_ch2                                               # devuelve los errores

def gen(t,V,f,φ):                          # esto es solo un seno para usar en los ajustes pero puesto bonito en contexto 
    return V*np.sin(2*np.pi*f*t + φ)


# este codigo es para las primeras mediciones, la que llamamos de "transferencia"
# es el mismo codigo copiado que en el analisis de filtros RC y es el mismo que iba a usar 
# para los que llamamos de "Potencia", que son los de 104nF y 1200 Ω Υ 600 Ω
# lo separe en codigos distintos solo por comodidad, pero es exactamente lo mismo

# codigo Resonancia == Resonancia2  (solo cargan datos distintos)

# el codigo de filtros es Filtros (FilstrosSim, el de las simulaciones (anterior), es distinto en la forma en que se cargan los datos, pq son distintos)
# el codigo de Resonancia2 es el de "Potencia", las ultimas dos mediciones (104nF)(1200) y (104nF)(600)


# Clase 6

#Transferencia                                 # apunte de cosas usadas
 #C 1uF (0.000001) /F 0.010H /R 1200 ohm  
 # w0 = 10000, f = 1591.5, dw = 120000, Q = 0.08)

ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Resonancia/"         # ruta donde tengo guardados los archivos, cambiar por lo que sea
                                                                                          # si ya se tiene el codigo para cargar common_settings desde drive se usa un codigo similar para cargar datos de drive
                                                                                          # poner la ruta de drive, o subir los datos y usar el codigo de cargar los datos de ahi (si se usa colab) (no me acuerdo como es)
# cargando los datos                                            
dfT = pd.read_csv(ruta + "RLC_Transferencia(1uF).csv", index_col=["Frecuencias"])         # cargo en un dataframe (excel) de pandas los datos usando como indice la columna frecuencias 
                                                                                          #                                                               (para poder llamarlos a partir de esta)
frecuencias = np.unique(dfT.index)            # extraigo en una lista los valores de frecuencias

# Ajustes para "reducir ruido" y ver valores de amplitud y fase y las diferencias
Chis1 = []                    # creo listas vacias de cosas q voy a guardar
pvals1 = []
Chis2 = []
pvals2 = []

dif_fase = []
dif_fase_std = []

Vin = []
dif_amp = []
dif_amp_std = []

# Gran For
for frec in frecuencias:                         # un for que pasa por todas las frec en la lista de frecuencias (equivalente a:     for i in range(len(frecuencias)):    )
    tiempo = dfT.loc[frec]["Tiempo"].values      #     entonces trabajando con cada frecuencia                                         frec = frecuencias[i]
    vch1 = dfT.loc[frec]["VoltajeCH1"].values    #     extraigo los valores de las columnas de tiempo ch1 y ch2
    vch2 = dfT.loc[frec]["VoltajeCH2"].values

    amplitud_ch1 = (np.max(vch1)-np.min(vch1))   # busca el maximo y el minimo de la lista voltaje de cada canal y los resta (saca Vpp) 
    amplitud_ch2 = (np.max(vch2)-np.min(vch2))
    amplitud_ch1 = np.array(amplitud_ch1)        # convierte la lista en un array de numpy, simplemente por "comodidad"
    amplitud_ch2 = np.array(amplitud_ch2)     # el profe en su codigo utiliza esto ya como medida definitiva de amplitud, yo lo uso como valor inicial para el ajuste despues

    stdIn, stdOut = std(dfT, frec)        # calculo errores para las mediciones (en base a la escala) usando la funcion de antes 

#in  (CH1)                # AJUSTES de suavizado
    popIn, pcovIn = curve_fit(gen, tiempo, vch1, sigma = stdIn, p0 = (amplitud_ch1/2, frec, 0), absolute_sigma=True)
    popstdIn = np.sqrt(np.diag(pcovIn))
#out (CH2)
    popOut, pcovOut = curve_fit(gen, tiempo, vch2, sigma = stdOut, p0 = (amplitud_ch2/2, frec, 0), absolute_sigma=True)
    popstdOut = np.sqrt(np.diag(pcovOut))

# # chequeo                                                 # esto sirve para chequear que los datos se estan cargando bien y los ajustes tienen sentido 
#     plt.figure()                                          # se recomienda desmarcarlo aunque sea una vez, despues volver a comentarlo (#)
#     plt.title(f"frecuencia: {frec:.1f} Hz")               # ( OJO! grafica todos los datos ) 
#     plt.xlabel("tiempo [s]")
#     plt.ylabel("V")
#     plt.errorbar(tiempo,vch1,stdIn, fmt=".", color="b", label = "CH1 Vin")
#     plt.errorbar(tiempo,vch2,stdOut, fmt=".", color="g", label = "CH2 Vout")
#     plt.plot(tiempo, gen(tiempo,*popIn), "r")
#     plt.plot(tiempo, gen(tiempo,*popOut), "r")
#     plt.legend()
#     plt.show(block=True)

#Bondad                                                                          # calculo del chi y el pvalor de los ajustes (chequear las funciones en common_settings)
    Aj1 = chi2_pvalor(vch1, stdIn, gen(tiempo, *popIn), ("V", "f", "φ"))         # el asterisco en *popIn lo que hace es desplegar todos los valores de esa lista 
    Aj2 = chi2_pvalor(vch2, stdOut, gen(tiempo, *popOut), ("V", "f", "φ"))       #                                 ( equivalente a popIn[0], popIn[1], popIn[...], ...) (lo grade q sea)
    chi1 = Aj1[0]/Aj1[2]               # la funcion chi2_pvalor devuelve chi2, pval, gl (grados de libertad)
    pval1 = Aj1[1]                     # Aca lo que hago es escribir que es cada cosa, el chi/gl es para normarizarlo y que deba dar ­­∼1
    chi2 = Aj2[0]/Aj2[2]
    pval2 = Aj2[1]
    Chis1.append(chi1)                 # guardo el valor de la frecuencia en la que estamos en una lista para todas las frecuencias
    pvals1.append(pval1)  
    Chis2.append(chi2)  
    pvals2.append(pval2)  

#Fase
    dif_fase.append(popOut[2] - popIn[2])                                         # calculo el desfasaje a partir de los parametros y el error mediante propagación
    dif_fase_std.append(np.sqrt((popstdOut[2]*chi2)**2 + (popstdIn[2]*chi1)**2))  #             le agregue el multiplicar los errores por el chi del ajuste de suavizado

#Amplitud                                              # lo mismo pero con amplitud
    Vin.append(popIn[0])  # Necesito guardar el Vin para usar desp en corriente y potencia   #(esto no esta en el codigo de filtros (leer aclaracion del principio))
    dif_amp.append(popOut[0]/popIn[0])
    dif_amp_std.append(np.sqrt(((popstdOut[0]*chi2)/popIn[0])**2 + (popOut[0]*(popstdIn[0]*chi1)/popIn[0]**2)**2))

# aca ya estamos fuera del for

dif_fase = np.array(dif_fase)         # convierte todo a np.arrays como antes
dif_fase_std = np.array(dif_fase_std)
dif_amp = np.array(dif_amp)
dif_amp_std = np.array(dif_amp_std)

w = frecuencias*2*np.pi                                 # paso frecuencia en Hz a angular rad/s
ejex = np.linspace(np.min(w), np.max(w), 10000)         # creo un linspace (espacio lineal) para graficar

def I(w,V,R,L,C):                                          # aca defino las funciones de corriente y potencia como estan en el apunte
    return modulo(V)/np.sqrt(R**2 + (w*L - 1/(w*C))**2)         # la funcion modulo es personal (common_settings) se puede cambiar por otra ya hecha de numpy (pero ns como es)
def P(w,V,R,L,C):
    return (R*(V/np.sqrt(2))**2)/(R**2 + (w*L - 1/(w*C))**2)

def Zab(w,R,L,C):                                    # aca definio la funcion de impedancia vista desde la fuente 
    return np.sqrt(R**2 + (w*L - 1/(w*C))**2)        # pensando en calcular la corriente a partir de esto (I = V/Z) pero termina siendo lo mismo que la funcion I

corr = [I(w[i], Vin[i], 1200, 0.010, 0.000001) for i in range(len(Vin))]         # calculo la corriente con las frecuencias y los voltajes 
corr = [i*1000 for i in corr] #mA  cambio de unidad para q quede lindo  //        (esta escrito de forma abreviada en una unica linea, es lo mismo que crear la lista vacia fuera del for, hacer el for y desp .append)

pot = [1200*i**2 for i in corr]      # calculo la potencia como R*I**2     (devuelta es una forma mas corta de escribir el for, esta vez es similar a corr[i] for i in range(len(corr)))


# Graficos

# Corriente
plt.figure()
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0")     # grafico una linea (plt.ax(v)line(...))   donde se supone que está w0 
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw")    #                       (vertical)            (aca se deberia cambiar luego por un pop_Corr[0], o como sea que el ajuste calcula w0)
plt.plot(frecuencias, corr ,'.b', label="Mediciones") # grafico datos        # tmb linea vertical para el ancho de banda (w0 + dw)
plt.ylabel("I [mA]")                                                                   
plt.xlabel("Frecuencias [Hz]")                              # el eje x está en frecuencias (Hz) por lo que w y todo lo relacionado hay que dividirlo por 2π 
plt.legend()  #esto grafica los labels                      # si se cambia por w ya no hace falta 

                      # habria que calcular la propagación y el plot(frec...) cambiar por errorbar(frec...)

#Potencia
plt.figure() 
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0")  #lo mismo que antes
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw") 
plt.plot(frecuencias, pot, '.b', label="Mediciones")
# plt.plot(ejex, P(ejex,0.5,1200,0.010,0.000001), "r", label ="modelo")    # esto es un intento de graficar la función
plt.ylabel("Potencia [W]")
plt.xlabel("Frecuencias [Hz]")
plt.legend()


# Transferencia
plt.figure()
plt.axhline(y=0.7, color='g', linestyle='--', label = "70%") # grafica linea horizontal (h) 
plt.axvline(x= 10000/(2*np.pi), color='g', linestyle='--', label = "w0") 
plt.axvline(x= 130000/(2*np.pi), color='g', linestyle='--', label = "dw") 
plt.errorbar(frecuencias, dif_amp, dif_amp_std, fmt='.', color="b", label="Mediciones")  # aca hay un errorbar escrito completo (usarlo para escribir cualquier otro)
plt.ylabel("Transferencia")                                                         
plt.xlabel("Frecuencias [Hz]")
plt.legend()


plt.show(block=True)    # muestra lo graficado (colab lo hace automatico, no hace falta aclararlo, y el block=True frena el codigo hasta q se cierre la pestaña q se abre con la imagen)
                        # literalmente no tiene sentido en colab (aca es necesario, es mas estricto, lo mismo que el print())

exit()  # esto termina la ejecución  

# estos son otros graficos para el chuqueo de los ajustes de suavizado, grafican en histogramas los Chis y los pvals
# terminaron aca abajo por comodidad, pero se recomienda despues de cargar los datos y ejecutar el  "Gran For", ver todos los graficos de 
# datos y ajustes y luego printear Chis1 Chis2 pvals1 pvals2 para ver que valores tienen y luego ver estos graficos

#Chis
plt.figure()
plt.title(r"Bondad de ajuste $(\chi^2)$")
plt.xlabel("Chi/nu")
plt.hist(Chis2, bins=int(np.sqrt((len(Chis2)))), alpha=0.5, label=f"CH2: {np.mean(Chis2)}")
plt.hist(Chis1, bins=int(np.sqrt(len(Chis1))), alpha=0.5, label=f"CH1: {np.mean(Chis1)}")
plt.legend()

#Pvals
plt.figure()
plt.title("Bondad de Ajuste (P-valor)")
plt.xlabel("P-valor")
plt.hist(pvals1, bins=int(np.sqrt(len(pvals1))), alpha=0.5, label=f"p.val1: {np.mean(pvals1)}")
plt.hist(pvals2, bins=int(np.sqrt(len(pvals2))), alpha=0.5, label=f"p.val2: {np.mean(pvals2)}")
plt.legend()
plt.show(block=True)