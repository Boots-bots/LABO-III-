import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

def std(df):
    resolucion_ch1 = df["ResolucionVCH1"].values
    resolucion_ch2 = df["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 #/2
    std_ch2 = resolucion_ch2 #/2
    return std_ch1, std_ch2

# datos

save_folder = ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Transitorio/"
#indices
rcn = (11,12,21,22,31,32,41,42,51,52,61,62,71,72,81,82,91,92,101,102)
rln = (str("051"),str("052"),str("091"),str("092"),11,12,171,172,21,22,271,272,31,32,371,372,41,42,471,472) #,51,52,61,62
rlcn = (1,22,2,15,3,4,14,13,5,6,7,12,19,199,8,9,16,166,10,17,18,11)

DatosRC=[]
DatosRL=[]
DatosRLC=[]

DatosRC = [pd.read_csv(save_folder + "/RC/" +"RC" + str(n) + ".csv") for n in rcn]
DatosRL = [pd.read_csv(save_folder + "/RL/" + "RL" + str(n) + ".csv") for n in rln]
DatosRLC = [pd.read_csv(save_folder + "/RLC/" + "RLC" + str(n) + ".csv") for n in rlcn]

# cortado

def cortadora(df):
    tiempo = df["Tiempo"]
    vch1 = df["VoltajeCH1"]
    indices = [0]
    for i in range(len(tiempo)-1):
        if (vch1[i+1]-vch1[i]) > 0.4:
            indices.append(i + 1)
        elif (vch1[i+1]-vch1[i]) < -0.4:
            indices.append(i + 1)
    indices.append(len(tiempo))
    df_cortados = [df.iloc[indices[j]:indices[j + 1]] for j in range(len(indices) - 1)]
    return df_cortados

RCcortados = [cortadora(dRC) for dRC in DatosRC]
RLcortados = [cortadora(dRL) for dRL in DatosRL]
RLCcortados = [cortadora(dRLC) for dRLC in DatosRLC]

# Limpieza de puntas

for i in range(len(RCcortados)):
    if i in (4,6,10):
        if i == 10:
            pass
        else:
            RCcortados[i] = [item for idx, item in enumerate(RCcortados[i]) if idx not in (0,)]   
    else:
        RCcortados[i] = [item for idx, item in enumerate(RCcortados[i]) if idx not in (0, len(RCcortados[i])-1)]
for i in range(len(RLcortados)):
    if i in (0,1,4,5,12,13):
            RLcortados[i] = [item for idx, item in enumerate(RLcortados[i]) if idx not in (0,)]   
    else:
        RLcortados[i] = [item for idx, item in enumerate(RLcortados[i]) if idx not in (0, len(RLcortados[i])-1)]
for i in range(len(RLCcortados)):
    if i in (3,6,7,12,13,16,17):
        RLCcortados[i] = [item for idx, item in enumerate(RLCcortados[i]) if idx not in (0,)]   
    else:
        RLCcortados[i] = [item for idx, item in enumerate(RLCcortados[i]) if idx not in (0, len(RLCcortados[i])-1)]


for m in range(len(RLcortados)):
    for c in range(len(RLcortados[m])):
        df = RLcortados[m][c]
        vch2 = df["VoltajeCH2"]
        
        if m in (0, 13):
            # Orden inverso: mínimo, máximo
            if c % 2 == 0:
                index = vch2.idxmin()
            else:
                index = vch2.idxmax()
        else:
            # Orden normal: máximo, mínimo
            if c % 2 == 0:
                index = vch2.idxmax()
            else:
                index = vch2.idxmin()
        
        # Recortar el DataFrame desde el índice encontrado
        RLcortados[m][c] = df.loc[index:]

#Errores

RCcortadosStd = []
for j in range(len(RCcortados)):           # cantidad de experimentos
    rcstd = []
    for i in range(len(RCcortados[j])):    # cantidad de ciclos
        df = RCcortados[j][i]     
        vch1std, vch2std = std(df)  
        u = (vch1std,vch2std)
        rcstd.append(u)
    RCcortadosStd.append(rcstd)

RLcortadosStd = []
for j in range(len(RLcortados)):         
    rlstd = []
    for i in range(len(RLcortados[j])):    
        df = RLcortados[j][i]       
        vch1std, vch2std = std(df)  
        u = (vch1std,vch2std)
        rlstd.append(u)
    RLcortadosStd.append(rlstd)

RLCcortadosStd = []
for j in range(len(RLCcortados)):       
    rlcstd = []
    for i in range(len(RLCcortados[j])): 
        df = RLCcortados[j][i]       
        vch1std, vch2std = std(df)  
        u = (vch1std,vch2std)
        rlcstd.append(u)
    RLCcortadosStd.append(rlcstd)

# Ajustes

def Vrc(t,R,C,V0,c):   #V0 fuente Vc(0) sin corriente sobre el capacitor Q/C
    return V0*(1 - np.exp(-t/(R*C))) + c

def Vrl(t,R,L,V0,c):
    return V0*np.exp(-t*R/L) + c

def Vrlc(t,R,L,V0,b,c,n):
    if n == 0: # sub
        return
    if n == 1: #critrico
        return
    if n == 2: #sobre
        return

#pops
Rc = (1200,1200,2200,2200,3200,3200,4200,4200,5200,5200,6200,6200,7200,7200,8200,8200,9200,9200,10200,10200) # Ω
Rl = (500,500,900,900,1200,1200,1700,1700,2200,2200,2700,2700,3200,3200,3700,3700,4200,4200,4700,4700) #,5200,5200,6200,6200
Rlc = (157.5,240,250,387,500,620,640,1000,1180,1700,2003,2100,2110,2110,3020,4000,4700,4700,6000,6400,8390,10000)
Cr = 1 * 10**(-6) # F    (μ)
Crl = 8.9 * 10**(-9) # F (n)
L = 0.01  # H

POPRC = []           # array de arrays de experimentos 
POPRCstd = []
CHIRC = []
PVRC = []
datarc = []

POPRL = []           
POPRLstd = []
CHIRL = []
PVRL = []
datarl = []

POPRLC = []           
POPRLCstd = []
CHIRLC = []
PVRLC = []
datarlc = []

########RC####################

# for j in range(len(RCcortados)):  # para cada experimento 
#     Pop = []
#     Popstd= []
#     Chi = []
#     Pv = []
#     for i in range(len(RCcortados[j])):  # para cada recorte
#         df = RCcortados[j][i]
#         tiempo = df["Tiempo"]
#         vch1 = df["VoltajeCH1"]       
#         vch2 = df["VoltajeCH2"]
#         vch2std = RCcortadosStd[j][i][1]

#         vch1 = vch1 + 0.5
#         vch2 = vch2 + 0.5

#         p0 = (Rc[j], Cr, np.mean(vch1), 0)

#         bounds = ([0, 0, -np.inf, -np.inf],[100000, 1, np.inf, np.inf])
#         pop, cov = curve_fit(Vrc, tiempo, vch2, p0 = p0, sigma = vch2std, absolute_sigma = True, bounds = bounds) 
#         popstd = np.sqrt(np.diag(cov))

#         Pop.append(pop)          # # los meto todos de corrido los de un mismo experimento (no separados en recortes)
#         Popstd.append(popstd)

#         chi, pv, n = chi2_pvalor(vch2, vch2std, Vrc(tiempo, *pop), ("R", "C", "V0", "c"))
#         Chi.append(chi/n)
#         Pv.append(pv)

#         datarc.append({
#             "Medición": j,
#             "Recorte": i,
#             "Parametros": pop,
#             "Std": popstd,
#             "Chi": chi / n,
#             "Pv": pv
#         })    

#######################RL#################################

for j in range(len(RLcortados)):  # para cada experimento 
    Pop = []
    Popstd= []
    Chi = []
    Pv = []
    for i in range(len(RLcortados[j])):  # para cada recorte
        df = RLcortados[j][i]
        tiempo = df["Tiempo"]
        vch1 = df["VoltajeCH1"]       
        vch2 = df["VoltajeCH2"]
        vch2std = RLcortadosStd[j][i][1]

        vch1 = vch1
        vch2 = vch2

        p0 = (Rl[j], L, np.mean(vch1),0)

        bounds = ([0, 0, -np.inf, -np.inf],[100000, 1, np.inf, np.inf])
        pop, cov = curve_fit(Vrl, tiempo, vch2, p0 = p0, sigma = vch2std, absolute_sigma = True, bounds = bounds) 
        popstd = np.sqrt(np.diag(cov))

        Pop.append(pop)          # # los meto todos de corrido los de un mismo experimento (no separados en recortes)
        Popstd.append(popstd)

        chi, pv, n = chi2_pvalor(vch2, vch2std, Vrl(tiempo, *pop), ("R", "L", "V0", "c"))
        Chi.append(chi/n)
        Pv.append(pv)

        datarl.append({
            "Medición": j,
            "Recorte": i,
            "Parametros": pop,
            "Std": popstd,
            "Chi": chi / n,
            "Pv": pv
        })  
        
    POPRL.append(Pop)
    POPRLstd.append(Popstd)
    CHIRL.append(Chi)
    PVRL.append(Pv)

##################RLC#########################



######################################

# dfRC = pd.DataFrame(datarc)
# dfRC.set_index(["Medición", "Recorte"], inplace=True)

dfRL = pd.DataFrame(datarl)
dfRL.set_index(["Medición", "Recorte"], inplace=True)



# dfRL.to_csv(save_folder+"dfRL.csv", index=False, header=True)

# for j in range(len(RLCcortados)):
#     plt.figure()
#     plt.title("Medición: RLC" + str(rlcn[j]))
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Voltaje [V]")
#     # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH1"], label = "CH1")
#     # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH2"], label = "CH2")
#     for i in range(len(RLCcortados[j])):
#         df = RLCcortados[j][i]
#         tiempo = df["Tiempo"]
#         vch1 = df["VoltajeCH1"]
#         vch2 = df["VoltajeCH2"]
#         ejex= np.linspace(np.min(tiempo), np.max(tiempo), 1000)
#         s = 2        
#         plt.errorbar(tiempo,vch1, yerr = RLCcortadosStd[j][i][0], fmt=".", label = f"CH1{i}")
#         plt.errorbar(tiempo,vch2, yerr = RLCcortadosStd[j][i][1], fmt=".", label = f"CH2{i}")
#         plt.plot(ejex, Vrlc(ejex, *POPRLC[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHIRLC[j][i]:.1f}", linewidth = s, zorder = 1)         
#     plt.legend()
#     plt.show(block=True)


# resultados

# print("")
# for j in range(len(RCcortados)):
#     print(f"medición {j}")
#     for i in range(len(RCcortados[j])):
#         print(f"recorte {i}")
#         print("Parametros")
#         print(dfRC.loc[j]["Parametros"][i], "±", dfRC.loc[j]["Std"][i])
#         print("Bondad")
#         print(dfRC.loc[j]["Chi"][i], dfRC.loc[j]["Pv"][i])
#     print("")


# dfRCb = dfRC[dfRC["Chi"] <= 2]
dfRLb = dfRL[dfRL["Chi"] <= 2]


RsRC = []
CsRC = []
VsRC = []
RsRCstd = []
CsRCstd = []
VsRCstd = []

RsRL = []
LsRL= []
VsRL = []
RsRLstd = []
LsRLstd = []
VsRLstd = []

# medicionesC = np.array(dfRCb.index.get_level_values('Medición').unique())
medicionesL = np.array(dfRLb.index.get_level_values('Medición').unique()) 

# for j in medicionesC:
#     subset  = dfRCb.loc[j]
#     p = subset["Parametros"]
#     R1 = []
#     C1 = []
#     V1 = []
#     for i in subset.index:
#         ri = subset["Parametros"][i][0]
#         ci = subset["Parametros"][i][1]
#         vi = subset["Parametros"][i][2]
#         R1.append(ri)
#         C1.append(ci)
#         V1.append(vi)   
#     RsRC.append(np.mean(R1))
#     CsRC.append(np.mean(C1))
#     VsRC.append(np.mean(V1))
#     RsRCstd.append(np.std(R1))
#     CsRCstd.append(np.std(C1))
#     VsRCstd.append(np.std(V1))

for j in medicionesL:
    subset  = dfRLb.loc[j]
    p = subset["Parametros"]
    R1 = []
    L1 = []
    V1 = []
    for i in subset.index:
        ri = subset["Parametros"][i][0]
        li = subset["Parametros"][i][1]
        vi = subset["Parametros"][i][2]
        R1.append(ri)
        L1.append(li)
        V1.append(vi)   
    RsRL.append(np.mean(R1))
    LsRL.append(np.mean(L1))
    VsRL.append(np.mean(V1))
    RsRLstd.append(np.std(R1))
    LsRLstd.append(np.std(L1))
    VsRLstd.append(np.std(V1))

RC = []
RCstd = []
CR = np.mean(CsRC)
CRstd = np.std(CsRC)

RL = (np.mean((RsRL[0], RsRL[1])),np.mean((RsRL[2], RsRL[3])),np.mean((RsRL[4], RsRL[5])),np.mean((RsRL[6], RsRL[7])),np.mean((RsRL[8], RsRL[9])))
RLstd = (np.mean((RsRLstd[0], RsRLstd[1])),np.mean((RsRLstd[2], RsRLstd[3])),np.mean((RsRLstd[4], RsRLstd[5])),np.mean((RsRLstd[6], RsRLstd[7])),np.mean((RsRLstd[8], RsRLstd[9])))
CL = np.mean(LsRL)
CLstd = np.mean(LsRLstd)

for i in range(len(RsRC)-1):
    RC.append(np.mean((RsRC[i],RsRC[i+1])))
    RCstd.append(np.mean(((RsRCstd[i],RsRCstd[i+1]))))

chicuadrado = dfRLb["Chi"].values
pvalor = dfRLb["Pv"].values

print(RL)
print(RLstd)
print(CL)
print(CLstd)
print(np.mean(chicuadrado), np.std(chicuadrado))
print(np.mean(pvalor), np.std(pvalor))

exit()
plt.figure()
plt.title(r"$\chi^2 / \nu $")
plt.hist(chicuadrado, bins = int(np.sqrt(len(chicuadrado))))
plt.figure()
plt.title("P-valor")
plt.hist(pvalor, bins = int(np.sqrt(len(pvalor))))
plt.show(block=True)


exit()
# plots

# for j in range(len(RCcortados)):
#     plt.figure()
#     plt.title("Medición: RC" + str(rcn[j]))
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Voltaje [V]")
#     # plt.plot(DatosRC[j]["Tiempo"],DatosRC[j]["VoltajeCH1"], label = "CH1")
#     # plt.plot(DatosRC[j]["Tiempo"],DatosRC[j]["VoltajeCH2"], label = "CH2")
#     for i in range(len(RCcortados[j])):
#         df = RCcortados[j][i]
#         tiempo = df["Tiempo"]
#         vch1 = df["VoltajeCH1"] + 0.5
#         vch2 = df["VoltajeCH2"] + 0.5
#         ejex= np.linspace(np.min(tiempo), np.max(tiempo), 1000)
#         s = 2
#         plt.errorbar(tiempo,vch1, yerr = RCcortadosStd[j][i][0], fmt=".", zorder = 2)
#         plt.errorbar(tiempo,vch2, yerr = RCcortadosStd[j][i][1], fmt= ".", zorder = 2)
#         parametros_formateados = ", ".join([f"{param}" for param in POPRC[j][i]])
#         plt.plot(ejex, Vrc(ejex, *POPRC[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHIRC[j][i]:.1f}", linewidth = s, zorder = 1) #f"Aj:{str(i)} {parametros_formateados}"
#         plt.ylim(0,1.2)
#     plt.legend()
#     plt.show(block=True)


# for j in range(len(RLcortados)):
#     plt.figure()
#     plt.title("Medición: RL" + str(rln[j]))
#     plt.xlabel("Tiempo [s]")
#     plt.ylabel("Voltaje [V]")
#     plt.plot(DatosRL[j]["Tiempo"],DatosRL[j]["VoltajeCH1"], label = "CH1")
#     plt.plot(DatosRL[j]["Tiempo"],DatosRL[j]["VoltajeCH2"], label = "CH2")
#     for i in range(len(RLcortados[j])):
#         df = RLcortados[j][i]
#         tiempo = df["Tiempo"]
#         vch1 = df["VoltajeCH1"]
#         vch2 = df["VoltajeCH2"]
#         ejex= np.linspace(np.min(tiempo), np.max(tiempo), 1000)
#         s = 2
#         plt.errorbar(tiempo,vch1,  yerr = RLcortadosStd[j][i][0], fmt=".", zorder = 2)
#         plt.errorbar(tiempo,vch2,  yerr = RLcortadosStd[j][i][1], fmt=".", zorder = 2)
#         plt.plot(ejex, Vrl(ejex, *POPRL[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHIRL[j][i]:.1f}", linewidth = s, zorder = 1)
#     plt.legend()
#     plt.show(block=True)

for j in range(len(RLCcortados)):
    plt.figure()
    plt.title("Medición: RLC" + str(rlcn[j]))
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Voltaje [V]")
    # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH1"], label = "CH1")
    # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH2"], label = "CH2")
    for i in range(len(RLCcortados[j])):
        df = RLCcortados[j][i]
        tiempo = df["Tiempo"]
        vch1 = df["VoltajeCH1"]
        vch2 = df["VoltajeCH2"]
        ejex= np.linspace(np.min(tiempo), np.max(tiempo), 1000)
        s = 2        
        plt.errorbar(tiempo,vch1, yerr = RLCcortadosStd[j][i][0], fmt=".", label = f"CH1{i}")
        plt.errorbar(tiempo,vch2, yerr = RLCcortadosStd[j][i][1], fmt=".", label = f"CH2{i}")
        plt.plot(ejex, Vrlc(ejex, *POPRLC[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHIRLC[j][i]:.1f}", linewidth = s, zorder = 1)         
    plt.legend()
    plt.show(block=True)