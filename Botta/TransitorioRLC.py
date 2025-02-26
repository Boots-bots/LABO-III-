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
rlcn = (1,22,2,15,3,4,14,13,5,6,7,12,19,199,8,9,16,166,10,17,18,11)
DatosRLC=[]
DatosRLC = [pd.read_csv(save_folder + "/RLC/" + "RLC" + str(n) + ".csv") for n in rlcn]

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

RLCcortados = [cortadora(dRLC) for dRLC in DatosRLC]

for i in range(len(RLCcortados)):
    if i in (3,6,7,12,13,16,17):
        RLCcortados[i] = [item for idx, item in enumerate(RLCcortados[i]) if idx not in (0,)]   
    else:
        RLCcortados[i] = [item for idx, item in enumerate(RLCcortados[i]) if idx not in (0, len(RLCcortados[i])-1)]

RLCcortadosStd = []
for j in range(len(RLCcortados)):       
    rlcstd = []
    for i in range(len(RLCcortados[j])): 
        df = RLCcortados[j][i]       
        vch1std, vch2std = std(df)  
        u = (vch1std,vch2std)
        rlcstd.append(u)
    RLCcortadosStd.append(rlcstd)

#Ajustes

def Vsub(t,R,L,C,Q,V0): #R<2SQRT(L/C)
    if R >= 2*np.sqrt(L/C):
        return np.full_like(t,np.nan) 
    else:
        a = R/(2*L)
        b = np.sqrt(np.abs(a**2 - 1/(L*C)))
        w = np.sqrt(np.abs(1/(L*C) - a**2))
        return (Q/C)*np.exp(-a*t)*(np.cos(w*t) + (a/w)*np.sin(w*t)) + V0

def Vcrit(t,R,L,C,Q,V0): #R=2SQRT(L/C)
    a = R/(2*L)
    b = np.sqrt(np.abs(a**2 - 1/(L*C)))
    return (Q/C)*(1 + a*t)*np.exp(-a*t) + V0

def Vsob(t,R,L,C,Q,V0): #R>2sqrt(L/C)
    if R <= 2*np.sqrt(L/C):
        return np.full_like(t,np.nan) 
    else:
        a = R/(2*L)
        b = np.sqrt(np.abs(a**2 - 1/(L*C)))
        return (Q/C)*np.exp(-a*t)*(np.cosh(b*t) + (a/b)*np.sinh(b*t)) + V0

#pops
R = (157.5,240,250,387,500,620,640,1000,1180,1700,2003,2100,2110,2110,3020,4000,4700,4700,6000,6400,8390,10000)
C = 8.9 * 10**(-9) # F (n)
L = 0.01  # H

POP = []           
POPstd = []
CHI = []
PV = []
data = []


for j in range(len(RLCcortados)-2):  # para cada experimento #21 22 no andan tiran error nan
    Pop = []
    Popstd= []
    Chi = []
    Pv = []
    for i in range(len(RLCcortados[j])):  # para cada recorte
        df = RLCcortados[j][i]
        tiempo = df["Tiempo"]
        vch1 = df["VoltajeCH1"]       
        vch2 = df["VoltajeCH2"]
        vch2std = RLCcortadosStd[j][i][1]

        p0 = (R[j], L, C, 0, np.mean(vch1))
        bounds = ([1e-12, 1e-12, 1e-12, -np.inf, -np.inf],[100000, 1, 1, np.inf, np.inf])


        if j < 9:
            pop, cov = curve_fit(Vsub, tiempo, vch2, p0 = p0, sigma = vch2std, absolute_sigma = True, bounds = bounds) 
            popstd = np.sqrt(np.diag(cov))
            chi, pv, n = chi2_pvalor(vch2, vch2std, Vsub(tiempo, *pop), ("R", "L", "C", "Q", "V0"))
            Chi.append(chi/n)
            Pv.append(pv)
        if j in (9,10,11,12,13):
            pop, cov = curve_fit(Vcrit, tiempo, vch2, p0 = p0, sigma = vch2std, absolute_sigma = True, bounds = bounds) 
            popstd = np.sqrt(np.diag(cov))
            chi, pv, n = chi2_pvalor(vch2, vch2std, Vcrit(tiempo, *pop), ("R", "L", "C", "Q", "V0"))
            Chi.append(chi/n)
            Pv.append(pv)
        if j >= 14:
            pop, cov = curve_fit(Vsob, tiempo, vch2, p0 = p0, sigma = vch2std, absolute_sigma = True, bounds = bounds) 
            popstd = np.sqrt(np.diag(cov))
            chi, pv, n = chi2_pvalor(vch2, vch2std, Vsob(tiempo, *pop), ("R", "L", "C", "Q", "V0"))
            Chi.append(chi/n)
            Pv.append(pv)
        Pop.append(pop)          
        Popstd.append(popstd)
        print(f"Curv{j}{i}")
        data.append({
            "Medición": j,
            "Recorte": i,
            "Parametros": pop,
            "Std": popstd,
            "Chi": chi / n,
            "Pv": pv
        })    

    print(f"finalizado {j}")
    POP.append(Pop)
    POPstd.append(Popstd)
    CHI.append(Chi)
    PV.append(Pv)

dfRLC = pd.DataFrame(data)
dfRLC.set_index(["Medición", "Recorte"], inplace=True)
# dfRLC.to_csv(save_folder+"dfRLC.csv", index=False, header=True)

def e(t,R,L,c):
    a = R/(2*L)
    return np.exp(-a*t) + c

# try:
#     for j in range(len(RLCcortados)):
#         plt.figure()
#         plt.title("Medición: RLC" + str(R[j]))
#         plt.xlabel("Tiempo [s]")
#         plt.ylabel("Voltaje [V]")
#         # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH1"], label = "CH1")
#         # plt.plot(DatosRLC[j]["Tiempo"],DatosRLC[j]["VoltajeCH2"], label = "CH2")
#         for i in range(len(RLCcortados[j])):
#             df = RLCcortados[j][i]
#             tiempo = df["Tiempo"]
#             vch1 = df["VoltajeCH1"]
#             vch2 = df["VoltajeCH2"]
#             ejex= np.linspace(np.min(tiempo), np.max(tiempo), 1000)
#             s = 2        
#             plt.errorbar(tiempo,vch1, yerr = RLCcortadosStd[j][i][0], fmt=".", label = f"CH1{i}")
#             plt.errorbar(tiempo,vch2, yerr = RLCcortadosStd[j][i][1], fmt=".", label = f"CH2{i}")
#             if j <= 9:
#                 plt.plot(ejex,e(ejex,POP[j][i][0], POP[j][i][1],POP[j][i][4]), label = "Decaimiento")
#                 plt.plot(ejex, Vsub(ejex, *POP[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHI[j][i]:.1f}", linewidth = s, zorder = 1)         
#                 plt.ylim(-1,2)
#             if j in (10,11,12,13):
#                 plt.plot(ejex, Vcrit(ejex, *POP[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHI[j][i]:.1f}", linewidth = s, zorder = 1)         
#             if j >=14:
#                 plt.plot(ejex, Vsob(ejex, *POP[j][i]), "r", label= f"Aj{str(i)}: $\chi^2$ = {CHI[j][i]:.1f}", linewidth = s, zorder = 1)         
#         plt.legend()
#         plt.show(block=True)
# except:
#     pass

dfRLCb = dfRLC[dfRLC["Chi"] <= 10] # ver

print(dfRLCb)

RsRLC = []
LsRLC = [] 
CsRLC = []
QsRLC = []
VsRLC = []
RsRLCstd = []
LsRLCstd = []
CsRLCstd = []
QsRLCstd = []
VsRLCstd = []


mediciones = np.array(dfRLCb.index.get_level_values('Medición').unique()) 

for j in mediciones:
    subset  = dfRLCb.loc[j]
    p = subset["Parametros"]
    R1 = []
    L1 = []
    C1 = []
    Q1 = []
    V1 = []
    for i in subset.index:
        ri = subset["Parametros"][i][0]
        li = subset["Parametros"][i][1]
        ci = subset["Parametros"][i][2]
        qi = subset["Parametros"][i][3]
        vi = subset["Parametros"][i][4]
        R1.append(ri)
        L1.append(li)
        C1.append(ci)
        Q1.append(qi)
        V1.append(vi)   
    RsRLC.append(np.mean(R1))
    LsRLC.append(np.mean(L1))
    CsRLC.append(np.mean(C1))
    QsRLC.append(np.mean(Q1))
    VsRLC.append(np.mean(V1))
    RsRLCstd.append(np.std(R1))
    LsRLCstd.append(np.std(L1))
    CsRLCstd.append(np.std(C1))
    QsRLCstd.append(np.std(Q1))
    VsRLCstd.append(np.std(V1))

print(RsRLC)
print(RsRLCstd)
print(LsRLC)
print(LsRLCstd)
print(CsRLC)
print(CsRLCstd)
print(QsRLC)
print(QsRLCstd)
print(VsRLC)
print(VsRLCstd)

exit()