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


save_folder = ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/Diodos/"

# # curva I-V de diodos 
#   # diodo comun 

# Comun = pd.read_csv(save_folder + "IVdiodocomun.csv")

# tc = Comun["Tiempo"]
# vch1c = Comun["VoltajeCH1"] # R
# vch2c = -Comun["VoltajeCH2"]  # D
# vch1cstd, vch2cstd = std(Comun)

#   # diodo rápido 

# Rapido = pd.read_csv(save_folder + "IVdiodorapido.csv")

# tr = Rapido["Tiempo"]
# vch1r = Rapido["VoltajeCH1"] # R
# vch2r = -Rapido["VoltajeCH2"]  # D
# vch1rstd, vch2rstd = std(Rapido)

#   # diodo Zener

# Zener = pd.read_csv(save_folder + "IVdiodozen4.csv")

# tz = Zener["Tiempo"]
# vch1z = Zener["VoltajeCH1"] # R
# vch2z = -Zener["VoltajeCH2"]  # D
# vch1zstd, vch2zstd = std(Zener)


# # plt.figure()
# # plt.errorbar(tc,vch1c,yerr=vch1cstd,fmt=".",label="Voltaje Resistencia")
# # plt.errorbar(tc,vch2c,yerr=vch2cstd,fmt=".",label="Voltaje Diodo")
# # plt.title("Datos Diodo Común")
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Voltaje [V]")
# # plt.legend()

# # plt.figure()
# # plt.errorbar(tr,vch1r,yerr=vch1rstd,fmt=".",label="Voltaje Resistencia")
# # plt.errorbar(tr,vch2r,yerr=vch2rstd,fmt=".",label="Voltaje Diodo")
# # plt.title("Datos Diodo Rápido")
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Voltaje [V]")
# # plt.legend()

# # plt.figure()
# # plt.errorbar(tz,vch1z,yerr=vch1zstd,fmt=".",label="Voltaje Resistencia")
# # plt.errorbar(tz,vch2z,yerr=vch2zstd,fmt=".",label="Voltaje Diodo")
# # plt.title("Datos Diodo Zener")
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Voltaje [V]")
# # plt.legend()

# Ic = (vch1c/1000)*1000 # V/R # mA
# Icstd = (vch1cstd/1000)*1000
# Ir = (vch1r/1000)*1000 # V/R # mA
# Irstd = (vch1rstd/1000)*1000
# Iz = (vch1z/1000)*1000 # V/R # mA
# Izstd = (vch1zstd/1000)*1000


# def I(V,I0,Vt):
#     return I0*(np.exp(V/Vt)-1)

# def Rd(V,I0,Vt):
#     return Vt/(I0*np.exp(V/Vt))

# pc,cc = curve_fit(I, vch2c, Ic, p0=(1e-9,0.026), sigma=Icstd, absolute_sigma=True)
# pr,cr = curve_fit(I, vch2r, Ir, p0=(1e-9,0.026), sigma=Irstd, absolute_sigma=True)
# pz,cz = curve_fit(I, vch2z, Iz, p0=(1e-9,0.026), sigma=Izstd, absolute_sigma=True)

# x2c, pvc, nc = chi2_pvalor(Ic,Icstd,I(vch1c,*pc),("V","I0","Vt"))
# print(pc , np.sqrt(np.diag(cc)))
# print(x2c/nc,pvc)
# print("")
# x2r, pvr, nr = chi2_pvalor(Ir,Irstd,I(vch1r,*pr),("V","I0","Vt"))
# print(pr , np.sqrt(np.diag(cr)))
# print(x2r/nr,pvr)
# print("")
# x2z, pvz, nz = chi2_pvalor(Iz,Izstd,I(vch1z,*pz),("V","I0","Vt"))
# print(pz , np.sqrt(np.diag(cz)))
# print(x2z/nz,pvz)
# print("")

# ejexc = np.linspace(np.min(vch2c),np.max(vch2c),1000)
# ejexr = np.linspace(np.min(vch2r),np.max(vch2r),1000)
# ejexz = np.linspace(np.min(vch2z),np.max(vch2z),1000)

# # plt.figure()
# # plt.errorbar(vch2c,Ic,yerr=Icstd,fmt=".",label = "Datos I-V")
# # plt.plot(ejexc,I(ejexc,*pc), "r")
# # plt.axvline(x=0.42, color='g', linestyle='--', label = f"Vd = 0.42 V") 
# # plt.title("Curva Característica Diodo Común")
# # plt.ylabel("Corriente [mA]")
# # plt.xlabel("Voltaje [V]")
# # plt.legend()

# # plt.figure()
# # plt.errorbar(vch2r,Ir,yerr=Irstd,fmt=".",label = "Datos I-V")
# # plt.plot(ejexr,I(ejexr,*pr), "r")
# # plt.axvline(x=0.35, color='g', linestyle='--', label = f"Vd = 0.35 V") 
# # plt.title("Curva Característica Diodo Rápido")
# # plt.ylabel("Corriente [mA]")
# # plt.xlabel("Voltaje [V]")
# # plt.legend()

# plt.figure()
# plt.errorbar(vch2z,Iz,yerr=Izstd,fmt=".",label = "Datos I-V")
# plt.plot(ejexz,I(ejexz,*pz), "r")
# plt.axvline(x=0.5, color='g', linestyle='--', label = f"Vd = 0.5 V") 
# plt.title("Curva Característica Diodo Zener")
# plt.ylabel("Corriente [mA]")
# plt.xlabel("Voltaje [V]")
# plt.legend()


# # plt.figure()
# # plt.plot(ejexc,Rd(ejexc,*pc),"r",label="Resistencia Variable")
# # plt.title("Resistencia Variable Diodo Común")
# # plt.xlabel("Voltaje [V]")
# # plt.ylabel("R[Ω]")
# # plt.ylim(0,20000)
# # # plt.yscale("log")
# # plt.legend()

# # plt.figure()
# # plt.plot(ejexr,Rd(ejexr,*pr),"r",label="Resistencia Variable")
# # plt.title("Resistencia Variable Diodo Rápido")
# # plt.xlabel("Voltaje [V]")
# # plt.ylabel("R[Ω]")
# # plt.ylim(0,20000)
# # # plt.yscale("log")
# # plt.legend()


#plt.show(block=True)


# Tiempo de respuesta

# comun

# CS = pd.read_csv(save_folder + "Transcomunsubida.csv")

# ts = CS["Tiempo"]
# vch1s =  CS["VoltajeCH1"]
# vch2s = CS["VoltajeCH2"]
# vch1sstd, vch2sstd = std(CS)

# plt.figure()
# plt.errorbar(ts,vch2s,yerr=vch2sstd,fmt=".",label="Voltaje Diodo")
# plt.errorbar(ts,vch1s,yerr=vch1sstd ,fmt=".", label="Fuente")
# plt.axvline(x=-13e-9, color='g', linestyle='--') 
# plt.axvline(x=140e-9, color='g', linestyle='--', label = f"t = 153 ns") 
# plt.title("Transitorio Activación del Diodo Compun")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Voltaje [V]")
# plt.legend()

# CB1 = pd.read_csv(save_folder + "Transcomunbajada1.csv")
# CB2 = pd.read_csv(save_folder + "Transcomunbajada2.csv")

# t1 = CB1["Tiempo"]
# vch11 =  CB1["VoltajeCH1"] 
# vch21 = CB1["VoltajeCH2"] # diodo
# vch11std, vch21std = std(CB1)
# t2 = CB2["Tiempo"]
# vch12 =  CB2["VoltajeCH1"]
# vch22 = CB2["VoltajeCH2"]
# vch12std, vch22std = std(CB2)

# # plt.figure()
# # plt.errorbar(t1,vch21,yerr=vch21std,fmt=".",label="Voltaje Diodo")
# # plt.errorbar(t1,vch11,yerr=vch11std,fmt=".",label="Fuente")
# # # plt.axvline(x=-13e-9, color='g', linestyle='--') 
# # # plt.axvline(x=140e-9, color='g', linestyle='--', label = f"t = 153 ns") 
# # plt.title("Transitorio Desactivación del Diodo Común")
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Voltaje [V]")
# # plt.legend()

# plt.figure()
# plt.errorbar(t2,vch22,yerr=vch22std,fmt=".",label="Voltaje Diodo")
# plt.errorbar(t2,vch12,yerr=vch12std,fmt=".",label="Fuente")
# plt.axvline(x=1.088e-4, color='g', linestyle='--') 
# plt.axvline(x=1.00150e-4, color='g', linestyle='--', label = f"t = 8.7 μs") 
# plt.title("Transitorio Desactivación del Diodo Común")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Voltaje [V]")
# plt.legend()

# plt.show(block=True)


# rapido

# RS = pd.read_csv(save_folder + "Transrapidosubida.csv")

# ts = RS["Tiempo"]
# vch1s =  RS["VoltajeCH1"]
# vch2s = RS["VoltajeCH2"]
# vch1sstd, vch2sstd = std(RS)

# # plt.figure()
# # plt.errorbar(ts,vch2s,yerr=vch2sstd,fmt=".",label="Voltaje Diodo")
# # plt.errorbar(ts,vch1s,yerr=vch1sstd ,fmt=".", label="Fuente")
# # plt.axvline(x=-11e-9, color='g', linestyle='--') 
# # plt.axvline(x=112e-9, color='g', linestyle='--', label = f"t = 101 ns") 
# # plt.title("Transitorio Activación del Diodo Rápido")
# # plt.xlabel("Tiempo [s]")
# # plt.ylabel("Voltaje [V]")
# # plt.legend()

# RB = pd.read_csv(save_folder + "Transrapidobajada.csv")

# tb = RB["Tiempo"]
# vch1 =  RB["VoltajeCH1"] 
# vch2 = RB["VoltajeCH2"] # diodo
# vch1std, vch2std = std(RB)


# plt.figure()
# plt.errorbar(tb,vch2,yerr=vch2std,fmt=".",label="Voltaje Diodo")
# plt.errorbar(tb,vch1,yerr=vch1std,fmt=".",label="Fuente")
# plt.axvline(x=1.0032e-4, color='g', linestyle='--') 
# plt.axvline(x=0.99985e-4, color='g', linestyle='--', label = f"t = 335 ns") 
# plt.title("Transitorio Desactivación del Diodo Rápido")
# plt.xlabel("Tiempo [s]")
# plt.ylabel("Voltaje [V]")
# plt.legend()

# plt.show(block=True)

# Resistencia Variable
# subida

# Comun
Cs270 = pd.read_csv(save_folder + "270Cs.csv")
Cs4630 = pd.read_csv(save_folder + "4630Cs.csv")
Cs8k = pd.read_csv(save_folder + "8kCs.csv")

# Rapido
Rs270 = pd.read_csv(save_folder + "270Rs.csv")
Rs1k = pd.read_csv(save_folder + "1kRs.csv")
Rs4630 = pd.read_csv(save_folder + "4630Rs.csv")
Rs8k = pd.read_csv(save_folder + "8kRs.csv")

plt.figure()
plt.errorbar(Cs270["Tiempo"],Cs270["VoltajeCH2"],yerr=std(Cs270)[1],fmt=".",label="Voltaje Diodo")
plt.errorbar(Cs270["Tiempo"],Cs270["VoltajeCH1"],yerr=std(Cs270)[0],fmt=".",label="Fuente")
plt.axvline(x=70e-9, color='g', linestyle='--') 
plt.axvline(x=-17e-9, color='g', linestyle='--', label = f"t = 64 ns") 
plt.title("Transitorio Activación del Diodo Común Rc 270 Ω")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.legend()

plt.show(block=True)