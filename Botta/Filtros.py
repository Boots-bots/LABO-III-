import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

# CLASE 5

R = 217    # Ω  # 220
C = 1e-6   # F
L = 0.001 # H

# 1V

def std(df, f):
    subset = df.loc[f]
    resolucion_ch1 = subset["ResolucionVCH1"].values
    resolucion_ch2 = subset["ResolucionVCH2"].values
    std_ch1 = resolucion_ch1 / 2
    std_ch2 = resolucion_ch2 / 2
    return std_ch1, std_ch2

def gen(t,V,f,φ):
    return V*np.sin(2*np.pi*f*t + φ)


print("elegir circuito:")
print("PA, PB, RLC")
circuito = input()

if circuito in ("PA", "pa", "Pa", "pA"):
# Pasa Altos RC
    ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/PasaAltosRC/"

    # df_PA2 = pd.read_csv(ruta + "PARC2.csv")
    # df_PA2.set_index(["Frecuencias", "Tiempo"])
    # df_PA3 = pd.read_csv(ruta + "PARC3.csv")
    # df_PA3.set_index(["Frecuencias", "Tiempo"])

    # df_PA = pd.concat([df_PA1, df_PA2, df_PA3])
    # df_PA.set_index(["Frecuencias", "Tiempo"])
    # df_PA.to_csv(ruta+"PA.csv", index=False, header=True)

    df_PA = pd.read_csv(ruta + "PA.csv", index_col=["Frecuencias"])

    frecuencias = np.unique(df_PA.index) 

    Chis1 = []
    pvals1 = []
    Chis2 = []
    pvals2 = []

    dif_fase = []
    dif_fase_std = []

    dif_amp = []
    dif_amp_std = []

    for frec in frecuencias:
        tiempo = df_PA.loc[frec]["Tiempo"].values
        vch1 = df_PA.loc[frec]["VoltajeCH1"].values
        vch2 = df_PA.loc[frec]["VoltajeCH2"].values

        amplitud_ch1 = (np.max(vch1)-np.min(vch1))
        amplitud_ch2 = (np.max(vch2)-np.min(vch2))
        amplitud_ch1 = np.array(amplitud_ch1)
        amplitud_ch2 = np.array(amplitud_ch2)

        stdIn, stdOut = std(df_PA, frec)

        #in  (CH1)
        popIn, pcovIn = curve_fit(gen, tiempo, vch1, sigma = stdIn, p0 = (amplitud_ch1/2, frec, 0), absolute_sigma=True)
        popstdIn = np.sqrt(np.diag(pcovIn))
        #out (CH2)
        popOut, pcovOut = curve_fit(gen, tiempo, vch2, sigma = stdOut, p0 = (amplitud_ch2/2, frec, 0), absolute_sigma=True)
        popstdOut = np.sqrt(np.diag(pcovOut))

        #chequeo
        # plt.figure()
        # plt.title(f"frecuencia: {frec:.1f} Hz")
        # plt.xlabel("tiempo [s]")
        # plt.ylabel("V")
        # plt.errorbar(tiempo,vch1,stdIn, fmt=".", color="b", label = "CH1 Vin")
        # plt.errorbar(tiempo,vch2,stdOut, fmt=".", color="g", label = "CH2 Vout")
        # plt.plot(tiempo, gen(tiempo,*popIn), "r")
        # plt.plot(tiempo, gen(tiempo,*popOut), "r")
        # plt.legend()
        # plt.show(block=True)

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
        dif_amp.append(popOut[0]/popIn[0])
        dif_amp_std.append(np.sqrt(((popstdOut[0]*chi2)/popIn[0])**2 + (popOut[0]*(popstdIn[0]*chi1)/popIn[0]**2)**2))

    dif_fase = np.array(dif_fase)
    dif_fase_std = np.array(dif_fase_std)
    dif_amp = np.array(dif_amp)
    dif_amp_std = np.array(dif_amp_std)


    def fase(w,w0):
        return ang(np.arctan(w0/w))

    def Transferencia(w,w0):
        return 1/np.sqrt(1+(w0/w)**2)

    def Atenuación(T):
        return 20*np.log10(T) 
    def Atenuación2(w,w0):
        return -10*np.log10(1+(w0/w)**2) 

    Att = np.array([Atenuación(T) for T in dif_amp])
    Attstd = [np.sqrt(((20*dif_amp_std[i])/(dif_amp[i]*np.log(10)))**2) for i in range(len(dif_amp))]

    w = frecuencias*2*np.pi
    w0 = 1/(R*C)

    # corrección
    Att = np.delete(Att,(0,1))
    dif_fase = np.delete(dif_fase,(0,1))
    dif_amp = np.delete(dif_amp,(0,1))
    Attstd = np.delete(Attstd,(0,1))
    dif_fase_std = np.delete(dif_fase_std,(0,1))
    dif_amp_std = np.delete(dif_amp_std,(0,1))
    w = np.delete(w,(0,1))

    angdif_fase = ang(dif_fase)
    angdif_fase_std = ang(dif_fase_std)
    ejex=np.linspace(np.min(w), np.max(w), 10000)

    # Ajustes
    pop_Fase, cov_Fase = curve_fit(fase, w, angdif_fase, p0 = (w0), sigma = angdif_fase_std, absolute_sigma = True)
    pop_Trans, cov_Trans = curve_fit(Transferencia, w, dif_amp, p0 = (w0), sigma = dif_amp_std, absolute_sigma = True)
    pop_Att, cov_Att = curve_fit(Atenuación2, w, Att, p0 = (w0), sigma = Attstd, absolute_sigma = True)

    print()
    print("Teórico")
    print("w0 =", w0)
    print("Ajustes")
    print("Fase")
    print("w0 = ", pop_Fase[0], "±", np.sqrt(cov_Fase[0])[0])
    print("Transferencia")
    print("w0 = ", pop_Trans[0], "±", np.sqrt(cov_Trans[0])[0])
    print("Atenuación")
    print("w0 = ", pop_Att[0], "±", np.sqrt(cov_Att[0])[0])

    chiF,pvalF,glF = chi2_pvalor(angdif_fase,angdif_fase_std,fase(w,pop_Fase[0]),("w0"))
    chistdF = np.sqrt(2/glF)
    chiT,pvalT,glT = chi2_pvalor(dif_amp,dif_amp_std,Transferencia(w,pop_Trans[0]),("w0"))
    chistdT = np.sqrt(2/glT)
    chiA,pvalA,glA = chi2_pvalor(Att,Attstd,Atenuación2(w,pop_Att[0]),("w0"))
    chistdA = np.sqrt(2/glA)

    print()
    print("Bondad")
    print("Fase")
    print(f"$\chi^2$ = {chiF/glF:.1f} ± {chistdF:.1f}")
    print(f"p-valor = {pvalF:.1f}")
    print("Transferencia")
    print(f"$\chi^2$ = {chiT/glT:.1f} ± {chistdT:.1f}")
    print(f"p-valor = {pvalT:.1f}")
    print("Atenuación")
    print(f"$\chi^2$ = {chiA/glA:.1f} ± {chistdA:.1f}")
    print(f"p-valor = {pvalA:.1f}")

    # plt.text(0.8, 0.2, f"$\chi^2$ = {chi/nu:.1f} ± {s:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")
    # plt.text(0.8, 0.1, f"p-valor = {pv:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")

    #lin
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_fase_std), fmt='.', label = "Mediciones")
    plt.axhline(y=45, color='g', linestyle='--', label = "45°") 
    plt.axvline(x=pop_Fase[0], color='g', linestyle='--', label = f"ω0 = {pop_Fase[0]:.1f}") 
    plt.plot(ejex, fase(ejex, pop_Fase[0]), "r", label ="Ajuste")
    plt.title("Desfasaje en Filtro Pasaaltos RC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [Hz]")
    plt.legend()

    plt.figure()
    plt.errorbar(w, dif_amp, dif_amp_std, fmt='.', label = "Mediciones")
    plt.axhline(y=0.7, color='g', linestyle='--', label = "70%") 
    plt.axvline(x=pop_Trans[0], color='g', linestyle='--', label = f"ω0 = {pop_Trans[0]:.1f}")
    plt.plot(ejex, Transferencia(ejex, pop_Trans[0]), "r", label ="Ajuste")
    plt.title("Transferencia en Filtro Pasaaltos RC")
    plt.ylabel("Transferencia")
    plt.xlabel("ω [Hz]")
    plt.legend()

    #Log 
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_amp_std), fmt='.', label = "Mediciones")
    plt.axhline(y=45, color='g', linestyle='--', label = "45°") 
    plt.axvline(x=pop_Fase[0], color='g', linestyle='--', label = f"ω0 = {pop_Fase[0]:.1f}")
    plt.plot(ejex, fase(ejex, pop_Fase[0]), "r", label ="Modelo")
    plt.title("Desfasaje en Filtro Pasaaltos RC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [Hz]")
    plt.xscale('log')
    plt.legend()


    plt.figure()
    plt.axline((w[0],Att[0]), (w[4], Att[4]), linestyle = "-", color="k")
    plt.errorbar(w, Att, Attstd, fmt='.', label = "Mediciones")
    plt.axhline(y=-3.01, color='g', linestyle='--', label = "3.01 dB") 
    plt.axvline(x=pop_Att[0], color='g', linestyle='--', label = f"ω0 = {pop_Att[0]:.1f}")
    plt.plot(ejex, Atenuación2(ejex, pop_Att[0]), "r", label ="Ajuste")
    plt.axhline(y=0, color="k", linestyle = "-")
    plt.title("Atenuación en Filtro Pasaaltos RC")
    plt.ylabel("Atenuación [dB]")
    plt.xlabel("ω [Hz]")
    plt.xscale('log')
    plt.legend()

    plt.show(block = True)


    exit()
    #Bondad

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

################################################################

if circuito in ("PB", "pb", "Pb", "pB", ""):
# Pasa Bajos RC
    ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/PasaBajosRC/"

    # df_PB1 = pd.read_csv(ruta + "PBRC1.csv")
    # df_PB1.set_index(["Frecuencias", "Tiempo"])
    # df_PB2 = pd.read_csv(ruta + "PBRC2.csv")
    # df_PB2.set_index(["Frecuencias", "Tiempo"])
    # df_PB3 = pd.read_csv(ruta + "PBRC3.csv")
    # df_PB3.set_index(["Frecuencias", "Tiempo"])

    # df_PB = pd.concat([df_PB1, df_PB2, df_PB3])
    # df_PB.set_index(["Frecuencias", "Tiempo"])
    # df_PB.to_csv(ruta+"PB.csv", index=False, header=True)

    # print(df_PB)

    df_PB = pd.read_csv(ruta + "PB.csv", index_col=["Frecuencias"])

    frecuencias = np.unique(df_PB.index) 
    
    Chis1 = []
    pvals1 = []
    Chis2 = []
    pvals2 = []

    dif_fase = []
    dif_fase_std = []

    dif_amp = []
    dif_amp_std = []

    for frec in frecuencias:
        tiempo = df_PB.loc[frec]["Tiempo"].values
        vch1 = df_PB.loc[frec]["VoltajeCH1"].values
        vch2 = df_PB.loc[frec]["VoltajeCH2"].values

        amplitud_ch1 = (np.max(vch1)-np.min(vch1))
        amplitud_ch2 = (np.max(vch2)-np.min(vch2))
        amplitud_ch1 = np.array(amplitud_ch1)
        amplitud_ch2 = np.array(amplitud_ch2)

        stdIn, stdOut = std(df_PB, frec)

        #in  (CH1)
        popIn, pcovIn = curve_fit(gen, tiempo, vch1, sigma = stdIn, p0 = (amplitud_ch1/2, frec, 0), absolute_sigma=True)
        popstdIn = np.sqrt(np.diag(pcovIn))
        #out (CH2)
        popOut, pcovOut = curve_fit(gen, tiempo, vch2, sigma = stdOut, p0 = (amplitud_ch2/2, frec, 0), absolute_sigma=True)
        popstdOut = np.sqrt(np.diag(pcovOut))

        #chequeo
        # plt.figure()
        # plt.title(f"frecuencia: {frec:.1f} Hz")
        # plt.xlabel("tiempo [s]")
        # plt.ylabel("V")
        # plt.errorbar(tiempo,vch1,stdIn, fmt=".", color="b", label = "CH1 Vin")
        # plt.errorbar(tiempo,vch2,stdOut, fmt=".", color="g", label = "CH2 Vout")
        # plt.plot(tiempo, gen(tiempo,*popIn), "r")
        # plt.plot(tiempo, gen(tiempo,*popOut), "r")
        # plt.legend()
        # plt.show(block=True)

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
        dif_amp.append(popOut[0]/popIn[0])
        dif_amp_std.append(np.sqrt(((popstdOut[0]*chi2)/popIn[0])**2 + (popOut[0]*(popstdIn[0]*chi1)/popIn[0]**2)**2))

    dif_fase = np.array(dif_fase)
    dif_fase_std = np.array(dif_fase_std)
    dif_amp = np.array(dif_amp)
    dif_amp_std = np.array(dif_amp_std)

    def fase(w,w0):
        return ang(-np.arctan(w/w0))

    def Transferencia(w,w0):
        return 1/np.sqrt(1+(w/w0)**2)

    def Atenuación(T):
        return 20*np.log10(T) 
    def Atenuación2(w,w0):
        return -10*np.log10(1+(w/w0)**2) 
    
    Att = np.array([Atenuación(T) for T in dif_amp])
    Attstd = [np.sqrt(((20*dif_amp_std[i])/(dif_amp[i]*np.log(10)))**2) for i in range(len(dif_amp))]

    w = frecuencias*2*np.pi
    w0 = 1/(R*C)

    angdif_fase = ang(dif_fase)
    angdif_fase_std = ang(dif_fase_std)
    ejex=np.linspace(np.min(w), np.max(w), 10000)

    # Ajustes
    pop_Fase, cov_Fase = curve_fit(fase, w, angdif_fase, p0 = (w0), sigma = angdif_fase_std, absolute_sigma = True)
    pop_Trans, cov_Trans = curve_fit(Transferencia, w, dif_amp, p0 = (w0), sigma = dif_amp_std, absolute_sigma = True)
    pop_Att, cov_Att = curve_fit(Atenuación2, w, Att, p0 = (w0), sigma = Attstd, absolute_sigma = True)

    print()
    print("Teórico")
    print("w0 =", w0)
    print("Ajustes")
    print("Fase")
    print("w0 = ", pop_Fase[0], "±", np.sqrt(cov_Fase[0])[0])
    print("Transferencia")
    print("w0 = ", pop_Trans[0], "±", np.sqrt(cov_Trans[0])[0])
    print("Atenuación")
    print("w0 = ", pop_Att[0], "±", np.sqrt(cov_Att[0])[0])

    chiF,pvalF,glF = chi2_pvalor(angdif_fase,angdif_fase_std,fase(w,pop_Fase[0]),("w0"))
    chistdF = np.sqrt(2/glF)
    chiT,pvalT,glT = chi2_pvalor(dif_amp,dif_amp_std,Transferencia(w,pop_Trans[0]),("w0"))
    chistdT = np.sqrt(2/glT)
    chiA,pvalA,glA = chi2_pvalor(Att,Attstd,Atenuación2(w,pop_Att[0]),("w0"))
    chistdA = np.sqrt(2/glA)

    print()
    print("Bondad")
    print("Fase")
    print(f"$\chi^2$ = {chiF/glF:.1f} ± {chistdF:.1f}")
    print(f"p-valor = {pvalF:.1f}")
    print("Transferencia")
    print(f"$\chi^2$ = {chiT/glT:.1f} ± {chistdT:.1f}")
    print(f"p-valor = {pvalT:.1f}")
    print("Atenuación")
    print(f"$\chi^2$ = {chiA/glA:.1f} ± {chistdA:.1f}")
    print(f"p-valor = {pvalA:.1f}")

    # plt.text(0.8, 0.2, f"$\chi^2$ = {chi/nu:.1f} ± {s:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")
    # plt.text(0.8, 0.1, f"p-valor = {pv:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")

    #lin
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_fase_std), fmt='.', label = "Mediciones")
    plt.axhline(y=-45, color='g', linestyle='--', label = "- 45°") 
    plt.axvline(x=pop_Fase[0], color='g', linestyle='--', label = f"ω0 = {pop_Fase[0]:.1f}") 
    plt.plot(ejex, fase(ejex, pop_Fase[0]), "r", label ="Ajuste")
    plt.title("Desfasaje en Filtro Pasabajos RC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [rad/s]")
    plt.legend()

    plt.figure()
    plt.errorbar(w, dif_amp, dif_amp_std, fmt='.', label = "Mediciones")
    plt.axhline(y=0.7, color='g', linestyle='--', label = "70%") 
    plt.axvline(x=pop_Trans[0], color='g', linestyle='--', label = f"ω0 = {pop_Trans[0]:.1f}")
    plt.plot(ejex, Transferencia(ejex, pop_Trans[0]), "r", label ="Ajuste")
    plt.title("Transferencia en Filtro Pasabajos RC")
    plt.ylabel("Transferencia")
    plt.xlabel("ω [rad/s]")
    plt.legend()

    #Log 
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_amp_std), fmt='.', label = "Mediciones")
    plt.axhline(y=-45, color='g', linestyle='--', label = "- 45°") 
    plt.axvline(x=pop_Fase[0], color='g', linestyle='--', label = f"ω0 = {pop_Fase[0]:.1f}")
    plt.plot(ejex, fase(ejex, pop_Fase[0]), "r", label ="Modelo")
    plt.title("Desfasaje en Filtro Pasabajos RC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [rad/s]")
    plt.xscale('log')
    plt.legend()


    plt.figure()
    plt.axline((w[len(w)-8],Att[len(Att)-8]), (w[len(w)-1], Att[len(Att)-1]), linestyle = "-", color="k")
    plt.errorbar(w, Att, Attstd, fmt='.', label = "Mediciones")
    plt.axhline(y=-3.01, color='g', linestyle='--', label = "3.01 dB") 
    plt.axvline(x=pop_Att[0], color='g', linestyle='--', label = f"ω0 = {pop_Att[0]:.1f}")
    plt.plot(ejex, Atenuación2(ejex, pop_Att[0]), "r", label ="Ajuste")
    plt.axhline(y=0, color="k", linestyle = "-")
    plt.title("Atenuación en Filtro Pasabajos RC")
    plt.ylabel("Atenuación [dB]")
    plt.xlabel("ω [rad/s]")
    plt.xscale('log')
    plt.legend()

    plt.show(block = True)

    exit()
    #Bondad

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

################################################################

if circuito in ("RLC", "rlc", "Rlc", "rLc", "rlC", "RLc", "rLC", "RlC"):
# Pasa Banda RLC
    ruta = "C:/Users/faust/Documents/UBA/Actividades/Laboratorio/3/Datos/PasaBandaRLC/"

    # df_PBA1 = pd.read_csv(ruta + "RLC1.csv")
    # df_PBA1.set_index(["Frecuencias", "Tiempo"])
    # df_PBA2 = pd.read_csv(ruta + "RLC2.csv")
    # df_PBA2.set_index(["Frecuencias", "Tiempo"])
    # df_PBA3 = pd.read_csv(ruta + "RLC3.csv")
    # df_PBA3.set_index(["Frecuencias", "Tiempo"])
    # df_PBA4 = pd.read_csv(ruta + "RLC4.csv")
    # df_PBA4.set_index(["Frecuencias", "Tiempo"])

    # df_PBA = pd.concat([df_PBA1, df_PBA2, df_PBA3, df_PBA4])
    # df_PBA.set_index(["Frecuencias", "Tiempo"])
    # df_PBA.to_csv(ruta+"PBA.csv", index=False, header=True)

    df_PBA = pd.read_csv(ruta + "PBA.csv", index_col=["Frecuencias"])

    frecuencias = np.unique(df_PBA.index) 

    Chis1 = []
    pvals1 = []
    Chis2 = []
    pvals2 = []

    dif_fase = []
    dif_fase_std = []

    dif_amp = []
    dif_amp_std = []

    for frec in frecuencias:
        tiempo = df_PBA.loc[frec]["Tiempo"].values
        vch1 = df_PBA.loc[frec]["VoltajeCH1"].values
        vch2 = df_PBA.loc[frec]["VoltajeCH2"].values

        amplitud_ch1 = (np.max(vch1)-np.min(vch1))
        amplitud_ch2 = (np.max(vch2)-np.min(vch2))
        amplitud_ch1 = np.array(amplitud_ch1)
        amplitud_ch2 = np.array(amplitud_ch2)

        stdIn, stdOut = std(df_PBA, frec)

        #in  (CH1)
        popIn, pcovIn = curve_fit(gen, tiempo, vch1, sigma = stdIn, p0 = (amplitud_ch1/2, frec, 0), absolute_sigma=True)
        popstdIn = np.sqrt(np.diag(pcovIn))
        #out (CH2)
        popOut, pcovOut = curve_fit(gen, tiempo, vch2, sigma = stdOut, p0 = (amplitud_ch2/2, frec, 0), absolute_sigma=True)
        popstdOut = np.sqrt(np.diag(pcovOut))

        #chequeo
        # plt.figure()
        # plt.title(f"frecuencia: {frec:.1f} Hz")
        # plt.xlabel("tiempo [s]")
        # plt.ylabel("V")
        # plt.errorbar(tiempo,vch1,stdIn, fmt=".", color="b", label = "CH1 Vin")
        # plt.errorbar(tiempo,vch2,stdOut, fmt=".", color="g", label = "CH2 Vout")
        # plt.plot(tiempo, gen(tiempo,*popIn), "r")
        # plt.plot(tiempo, gen(tiempo,*popOut), "r")
        # plt.legend()
        # plt.show(block=True)

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
        dif_amp.append(popOut[0]/popIn[0])
        dif_amp_std.append(np.sqrt(((popstdOut[0]*chi2)/popIn[0])**2 + (popOut[0]*(popstdIn[0]*chi1)/popIn[0]**2)**2))

    dif_fase = np.array(dif_fase)
    dif_fase_std = np.array(dif_fase_std)
    dif_amp = np.array(dif_amp)
    dif_amp_std = np.array(dif_amp_std)

    def fase(w,r,l,c):
        return ang(np.arctan((w*r*c)/(w*l - 1/(w*c))))

    def Transferencia(w,r,l,c):
        return (w*r*c)/((r**2) + (w*l - 1/(w*c))**2)

    def Atenuación(T):
        return 20*np.log10(np.abs(T)) 
    def Atenuación2(w,r,l,c):                               #este fue el mas cercano (faltó escala)
        return 20*np.log10(np.abs(Transferencia(w,r,l,c))) 

    Att = np.array([Atenuación(T) for T in dif_amp])
    Attstd = [np.sqrt(((20*dif_amp_std[i])/(dif_amp[i]*np.log(10)))**2) for i in range(len(dif_amp))]

    def W0(L,C):
        return 1/np.sqrt(L*C)

    w = frecuencias*2*np.pi
    w0 = W0(L,C)
    dw = R/L

    angdif_fase = ang(dif_fase)
    angdif_fase_std = ang(dif_fase_std)
    ejex = np.linspace(np.min(w), np.max(w), 10000)

    # # Ajustes
    # pop_Fase, cov_Fase = curve_fit(fase, w, angdif_fase, p0 = (R,L,C), sigma = angdif_fase_std, absolute_sigma = True)
    # pop_Trans, cov_Trans = curve_fit(Transferencia, w, dif_amp, p0 = (R,L,C), sigma = dif_amp_std, absolute_sigma = True)
    # pop_Att, cov_Att = curve_fit(Atenuación2, w, Att, p0 = (R,L,C), sigma = Attstd, absolute_sigma = True)
    # stdF, stdT, stdA =  (np.sqrt(cov_Fase[1]),np.sqrt(cov_Fase[2])) , (np.sqrt(cov_Trans[1]), np.sqrt(cov_Trans[2])) , (np.sqrt(cov_Att[1]), np.sqrt(cov_Att[2]))
     
    print()
    print("Teórico")
    print("R =", R)
    print("L =", L)
    print("C =", C)   
    print("w0 =", w0)
    # print("Ajustes")
    # print("Fase")
    # print("R = ", pop_Fase[0], "±", np.sqrt(cov_Fase[0])[0])
    # print("L = ", pop_Fase[1], "±", np.sqrt(cov_Fase[1])[0])
    # print("C = ", pop_Fase[2], "±", np.sqrt(cov_Fase[2])[0])
    # print("w0 = ", 1/np.sqrt(pop_Fase[1]*pop_Fase[2]), "±", "ja")
    # print("Transferencia")
    # print("R = ", pop_Trans[0], "±", np.sqrt(cov_Trans[0])[0])
    # print("L = ", pop_Trans[1], "±", np.sqrt(cov_Trans[1])[0])
    # print("C = ", pop_Trans[2], "±", np.sqrt(cov_Trans[2])[0])
    # print("w0 = ", 1/np.sqrt(pop_Trans[1]*pop_Trans[2]), "±", "ja")
    # print("Atenuación")
    # print("R = ", pop_Att[0], "±", np.sqrt(cov_Att[0])[0])
    # print("L = ", pop_Att[1], "±", np.sqrt(cov_Att[1])[0])
    # print("C = ", pop_Att[2], "±", np.sqrt(cov_Att[2])[0])
    # print("w0 = ", 1/np.sqrt(pop_Att[1]*pop_Att[2]), "±", "ja")

    # chiF,pvalF,glF = chi2_pvalor(angdif_fase,angdif_fase_std,fase(w,pop_Fase[0],pop_Fase[1],pop_Fase[2]),("r","l","c"))
    # chistdF = np.sqrt(2/glF)
    # chiT,pvalT,glT = chi2_pvalor(dif_amp,dif_amp_std,Transferencia(w,pop_Trans[0],pop_Trans[1],pop_Trans[2]),("r","l","c"))
    # chistdT = np.sqrt(2/glT)
    # chiA,pvalA,glA = chi2_pvalor(Att,Attstd,Atenuación2(w,pop_Att[0], pop_Att[1], pop_Att[2]),("r","l","c"))
    # chistdA = np.sqrt(2/glA)

    # print()
    # print("Bondad")
    # print("Fase")
    # print(f"$\chi^2$ = {chiF/glF:.1f} ± {chistdF:.1f}")
    # print(f"p-valor = {pvalF:.1f}")
    # print("Transferencia")
    # print(f"$\chi^2$ = {chiT/glT:.1f} ± {chistdT:.1f}")
    # print(f"p-valor = {pvalT:.1f}")
    # print("Atenuación")
    # print(f"$\chi^2$ = {chiA/glA:.1f} ± {chistdA:.1f}")
    # print(f"p-valor = {pvalA:.1f}")

    # plt.text(0.8, 0.2, f"$\chi^2$ = {chi/nu:.1f} ± {s:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")
    # plt.text(0.8, 0.1, f"p-valor = {pv:.1f}", 
    #          transform=plt.gca().transAxes, fontsize=10, color="black")

    #lin
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_fase_std), fmt='.', label = "Mediciones")
    plt.axhline(y=0, color='g', linestyle='--', label = "0°") 
    plt.axvline(x=w0, color='g', linestyle='--', label = f"ω0 = {w0:.1f}") 
    # plt.plot(ejex, fase(ejex, R,L,C), "r", label ="Modelo")
    plt.title("Desfasaje en Filtro Pasa Banda RLC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [rad/s]")
    plt.legend()

    plt.figure()
    plt.errorbar(w, dif_amp, dif_amp_std, fmt='.', label = "Mediciones")
    plt.axhline(y=0.7, color='g', linestyle='--', label = "70%") 
    plt.axvline(x=w0, color='g', linestyle='--', label = f"ω0 = {w0:.1f}")
    plt.axvline(x=w0+dw, color='darkgreen', linestyle='-', label = f"Δω = {dw:.1f}")
    # plt.plot(ejex, Transferencia(ejex, R,L,C), "r", label ="Modelo")
    plt.title("Transferencia en Filtro Pasa Banda RLC")
    plt.ylabel("Transferencia")
    plt.xlabel("ω [rad/s]")
    plt.legend()

    #Log 
    plt.figure()
    plt.errorbar(w, ang(dif_fase), ang(dif_amp_std), fmt='.', label = "Mediciones")
    plt.axhline(y=0, color='g', linestyle='--', label = "0°") 
    plt.axvline(x=w0, color='g', linestyle='--', label = f"ω0 = {w0:.1f}")
    # plt.plot(ejex, fase(ejex, R,L,C), "r", label ="Modelo")
    plt.title("Desfasaje en Filtro Pasa Banda RLC")
    plt.ylabel("Diferencia de fase [°]")
    plt.xlabel("ω [rad/s]")
    plt.xscale('log')
    plt.legend()


    plt.figure()
    plt.errorbar(w, Att, Attstd, fmt='.', label = "Mediciones")
    plt.axhline(y=-3.01, color='g', linestyle='--', label = "- 3.01 dB") 
    plt.axvline(x=w0, color='g', linestyle='--', label = f"ω0 = {w0:.1f}")
    plt.axvline(x=w0+dw, color='darkgreen', linestyle='--', label = f"Δω = {dw:.1f}")
    plt.axvline(x=w0-dw, color='darkgreen', linestyle='-')
    # plt.plot(ejex, Atenuación2(ejex, R,L,C), "r", label ="Modelo")
    plt.title("Atenuación en Filtro Pasa Banda RLC")
    plt.ylabel("Atenuación [dB]")
    plt.xlabel("ω [rad/s]")
    plt.xscale('log')
    plt.legend()

    plt.show(block = True)

    exit()
    #Bondad
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