# codigo de pruebas generales
from common_settings import*


Rt = (500,900,1200,1700,2200,2700,3200,3700,4200,4700)
Rts = (4,7,10,10,20,20,30,30,30,40)
R = (540,970,1200,1800,2140,2800,3300,3750,4200,4650)
Rs = (20,60,200,80,50,100,100,40,60,70)

n = (1,2,3,4,5,6,7,8,9,10)

plt.figure()
plt.errorbar(n, Rt,yerr=Rts,fmt=".-", color="b", label = "Teórico")
plt.errorbar(n, R, yerr=Rs, fmt=".-", color ="r", label = "Medición")
plt.xlabel("Número de mediciones")
plt.ylabel("Resistencia [Ω]")
plt.legend()
plt.show(block=True)