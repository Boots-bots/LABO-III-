# codigo de pruebas generales
from common_settings import*

xcs = (0,153,633,973)
ycs = (5,20,70,100)
xcb = (0,8700)
ycb = (5,300)

xrs = (0,101,830)
yrs = (5,10,50)
xrb = (0,335)
yrb = (5,30)

ejex=np.linspace(0,1000,1000)

plt.figure()
plt.plot(xcs,ycs,"r",label="Comun subida")
# plt.plot(xcb,ycb,"g",label="Comun bajada")
plt.plot(xrs,yrs,"b",label="Rapido subida")
# plt.plot(xrb,yrb,"k",label="Rapido bajada")
plt.plot(ejex,lineal(ejex,20/153,5),"r",label="Comun")
for v in (64,633,973):
    plt.axvline(x=v, color='r', linestyle='--', label = f"Comun Val = {v}")
for v in (60,130,583,830):
    plt.axvline(x=v, color="b", linestyle='--', label = f"Rapido Val = {v}")
plt.title("Pruebas")
plt.xlabel("val")
plt.ylabel("std")
plt.legend()
plt.show(block=True)

