import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common_settings import*

resistenica_1 = 1172 # ±10
resistencia_2 = 1171
resistencia_4 = 2300
resistenica_aux = 1183

def errorMulti(L, x, n, Dms):
    return x*modulo(L) + n*Dms

x = 0.005
n = 1
v = np.array([375,310,242,181,122,65,9,3.9,2.6,0.1,-1.2,-44,-95])
v = v/1000 #[v]

def r4(r1,r2,r3):
    return r3*r2/r1
def rau(r1,r2,drau):
    return (r1 + drau)/ (r2 )

# r1/r2 = 1
dv1= np.array([-5.2,-4.6,-4.1,-3.5,-3,-2.4,-1.9,-1.3,-0.8,-0.3,0,0.6,1.1,1.6,2.2,2.7,3.3,3.8,4.3,4.9,5.4]) #mV
dv1std = np.array([errorMulti(v1,x,n,0.0001) for v1 in dv1]) 
dv2 = np.array([-4.6,-4.2,-3.7,-3.2,-2.7,-2.2,-1.7,-1.2,-0.7,-0.3,0,0.4,1,1.4,1.9,2.4,2.9,3.4,3.9,4.3,4.8])
dv2std = np.array([errorMulti(v1,x,n,0.0001) for v1 in dv2]) 
# dv3 = (-4.8,-4.3,-3.9,-3.4,-2.9,-2.4,-1.9,-1.4,-0.9,-0.5,0,0.1,0.6,1.1,1.5,2,2.5,3.0,3.5,3.9,4.4) originales (1/2)
dv3 = np.array([-4.8,-4.3,-3.9,-3.4,-2.9,-2.4,-1.9,-1.4,-0.9,-0.5,0,0.3,0.8,1.3,1.7,2.2,2.7,3.2,3.7,4.1,4.6]) # referencia -0.3
dv3std = np.array([errorMulti(v1,x,n,0.0001) for v1 in dv3]) 
dr = np.array([-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])

dv1n = dv1/1000
dv1nstd = dv1std/1000  
dv2n = dv2/1000
dv2nstd = dv2std/1000
dv3n = dv3/1000
dv3nstd = dv3std/1000
drn = dr/2300

def S(drn,s):
    return s*drn

p1, c1 = curve_fit(S, drn, dv1n, p0=1.1, sigma=dv1nstd, absolute_sigma=True)
p2, c2 = curve_fit(S, drn, dv2n, p0=1, sigma=dv2nstd, absolute_sigma=True)
p3, c3 = curve_fit(S, drn, dv3n, p0=1, sigma=dv3nstd, absolute_sigma=True)

print(p1[0], "±", np.sqrt(np.diag(c1))[0])
print(p2[0], "±", np.sqrt(np.diag(c2))[0])
print(p3[0], "±", np.sqrt(np.diag(c3))[0])

print(chi2_pvalor(dv1n,dv1nstd,S(drn,p1[0]),("s")))

ejex = np.linspace(np.min(drn),np.max(drn),1000)
plt.figure()
plt.title("Sensibilidad")
plt.xlabel(r"$\frac{\Delta R_4}{R_4}$")
plt.ylabel(r"$\frac{\Delta V_{ab}}{V_{cd}}$")
plt.errorbar(drn,dv1n,yerr=dv1nstd, fmt=".r", label = "R1/R2 = 1")
plt.errorbar(drn,dv2n,yerr=dv2nstd,fmt=".g", label = "R1/R2 = 2")
plt.errorbar(drn,dv3n,yerr=dv3nstd,fmt=".b", label = "R1/R2 = 1/2")
plt.plot(ejex, S(ejex,p1[0]),"r", label = f"Ajuste1: S = {p1[0]:.1f}")
plt.plot(ejex, S(ejex,p2[0]),"r", label = f"Ajuste2: S = {p2[0]:.1f}")
plt.plot(ejex, S(ejex,p3[0]),"r", label = f"Ajuste3: S = {p3[0]:.1f}")
plt.legend()
plt.show(block=True)