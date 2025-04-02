# codigo de pruebas generales
from common_settings import*

N = 1000
a = np.random.rand(N)
b = []
def f(n,N):
    N = N*np.ones(len(n))
    return n*N*0.75*0.05
for n in a:
    if n>=0.75:
        b.append(n)      
print(len(b))
plt.figure()
plt.hist(b,rwidth=0.05)
plt.plot(b,f(b,N),"r")
plt.xlim(-0.5,2)
plt.show(block=True)