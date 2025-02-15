# codigo de pruebas generales
from common_settings import*
from Botta.Filtros import*

plt.figure()
plt.title(r"Bondad de ajuste $(\chi^2)$")
plt.xlabel("Chi/nu")
plt.hist(Chis2, bins=int(np.sqrt((len(Chis2)))), alpha=0.5, label=f"CH2: {np.mean(Chis2)}")
plt.hist(Chis1, bins=int(np.sqrt(len(Chis1))), alpha=0.5, label=f"CH1: {np.mean(Chis1)}")
plt.legend()

plt.figure()
plt.title("Bondad de Ajuste (P-valor)")
plt.xlabel("P-valor")
plt.hist(pvals1, bins=int(np.sqrt(len(pvals1))), alpha=0.5, label=f"CH1: {np.mean(pvals1)}")
plt.hist(pvals2, bins=int(np.sqrt(len(pvals2))), alpha=0.5, label=f"CH2: {np.mean(pvals2)}")
plt.legend()
plt.show(block=True)