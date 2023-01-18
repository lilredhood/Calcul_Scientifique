# Imports
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from algo_GMRES import gmres
#%%
n = 1000
A = np.diag(2*np.ones(n)) + 0.5*np.random.rand(n,n)/np.sqrt(n)
b = np.random.randint(0,100,size=n)
xexact = np.linalg.solve(A, b)

xgmres, err_relatif, residu = gmres(A, b, xexact)

print(f"Nombre d'itérations : {len(err_relatif)}")

print(f"Test : {np.linalg.norm(xexact-xgmres)<1e-10}") # OK

# Graphe représentant l'erreur et le résidu en fonction des itérations
fig, ax = plt.subplots()
steps = range(len(err_relatif))
plt.loglog() #échelle logarithmique 
ax.plot(steps, err_relatif, label="erreur relative")
ax.plot(steps, residu, label="residu")

ax.set_xlabel('Step')
ax.set_ylabel('Erreur/Résidu')
ax.set_title("Représentation de l'erreur et le résidu en fonction des itérations")

plt.legend()
plt.show()

# L'erreur relative est très proche du résidu relatif à chaque itération. 