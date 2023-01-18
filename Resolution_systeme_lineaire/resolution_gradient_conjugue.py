# Imports
import numpy as np
import scipy.sparse as spsp
import matplotlib.pyplot as plt
from algo_gradient_conjugue import gradient_conjugue
#%%
d = 10
n = d*d

B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
b = np.random.randint(0,100,size=n)

xexact = np.linalg.solve(B.toarray(), b)
grad, err_relatif, residu = gradient_conjugue(B.toarray(), b, xexact)

print(f"Nombre d'itérations pour gradient_conjugue : {len(err_relatif)}")

print(f"Test pour gradient_conjugue : {np.linalg.norm(xexact-grad)<1e-10}") # OK

# Graphe représentant l'erreur et le résidu en fonction des itérations
fig, ax = plt.subplots()
steps = range(len(err_relatif))
plt.loglog()
ax.plot(steps, err_relatif, label="erreur relative")
ax.plot(steps, residu, label="residu")
plt.legend()

ax.set_xlabel('Step')
ax.set_ylabel('Erreur/Résidu')
ax.set_title("Représentation de l'erreur et le résidu en fonction des itérations (Gradient Conjugué)")

plt.legend()
plt.show()

# On retrouve le même graphe ainsi que le même nombre d"itérations nécéssaire pour GMRES
