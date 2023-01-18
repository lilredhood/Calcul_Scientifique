# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from algo_GMRES import gmres
#%%
"""
Nous allons appliquer $GMRES$ à la matrice $C$ ci-dessous. 
On va comparer les résultats entre un système préconditionné $M^{-1}Cx=M^{-1}b$ 
et le système non préconditionné. Ici, un préconditionnement diagonal.
"""
#%%
n = 10
print(f"Résultat avec n = {n}")
C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
M = np.diag(np.diag(C)) #matrice ayant la diagonale de C
b = np.random.randint(0,100,size=n)
xexact = np.linalg.solve(C, b)

deb1 = time.time()
gmres(C, b, xexact)
fin1 = time.time()

print(f"Temps de calcul systeme non préconditionné : {fin1-deb1}")


# Systeme préconditionné M^{-1}*Cx=M^{-1}b{{}}
deb2 = time.time()
gmres(np.linalg.inv(M)@C, np.linalg.inv(M)@b, xexact)
fin2 = time.time()

print(f"Temps de calcul systeme préconditionné : {fin2-deb2}")

eps = 1e-10
print(f"Ordre de grandeur de temps_non_precond/temps_precond : {(fin1-deb1)/(fin2-deb2 + eps)}")

# Conditionnement des matrices
print("Conditionnement de C= ",np.linalg.cond(C))
print("Conditionnement de M^{-1}*C= ",np.linalg.cond(np.linalg.inv(M)@C))
# On a bien cond(M^{-1}*C)<=cond(C)
"""
Le conditionnement de $M^{-1}C$ est bien plus petit que celui de $C$ (heureusement) 
et reste plus ou moins constant losrsqu'on augmente $n$ mais celui de $C$ augmente strictement. 
Pour petite dimension, le temps de calcul entre le système préconditionné et non préconditionné 
est presque identique. Nous allons tracer ci-dessous le graphe représentant le temps de calculs 
des deux systèmes en fonction de la dimension.
"""
#%%
time1 = [] # Nos deux listes de temps
time2 = []

for n in range(2,100) : # On va de la dimension 2 à 100
  C = np.diag(2+np.arange(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
  M = np.diag(np.diag(C))
  b = np.array([1 for k in range(n)])
  xexact = np.linalg.solve(C, b)

  # Le temps de calcul pour GMRES sur le système Cx=b
  deb=time.time()
  gmres(C, b, xexact)
  fin=time.time()
  time1.append(fin-deb)

  #Le temps de calcul pour GMRES sur le système M^{-1}Cx=bM^{-1}
  deb = time.time()
  gmres(np.linalg.inv(M)@C, np.linalg.inv(M)@b, xexact)
  fin = time.time()
  time2.append(fin-deb)


fig, ax = plt.subplots()
dim = range(2,100)
ax.plot(dim, time1, label="Sans préconditionnement")
ax.plot(dim, time2, label="Avec préconditionnement")

ax.set_xlabel('Dimension')
ax.set_ylabel('Temps (s)')
ax.set_title("Représentation du temps d'éxécution de GMRES en fonction de la dimnsion de la matrice pour les deux sytèmes (Matrice C)")

plt.legend()
plt.show()

"""
Plus la dimension est grande, plus le système préconditionné est efficace, 
le temps de calcul pour ce celui-ci est même presque constant à partir d'une certaine dimension.
"""
#%%
# On applique cette fois-ci $GMRES$ sur la matrice $D$ ci-dessous.
n = 25
print(f"Résultat avec n={n}")

D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
M = np.diag(np.diag(D))
b = np.random.randint(1,100,size=n)
xexact = np.linalg.solve(D, b)

deb1 = time.time()
xgmres, err_relatif, residu = gmres(D, b, xexact)
fin1 = time.time()
print(f"Temps de calcul systeme non préconditionné : {fin1-deb1}")

# Systeme préconditionné M^{-1}*Cx=M^{-1}b
deb2 = time.time()
xgmres, err_relatif, residu = gmres(np.linalg.inv(M)@D, np.linalg.inv(M)@b, xexact)
fin2 = time.time()
print(f"Temps de calcul systeme préconditionné : {fin2-deb2}")

eps = 1e-10
print(f"Ordre de grandeur de temps_non_precond/temps_precond : {(fin1-deb1)/(fin2-deb2 + eps)}")

# Conditionnement des matrices
print("Conditionnement de D= ",np.linalg.cond(D))
print("Conditionnement de M^{-1}*D= ",np.linalg.cond(np.linalg.inv(M)@D))
# On a bien cond(M^{-1}*D)<=cond(D), ici égalité

"""
Le conditionnement de $M^{-1}D$ est égal à celui de $D$ quelque soit la dimension, 
ce qui est normal par le calcul puisque $M$ est l'identité à une constante près.
"""
#%%
# Nous allons tracer ci-dessous le graphe représentant le temps de calculs des deux systèmes en fonction de la dimension.
time1 = [] #nos deux listes de temps
time2 = []

for n in range(2, 80) : #on va de la dimension 2 à 80
  D = np.diag(2*np.ones(n)) - np.diag(np.ones(n-1),1) - np.diag(np.ones(n-1),-1)
  M = np.diag(np.diag(D))
  b = np.array([1 for k in range(n)])
  xexact = np.linalg.solve(D, b)

  # Le temps de calcul pour GMRES sur le système Cx=b
  deb = time.time()
  gmres(D, b, xexact)
  fin = time.time()
  time1.append(fin-deb)

  # Le temps de calcul pour GMRES sur le système M^{-1}Cx=bM^{-1}
  deb = time.time()
  gmres(np.linalg.inv(M)@D, np.linalg.inv(M)@b, xexact)
  fin = time.time()
  time2.append(fin-deb)

fig, ax = plt.subplots()
dim = range(2, 80)
ax.plot(dim, time1, label="Sans préconditionnement")
ax.plot(dim, time2, label="Avec préconditionnement")

ax.set_xlabel('Dimension')
ax.set_ylabel('Temps (s)')
ax.set_title("Représentation du temps d'éxécution de GMRES en fonction de la dimnsion de la matrice pour les deux sytèmes (Matrice D)")

plt.legend()
plt.show()

"""
On remarque que le temps de calcul des deux systèmes est presque identique quelque
soit la dimension. C'est pas étonnant vu le calcul du conditionnement précédémment,
on a donc un mauvais préconditionnement.
"""