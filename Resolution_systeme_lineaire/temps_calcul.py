# Imports 
import numpy as np
import scipy.sparse as spsp
import time
from factorisation_LU import Facto_LU
#%%
d=3
n=d*d
# Création d'une matrice via la commande spsp.diags()
A = spsp.diags([-np.ones(n),- np.ones(n-1), 4*np.ones(n), -np.ones(n-1),-np.ones(n)], [-d,-1, 0, 1,d], shape=(n,n)) #version creuse
A = A.tocsr()
# print(A.toarray(),"\n") 
#%%
A = spsp.diags([-np.ones(n),- np.ones(n-1), 4*np.ones(n), -np.ones(n-1),-np.ones(n)], [-d,-1, 0, 1,d], shape=(n,n)) #version creuse
A = A.tocsr()

# Calcul du temps de calcul de la fonction LU pour A creux
deb1 = time.time()
Facto_LU(A)
fin1 = time.time()
print(f"(LU) temps de calcul pour la matrice creuse = {fin1-deb1}")
#%%
# Calcul du temps de calcul de la fonction LU pour A dense
# On écrit A sous forme dense
A_dense=-np.ones(shape=(n,n))
np.fill_diagonal(A_dense, 4)
A_dense[2,0]=7.45
# print(A_dense) A dense, c'est à dire ayant un grand nombre de coefficients non nuls. 
# De plus, elle admet une facto LU.

deb2 = time.time()
Facto_LU(A_dense)
fin2 = time.time()
print(f"(LU) temps de calcul pour la matrice creuse = {fin2-deb2}")

# Comparaison:
eps = 1e-10
print(f"Ordre de grandeur (temps_creux/temps_dense) : {(fin1-deb1)/(fin2-deb2 + eps)}")