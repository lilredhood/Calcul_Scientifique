# Imports 
import numpy as np
import matplotlib.pylab as plt
import scipy as sp
import scipy.sparse as spsp
import time
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import splu
#%%
#Création de la matrice A sous format CSR:
row = np.array([0, 0, 1, 2, 2, 2]) # Ligne
col = np.array([0, 2, 2, 0, 1, 2]) # Colonne
data = np.array([1, 2, 3, 4, 5, 6]) # Données
A = spsp.csr_matrix((data, (row, col)), shape=(3, 3))

#Les différents attributs du format CSR:
print(A.data) 
print(A.indptr) 
print(A.indices) 
print(f"La matrice A : \n {A.toarray()}")
#%%
"""
On considère une matrice $A$ définit au-dessus et on la convertit au format $CSR$. 
La commande $A.data$ nous renvoit les données de la matrice, c'est à dire les coefficients non nuls de celle-ci. 
Ensuite, $A.indptr$ nous renvoit les pointeurs vers les débuts de lignes dans les tableaux data et indices. 
Pour finir, $A.indices$ correspond aux indices des colonnes des coefficients non nulles de la matrice. 
A partir de ces trois attributs nous pouvons retrouver la matrice $A$.

De plus, lorsqu'on essaye d'afficher la matrice $A$ via la commande $print()$ nous obtenons des tuples correspondant 
aux indices des lignes et colonnes des coeffecients non nulles suivi de celui-ci. 
Pour afficher la matrice sous forme de tableau de tableaux comme on a l'habitude de faire, 
on peut utiliser $print(A.toarray())$.
"""
#%%#Les commandes A[0,:], et A[:,0]
lst_time = []
for _ in range(500) :
    b = time.time()
    __ = A[0,:]
    e = time.time()
    lst_time.append(e-b)
print(np.mean(lst_time))

lst_time = []
for _ in range(500) :
    b = time.time()
    __ = A[:, 0]
    e = time.time()
    lst_time.append(e-b)
print(np.mean(lst_time))
# %%
"""
La commande $A[0,:]$ nous donne un tuple correspondant aux
indices de lignes et colonnes des coefficients non nulles 
de la premiere ligne suivi du coefficient. D'autre part, $A[:,0]$ nous donne un tuple correspondant aux
indices de lignes et colonnes des coefficients non nulles 
de la premiere colonne suivi du coefficient.


$A[0,:]$ est plus rapide que $A[:,0]$ puisque cette commande parcourt une seule ligne alors
que $A[:,0]$ parourt les $n$ lignes de la matrice en prenant en compte que
le premier coefficient de ces lignes.
"""
#%%
# Somme de deux matrices de format CSR
# On définit une nouvelle matrice B
row2 = np.array([0, 0, 1, 2, 2, 2])
col2 = np.array([0, 2, 2, 0, 1, 2])
data2 = np.array([1, 7, 5, 8, 2, 0])
B = spsp.csr_matrix((data2, (row2, col2)), shape=(3, 3))
    
print(f"La matrice B = \n {B.toarray()}")

print((A+B).toarray(),"\n")
print(A+B) # OK 
#%%
"""
On considère une deuxième matrice $B$ définit ci-dessus aussi convertit sous format $CSR$. 
La somme $A+B$ nous donne bien le résultat attendu. 
On peut donc ajouter deux matrices stockées sous format $CSR$. 
"""
#%%
# Multiplication d'une matrice par un vecteur, ici A par b 
# La fonction multiplication:
def matvect_multiply(A, b):
  y = np.zeros(np.size(b)) 
  for i in range(np.size(b)): 
    for j in range(A.indptr[i], A.indptr[i + 1]): 
      c = A.indices[j]
      y[i] = y[i] + A.data[j] * b[c]
  return y
#%%
#On initialise un vecteur aléatoire b
b = np.random.randint(0, 10, 3)

#Vérification avec ce vecteur : 
y = matvect_multiply(A, b)
y_ex = A.dot(b)
print(np.linalg.norm(y-y_ex)<1.e-10) # OK

# Et avec A@b?
y_ex2=A@b
print(np.linalg.norm(y-y_ex2)<1.e-10) # OK

# Et np.dot(A,b) ?
y_ex3=np.dot(A,b)
#print(np.linalg.norm(y-y_ex3)<1.e-10) nous affiche une erreur
#Lorsqu'on affiche y_ex3 nous obtenons:
print(y_ex3)
#%%
"""
Notre fonction de multiplication entre $A$ et $b$ parcourt les éléments non nuls de $A$ à l'aide des attributs du format $CSR$. 
On s'assure qu'elle donne le bon résultat en comparant avec les multiplications déjà définit comme $A$@$b$ et $A.dot(b)$. 
Cependant, la commande $np.dot(A,b)$ nous renvoit pas de vecteur/tenseur...
"""