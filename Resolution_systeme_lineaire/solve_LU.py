# Imports 
import numpy as np
import scipy.sparse as spsp
from factorisation_LU import Facto_LU
from algo_descente_remonte import descente, remonte
#%%
"""
On s'inspire de la fonction $matvectmultiply()$ pour parcourir que les éléments 
non nuls de $L$ et $U$ à l'aide des attributs du format $CSR$.
"""
#Nous pouvons maintenant définir solve(A,b)
def solve_LU(A, b) :
  Facto_LU(A)
  # On convertit L et U sous format CSR pour nos algos de remonté/descente
  n = np.shape(A)[0]
  U = spsp.csr_matrix(np.triu(A),shape=(n,n))
  L = spsp.csr_matrix(np.tril(A,k=-1)+np.eye(n), shape=(n, n))

  x = np.zeros(np.shape(A)[0])
  y = np.zeros(np.shape(A)[0])
  
  y = descente(L,b) #on résout tout d'abord Ly=b
  x = remonte(U,y) #puis Ux=y
  return x
#%%
if __name__ == "__main__" :

  # Vérification sur une matrice A et d'un vecteur aléatoire b
  n = 3
  A = spsp.diags([- np.ones(n-1), 2*np.ones(n), -np.ones(n-1)], [-1, 0, 1])
  A = A.tocsr()
  b = np.random.normal(size=3)

  x_ex = np.linalg.inv(A.toarray())@b
  x = solve_LU(A.toarray(),b)
  print(np.linalg.norm(x-x_ex)<1.e-10) # OK