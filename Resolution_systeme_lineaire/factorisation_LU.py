# Imports 
import numpy as np
#%%
#Factorisation LU "inplace" d'une matrice A
def Facto_LU(A):
  for k in range(np.shape(A)[0]-1): 
    if(A[k,k]!=0):
      for j in range(k+1,np.shape(A)[0]):
        A[j,k]=A[j,k]/A[k,k]
        A[j,k+1:]-=A[j,k]*A[k,k+1:]
    else:
      print("Erreur: pivot nul")
#%%
if __name__ == "__main__" :
  # Test sur une matrice admettant une factorisation LU
  H = np.array([[2, 1, -1], [0., 4, 3], [0, 12, 7]])
  print(H,"  \n")
  Facto_LU(H)
  print(H) # pour s'assurer que la fonction modifie bien H

  # Vérification:
  # On récupere tout d'abord L et U
  L = np.tril(H,k=-1)+np.eye(np.shape(H)[0])
  U = np.triu(H)
  H_old = np.array([[2, 1, -1], [0., 4, 3], [0, 12, 7]])
  print(np.linalg.norm(L@U-H_old)<1.e-10) # OK
#%%
