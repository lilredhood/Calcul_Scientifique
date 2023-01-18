# Imports 
import numpy as np
import scipy.sparse as spsp
from factorisation_LU import Facto_LU
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt
#%%
N = 7
alpha = 45
# Construction de la matrice A
A1 = dok_matrix((N,N), dtype = float)
for i in range(N) :
  A1[0,i]=1
  A1[i,0]=1
  A1[i,i]=1
A1[0,0] = alpha

A = spsp.csr_matrix(A1) # A sous format CSR

#Visualisation de la matrice via la commande spy
fig, ax = plt.subplots(1, 2, figsize = (12, 8))
ax[0].spy(A, color = "r") #on retrouve clairement la structure de la matrice

#après la Facto LU
Facto_LU(A)
ax[1].spy(A, color = "g")

ax[0].set_title('Visualisation de la matrice creuse A')
ax[1].set_title('Visualisation de la factorisation LU résultante')

plt.show() # La facto LU n'a pas conservé le facteur creux de la matrice A
#%%
N = 7
beta = 3.14

# Construction de la matrice B
B1 = dok_matrix((N,N), dtype=float)
for i in range(N) :
  B1[N-1,i] = 1
  B1[i,N-1] = 1
  B1[i,i] = 1
B1[N-1,N-1] = beta

B = spsp.csr_matrix(B1) # B sous format CSR

# Visualisation de la matrice via la commande spy
fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].spy(B, color="r") # de même, on retrouve aussi la structure de la matrice

# après la Facto LU
Facto_LU(B)
ax[1].spy(B,color="g")

ax[0].set_title('Visualisation de la matrice creuse B')
ax[1].set_title('Visualisation de la factorisation LU résultante')

plt.show() # Cette fois-ci la facto LU préserve le côté creux de la matrice