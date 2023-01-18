# Imports 
import numpy as np
#%%
#Nous voulons ensuite programmer une fonction solve_LU(A,b)
#Pour cela, nous aurons besoin des algos de descente et de remonté

def descente(L,b):
  n=len(b)
  y=np.zeros(n)
  y[0]=b[0]
  for i in range(1,n):
    y[i]=b[i]
    for j in range(L.indptr[i], L.indptr[i + 1]-1): 
      c = L.indices[j]
      y[i] = y[i] - L.data[j] * y[c]
  return y

def remonte(U,y):
  n=len(y)
  x=np.zeros(n)
  x[n-1]=y[n-1]/U[n-1,n-1]
  for i in range(n-2,-1,-1):
    x[i]=y[i]
    for j in range(U.indptr[i]+1, U.indptr[i + 1]):
      c = U.indices[j]
      x[i]= x[i] - U.data[j] * x[c] 
    x[i]/=U[i,i]
  return x