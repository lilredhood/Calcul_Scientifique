# Imports
import numpy as np
#%%
# Algorithme du gradient conjugué
def gradient_conjugue(A, b, xexact) :

  # Initialisation
  n = np.shape(A)[0]
  x0 = np.zeros(shape=n) # On initialise la suite x_k à 0

  eps = 1.e-10
  itmax = 1000

  x = np.zeros(shape=n)
  x = x0 #la suite (x_k)_k
  r0 = b-A@x0
  d = r0
  r = r0 #la suite (r_k)_k
  it = 0

  err_relatif = [np.linalg.norm(xexact-x)/np.linalg.norm(xexact)]
  residu = [np.linalg.norm(r)/np.linalg.norm(r0)]

  while(np.linalg.norm(r)>eps and it<itmax):
    r_old = r 
    s = (r@r)/(np.dot(A,d)@d)
    x = x+s*d
    r = r-s*(A@d)
    b = (r@r)/(r_old@r_old)
    d = r+b*d
    it += 1

    err_relatif.append(np.linalg.norm(xexact-x)/np.linalg.norm(xexact))
    residu.append(np.linalg.norm(r)/np.linalg.norm(r0))
    
  return x, err_relatif, residu
