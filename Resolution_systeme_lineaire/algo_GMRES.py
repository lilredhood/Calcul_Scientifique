# Imports
import numpy as np
import scipy as sp
#%%
# Algorithme d'Arnoldi
def Arnoldi(A,V,H) :

  y=A@V[:,-1]     # On stocke Av_{p-1}
  n,p = np.shape(V)
  h = np.zeros(p+1)
  w = np.copy(y)

  Vp = np.zeros(shape=(n,p+1)) # On initialise la nouvelle matrice contenant la
  Hp = np.zeros(shape=(p+1,p)) # Nouvelle base de Krylov, on va de p à p+1

  # Ensuite, on remplit les matrices
  Vp[:,:-1]=V[:,:]
  Hp[:-1,:-1]=H[:,:]


  for i in range(p) :
    h[i] = y@V[:,i]   # Produit scalaire <y,v_i>= h_{j,p-1}
    w-=h[i]*V[:,i]  # On calcule w_p
    Hp[i,-1] = h[i]

  h[p] = np.linalg.norm(w) # h_{p,p-1}= ||w||
  vp = w/np.linalg.norm(w)
  Vp[:,-1] = vp 
  Hp[p,-1] = h[p]

  return Vp, Hp 
#%%
def gmres(A, b, xexact) : #A de dimension n

  # Initialisation.
  n = np.shape(A)[0]
  x0 = np.zeros(shape=n) #on initialise la suite (x_k)_k à 0

  eps = 1.e-10
  itmax = 1000
  it = 0

  r0 = b-A@x0
  v0 = r0/np.linalg.norm(r0)

  x = np.zeros(shape=n) # Notre suite (x_k)_k
  x = x0
  r = r0 # Notre suite (r_k)_k
  V = np.zeros(shape=(n,1)) # Début de la construction de la base de K_p
  V[:,0] = v0 # V une matrice colonne
  H = np.zeros(shape=(1,0)) # H (H_{-1}) une matrice vide (on l'initialise comme 
                          # une matrice à une ligne et 0 colonne)

  err_relatif = [np.linalg.norm(xexact-x)/np.linalg.norm(xexact)] #liste err relatif
  residu = [np.linalg.norm(r)/np.linalg.norm(r0)] #liste residu


  while(np.linalg.norm(r)>eps and it<itmax) :
    V_old = V # On stocke V avant de passer à la dim supérieure par Arnoldi
    V,H = Arnoldi(A,V,H)
    Q,R = sp.linalg.qr(H[:-1,:]) # Décomposition QR de la sous matrice de H(H chapeau)
    y = np.linalg.solve(R,np.linalg.norm(r0)*(Q.T)[:,0])
    x = x0+V_old@y
    r = b-A@x
    it+=1

    err_relatif.append(np.linalg.norm(xexact-x)/np.linalg.norm(xexact))
    residu.append(np.linalg.norm(r)/np.linalg.norm(r0))

  return x, err_relatif, residu