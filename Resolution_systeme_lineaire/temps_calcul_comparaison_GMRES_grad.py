# Imports
import numpy as np
import scipy.sparse as spsp
import time
from algo_gradient_conjugue import gradient_conjugue
from algo_GMRES import gmres
#%%
d = 10
n = d*d

B = spsp.diags([[4.]*n,[-1]*(n-1),[-1] *(n-1),[-1] *(n-d),[-1] *(n-d)],[0,1,-1,d,-d])
b = np.random.randint(0,100,size=n)
#%%
deb1 = time.time()
gradient_conjugue(B.toarray(), b, np.linalg.solve(B.toarray(),b))
fin1 = time.time()

print(f"Temps pour gradient conjugue : {fin1-deb1}")

deb2 = time.time()
gmres(B.toarray(), b, np.linalg.solve(B.toarray(),b))
fin2 = time.time()

print(f"Temps pour GMRES : {fin2-deb2}")

eps = 1e-10
print(f"Ordre de grandeur de tempsGMRES/tempsGRAD : {(fin2-deb2)/(fin1-deb1 + eps)}")
#%%
"""
On remarque que la fonction $GMRES$ prend plus de temps que la fonction $gradientconjugue$. 
De plus, si l'on augmente la dimension de la matrice, on trouve un écart entre les temps de calculs plus grand. 
Ces algorithmes sont bien adaptés aux structures creuses puisque la suite $(r_k)_{k≥1}$ tendra plus rapidement 
vers $0$ du au fait que la matrice a beaucoup de zéros. Ainsi la norme de $b-x_kA$ sera plus rapidement minimisée. 
Pour les matrices creuses, la complexité de ces algorithmes se ramenent à un $O(n)$ ou n étant la dimension de la matrice.
"""