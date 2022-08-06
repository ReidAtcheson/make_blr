import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def arnoldi_mgs(A,v,k):
    norm=np.linalg.norm
    dot=np.dot
    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        for i in range(0,j+1):
            H[i,j]=dot(w,V[:,i])
            w=w-H[i,j]*V[:,i]
        H[j+1,j]=norm(w)
        V[:,j+1]=w/H[j+1,j]
    return V,H

#From "Templates for the solution of linear algebraic eigenvalue problems" pg. 167 (ch7 algorithm 7.6)
def arnoldi_dgks(A,v,k):
    norm=np.linalg.norm
    dot=np.dot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        if norm(f) < eta*norm(h):
            s = V[:,0:j+1].T @ f
            f = f - V[:,0:j+1] @ s
            h = h + s
        beta=norm(f)
        H[j+1,j]=beta
        V[:,j+1]=f/beta
    return V,H

def arnoldi_dgks_nocond(A,v,k):
    norm=np.linalg.norm
    dot=np.dot
    eta=1.0/np.sqrt(2.0)

    m=len(v)
    V=np.zeros((m,k+1))
    H=np.zeros((k+1,k))
    V[:,0]=v/norm(v)
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        H[j+1,j]=beta
        V[:,j+1]=f/beta
    return V,H






def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(0.1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))




seed=23498732
rng=np.random.default_rng(seed)
m=4096
k=100
diag=3.0
A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
Ac = lambda x : A@x
v = rng.uniform(-1,1,size=m)
#V,H,f = arnoldi_mgs(Ac,v,k)
V,H = arnoldi_dgks(Ac,v,k)
print(np.linalg.norm(V.T @ V - np.eye(k+1)))
V,H = arnoldi_dgks_nocond(Ac,v,k)
print(np.linalg.norm(V.T @ V - np.eye(k+1)))
V,H = arnoldi_mgs(Ac,v,k)
print(np.linalg.norm(V.T @ V - np.eye(k+1)))




