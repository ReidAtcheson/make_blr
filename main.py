import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from jax.experimental import sparse
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap

@jit
def update(params,updates,step):
    return [(p0+step*u0,p1+step*u1) for (p0,p1),(u0,u1) in zip(params,updates)]

#Make a random sparse-banded matrix 
#with bands in `bands1
#its diagonal shifted by `diag`
def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(-1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
    offs = [-x for x in bands] + [0] + [x for x in bands]
    return sp.diags(subdiags,offs,shape=(m,m))

#Carve out block diagonal matrix from
#input matrix for making a 
#preconditioner - for testing
def make_block_precon(A,blocksize):
    A=sp.lil_matrix(A)
    m,_=A.shape
    blocks=[]
    for i in range(0,m,blocksize):
        beg=i
        end=min(i+blocksize,m)
        ids=list(range(beg,end))
        blocks.append(A[np.ix_(ids,ids)])
    return sp.block_diag(blocks)

#Create a preconditioner by blocking up
#range and putting a rank-`d`-update
#in each block and full matrices
#on diagonal 
def make_blr(A,blocksize,d=1):
    A=sp.lil_matrix(A)
    m,_=A.shape
    blocks={}
    for i in range(0,m,blocksize):
        blocks[i]={}
        for j in range(0,m,blocksize):
            if i==j:
                ids=list(range(i,min(i+blocksize,m)))
                blocks[i][j]=jnp.array(np.linalg.inv(A[np.ix_(ids,ids)].toarray()))
            else:
                ki=min(i+blocksize,m)-i
                kj=min(j+blocksize,m)-j
                blocks[i][j]=(jnp.zeros((ki,d)),jnp.zeros((d,kj)))
    return blocks


seed=23498732
rng=np.random.default_rng(seed)
m=512
diag=4
blocksize=32


#Plain preconditioned richardson
A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
#Ab=make_block_precon(A,blocksize)
#luAb=spla.splu(sp.csc_matrix(Ab))
b=np.ones(m)

blr=make_blr(A,blocksize)
