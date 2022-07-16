import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pickle


from jax.experimental import sparse
from jax import random
import jax.numpy as jnp
import jax.nn as jnn
from jax import grad, jit, vmap
from functools import partial
import optax

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
    key=random.PRNGKey(0)
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
                keys=random.split(key,3)
                blocks[i][j]=(random.normal(keys[0],(ki,d)),random.normal(keys[1],(d,kj)))
                key=keys[-1]
    return blocks

@partial(jit, static_argnums=[1,2])
def eval_blr(blocks,m,blocksize,x):
    ids=list(range(blocksize,m,blocksize))
    xs=jnp.vsplit(x,ids)
    out=[]
    for i in range(0,m,blocksize):
        def body(j):
            if i==j:
                D=blocks[i][j]
                return D@xs[j//blocksize]
            else:
                U,Vt=blocks[i][j]
                return U@(Vt@xs[j//blocksize])
        col = [body(j) for j in range(0,m,blocksize)]
        sumop=[jnp.eye(u.shape[0]) for u in col]
        out.append(jnp.hstack(sumop)@jnp.vstack(col))
    return jnp.vstack(out)

@partial(jit, static_argnums=[1,2])
def loss(params,m,blocksize,Ax,x):
    blrx=eval_blr(params,m,blocksize,Ax)
    return jnp.sum( (blrx-x)*(blrx-x) )/m



seed=23498732
rng=np.random.default_rng(seed)
m=512
diag=4
blocksize=32
batchsize=8
nepochs=100
lr=1e-3
opt = optax.adam(lr)




#Plain preconditioned richardson
A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
#Ab=make_block_precon(A,blocksize)
#luAb=spla.splu(sp.csc_matrix(Ab))
#b=np.ones((m,3))
#Ab=A@b
blr=make_blr(A,blocksize)

r=range(0,m,blocksize)

losses=[]
opt_state = opt.init(blr)


for it in range(nepochs):
    print("NEW SUBITERATIONS")
    x=rng.normal(size=(m,batchsize))
    Ax=A@x
    for i in range(0,100):
        start=time.time()
        g = grad(loss)(blr,m,blocksize,Ax,x)
        updates,opt_state = opt.update(g,opt_state)
        blr = optax.apply_updates(blr,updates)
        err=loss(blr,m,blocksize,Ax,x)
        stop=time.time()

        print(f"it = {it},  elapsed = {stop-start : .4f}, loss = {err}")
        losses.append(err)


f=open("blr.dat","wb")
pickle.dump(blr,f)

