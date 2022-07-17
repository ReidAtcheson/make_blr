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
from jax.lax import fori_loop
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
#on diagonal.
#Each block has the form:
#(is_diagonal,params)
#where is_diagonal indicates a diagonal block
def make_blr(A,blocksize,d=1):
    key=random.PRNGKey(0)
    A=sp.lil_matrix(A)
    m,_=A.shape
    blockVs=[]
    blockUs=[]
    Ds=[]
    for i in range(0,m,blocksize):
        ki=min(i+blocksize,m)-i
        keys=random.split(key,2)
        D.append(random.normal(keys[0],(ki,ki)))
        key=keys[-1]
        for j in range(0,m,blocksize):
            kj=min(j+blocksize,m)-j
            keys=random.split(key,3)
            Vs.append(random.normal(keys[0],(d,kj)))
            Us.append(random.normal(keys[1],(ki,d)))
            key=keys[-1]


        for j,_ in enumerate(range(0,m,blocksize)):
            if i==j:
                ids=list(range(i,min(i+blocksize,m)))
                blocks[i].append((True,jnp.array(np.linalg.inv(A[np.ix_(ids,ids)].toarray()))))
            else:
                ki=min(i+blocksize,m)-i
                kj=min(j+blocksize,m)-j
                keys=random.split(key,3)
                blocks[i].append((False,(random.normal(keys[0],(ki,d)),random.normal(keys[1],(d,kj)))))
                key=keys[-1]
    return blocks


@partial(jit, static_argnums=[1,2])
def eval_blr(blocks,m,blocksize,x):
    nblocks=len(blocks)
    ids=list(range(blocksize,m,blocksize))
    xs=jnp.vsplit(x,ids)

    def row(brow):

        def while_cond(





    return jnp.concatenate([row(brow) for brow in blocks],axis=0)




@partial(jit, static_argnums=[1,2])
def loss(params,m,blocksize,Axs,xs):
    blrxs=eval_blr(params,m,blocksize,Axs)
    return jnp.sum([jnp.sum((blrx-x)*(blrx-x)) for blrx,x in zip(blrxs,xs)])




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
    ids=list(range(blocksize,m,blocksize))
    xs=jnp.vsplit(x,ids)
    Axs=jnp.vsplit(Ax,ids)

    for i in range(0,100):
        start=time.time()
        g = grad(loss)(blr,m,blocksize,Axs,xs)
        updates,opt_state = opt.update(g,opt_state)
        blr = optax.apply_updates(blr,updates)
        err=loss(blr,m,blocksize,Axs,xs)
        stop=time.time()

        print(f"it = {it},  elapsed = {stop-start : .4f}, loss = {err}")
        if i==0:
            losses.append(err)


f=open("blr.dat","wb")
pickle.dump(blr,f)

