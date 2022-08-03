import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pickle


import jax
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
#with bands in `bands1/
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
def make_blr_random(A,blocksize,d=1):
    key=random.PRNGKey(0)
    A=sp.lil_matrix(A)
    m,_=A.shape
    assert( m%blocksize==0 )
    blockVs=[]
    blockUs=[]
    Ds=[]
    for i in range(0,m,blocksize):
        Us=[]
        Vs=[]
        ki=min(i+blocksize,m)-i
        keys=random.split(key,2)
        Ds.append(random.normal(keys[0],(ki,ki)))
        key=keys[-1]
        for j in range(0,m,blocksize):
            kj=min(j+blocksize,m)-j
            keys=random.split(key,3)
            Vs.append(random.normal(keys[0],(d,kj)))
            Us.append(random.normal(keys[1],(ki,d)))
            key=keys[-1]
        blockVs.append(jnp.asarray(Vs))
        blockUs.append(jnp.asarray(Us))

    return jnp.asarray(blockUs),jnp.asarray(blockVs),jnp.asarray(Ds)

#Create a preconditioner by blocking up
def make_blr(A,blocksize,d=1):
    key=random.PRNGKey(0)
    A=sp.lil_matrix(A)
    m,_=A.shape
    assert( m%blocksize==0 )
    blockVs=[]
    blockUs=[]
    Ds=[]
    for i in range(0,m,blocksize):
        Us=[]
        Vs=[]
        ki=min(i+blocksize,m)-i
        ids=list(range(i,i+ki))
        Ds.append(np.linalg.inv(A[np.ix_(ids,ids)].toarray()))
        for j in range(0,m,blocksize):
            kj=min(j+blocksize,m)-j
            Vs.append(jnp.zeros((d,kj)))
            Us.append(jnp.zeros((ki,d)))
        blockVs.append(jnp.asarray(Vs))
        blockUs.append(jnp.asarray(Us))

    return jnp.asarray(blockUs),jnp.asarray(blockVs),jnp.asarray(Ds)


@partial(jit, static_argnums=[1,2])
def eval_blr(blocks,m,blocksize,x):
    m,ncols=x.shape
    nblocks=m//blocksize
    Us,Vs,Ds=blocks
    xr = x.reshape((nblocks,blocksize,ncols))
    out=[]
    for i in range(0,nblocks):
        Vx = jax.lax.dot_general(Vs[i],xr,dimension_numbers=(
            ((2,), (1,)),
            ((0,), (0,))
            ))
        UVx = jax.lax.dot_general(Us[i],Vx,dimension_numbers=(
            ((0,), (0,)),
            ((2), (1))
            ))
        out.append(UVx)

    y=jnp.asarray(out).reshape((nblocks,blocksize,ncols))
    #Ds.shape = (16, 32, 32)
    #y.shape = (16, 32, 8)

    #print(Ds.shape)
    #print(y.shape)
    z=y+jax.lax.dot_general(Ds,xr,dimension_numbers=(
            ((2,), (1,)),
            ((0,), (0,))
            ))
    return z.reshape((m,ncols))




@partial(jit, static_argnums=[1,2])
def loss(params,m,blocksize,Ax,x):
    blrx=eval_blr(params,m,blocksize,Ax)
    sqrtm=jnp.sqrt(m)
    return jnp.sum(((blrx-Ax)/sqrtm)*((blrx-Ax)/sqrtm))




f=open("blr.dat","rb")
blr=pickle.load(f)


f=open("A.dat","rb")
A=pickle.load(f)
m,_=A.shape
blocksize=m//blr[0].shape[0]
Ab=make_block_precon(A,blocksize)
luAb=spla.splu(sp.csc_matrix(Ab))
b=np.ones(m)



errs=[]
it=0
def callback(xk):
    global it
    err=np.linalg.norm(b-A@xk)
    print(f"it = {it}, res = {err}")
    errs.append(err)

    it=it+1

Afull=A.toarray()
Iblr=eval_blr(blr,m,blocksize,Afull)
Ib=luAb.solve(Afull)
print(f"norm(Iblr-I) = {np.linalg.norm(Iblr.flatten()-np.eye(m).flatten())}")
print(f"norm(Ib-I) = {np.linalg.norm(Ib.flatten()-np.eye(m).flatten())}")


seed=23084
rng=np.random.default_rng(seed)
restart=10
maxiter=50
print("STANDARD BLOCK JACOBI")
spla.gmres(spla.LinearOperator((m,m),matvec=lambda x : luAb.solve(A@x)),
    luAb.solve(b),callback=callback,restart=restart,maxiter=maxiter,callback_type="x")

plt.semilogy(errs,linewidth=2)
plt.xlabel("GMRES Iteration")
plt.ylabel("Residual")
it=0
errs=[]


def blr_precon(x):
    x=x.reshape((m,1))
    x=eval_blr(blr,m,blocksize,x)
    return x.reshape((m,))


x0=rng.uniform(-1,1,size=m)
x1=rng.uniform(-1,1,size=m)
alpha=0.2
beta=-0.1

print("LINEARITY TEST")
print(np.linalg.norm(blr_precon(alpha*x0 + beta*x1) - alpha*blr_precon(x0) - beta*blr_precon(x1)))



print("LEARNED BLR PRECONDITIONER")
spla.gmres(spla.LinearOperator((m,m),lambda x : blr_precon(luAb.solve(A@x))),
        blr_precon(luAb.solve(b)),
        callback=callback,restart=restart,maxiter=maxiter,callback_type="x")

plt.semilogy(errs,linewidth=2)
plt.legend(["Block Jacobi","Learned BLR"])
plt.title("Comparing Block Jacobi to Block-Low-Rank with same block size")
plt.savefig("precon_compare.svg")
