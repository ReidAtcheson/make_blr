import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pickle


from jax.experimental import sparse
from jax import random
import jax.nn as jnn
import jax
import jax.scipy.linalg as jla
from jax import grad, jit, vmap
import jax.numpy as jnp
from functools import partial
import optax
from jax.config import config
config.update("jax_enable_x64", True)


@jit
def update(params,updates,step):
    return [(p0+step*u0,p1+step*u1) for (p0,p1),(u0,u1) in zip(params,updates)]

#Make a random sparse-banded matrix 
#with bands in `bands1
#its diagonal shifted by `diag`
def make_banded_matrix(m,diag,bands,rng):
    subdiags=[rng.uniform(-1,1,m) for _ in bands] + [rng.uniform(0.1,1,m) + diag] + [rng.uniform(-1,1,m) for _ in bands]
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
    return jnp.asarray(blockUs,dtype=np.float64),jnp.asarray(blockVs,dtype=np.float64),jnp.asarray(Ds,dtype=np.float64)


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

    return jnp.asarray(blockUs,dtype=np.float64),jnp.asarray(blockVs,dtype=np.float64),jnp.asarray(Ds,dtype=np.float64)

#Create a preconditioner by blocking up starting with identity
def make_blr_id(A,blocksize,d=1):
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
        Ds.append(jnp.eye(len(ids)))
        for j in range(0,m,blocksize):
            kj=min(j+blocksize,m)-j
            Vs.append(jnp.zeros((d,kj)))
            Us.append(jnp.zeros((ki,d)))
        blockVs.append(jnp.asarray(Vs))
        blockUs.append(jnp.asarray(Us))

    return jnp.asarray(blockUs,dtype=np.float64),jnp.asarray(blockVs,dtype=np.float64),jnp.asarray(Ds,dtype=np.float64)




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
            ((0,2,), (0,1)),
            ((), ())
            ))
        out.append(UVx)
    y=jnp.asarray(out).reshape((nblocks,blocksize,ncols))
    z=y+jax.lax.dot_general(Ds,xr,dimension_numbers=(
            ((2,), (1,)),
            ((0,), (0,))
            ))
    return z.reshape((m,ncols))




@partial(jit, static_argnums=[1,2])
def loss(params,m,blocksize,A):
    blrA=eval_blr(params,m,blocksize,A)
    eigA = jla.eigh(blrA,eigvals_only=True)
    return jnp.vdot(eigA-1.0,eigA-1.0)






seed=23498732
plot_eigs=True
rng=np.random.default_rng(seed)
m=512
diag=2.0
blocksize=32
batchsize=m
nepochs=1
inner=2000
lr=1e-3
k=1
d=1
#opt = optax.adam(lr)
opt = optax.sgd(lr)




A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
Ab=make_block_precon(A,blocksize)
luAb=spla.splu(sp.csc_matrix(Ab))
b=np.ones((m,3))
#Ab=A@b
#blr=make_blr_random(A,blocksize,d=d)
#blr=make_blr(A,blocksize,d=d)
blr=make_blr_id(A,blocksize,d=d)

Ah=luAb.solve(A.toarray())
eigAh = la.eigvals(Ah)
valid = jnp.vdot(eigAh-1.0,eigAh-1.0)




#open from checkpoint
#f=open("blr.dat","rb")
#blr=pickle.load(f)


r=range(0,m,blocksize)

losses=[]
valids=[]
opt_state = opt.init(blr)

print(luAb.solve(b))
print(eval_blr(blr,m,blocksize,b))


for it in range(nepochs):
    print("NEW SUBITERATIONS")
    Afull=A.toarray()
    eigA = la.eigvals(Afull)

    for i in range(0,inner):
        if plot_eigs:
            minx=min(np.real(eigA))
            maxx=max(np.real(eigA))
            miny=min(np.imag(eigA))
            maxy=max(np.imag(eigA))
            plt.close()
            blrA = eval_blr(blr,m,blocksize,Afull)
            eigbA = la.eigvals(blrA)
            plt.scatter(np.real(eigbA),np.imag(eigbA))
            ax = plt.gca()
            ax.set_xlim([minx,maxx])
            ax.set_ylim([miny,maxy])
            istr=str(i).zfill(5)
            plt.savefig(f"eigs/{istr}.png")


        start=time.time()
        err=loss(blr,m,blocksize,Afull)
        g = grad(loss)(blr,m,blocksize,Afull)
        updates,opt_state = opt.update(g,opt_state)
        blr = optax.apply_updates(blr,updates)
        stop=time.time()


        if i==0:
            losses.append(err)
        print(f"it = {i},     elapsed = {stop-start : .4f},    loss = {err : 4f},    valid = {valid : 4f}")
#    if len(losses)>5000:
#        inner=20



plt.semilogy(losses)
plt.title("Loss at start of new epoch")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("loss.svg")
plt.close()


f=open("blr.dat","wb")
pickle.dump(blr,f)
f=open("A.dat","wb")
pickle.dump(A,f)

