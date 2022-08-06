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
from jax import scipy as jsp
import jax.scipy.linalg as jla
import jax.numpy.linalg as jnla
import jax.scipy.sparse.linalg as jspla
from jax import grad, jit, vmap
import jax.numpy as jnp
from functools import partial
import optax
import jaxopt
import jax.lax.linalg as lax_linalg
import jax.lax


from jax.config import config
config.update("jax_enable_x64", True)


@partial(jit, static_argnums=[0,2])
def arnoldi_dgks(A,v,k):
    norm=jnp.linalg.norm
    dot=jnp.dot
    eta=1.0/jnp.sqrt(2.0)

    m=len(v)
    V=jnp.zeros((m,k+1))
    H=jnp.zeros((k+1,k))
    #V[:,0]=v/norm(v)
    V = V.at[:,0].set(v/norm(v))
    for j in range(0,k):
        w=A(V[:,j])
        h=V[:,0:j+1].T @ w
        f=w-V[:,0:j+1] @ h
        s = V[:,0:j+1].T @ f
        f = f - V[:,0:j+1] @ s
        h = h + s
        beta=norm(f)
        #H[j+1,j]=beta
        H = H.at[j+1,j].set(beta)
        #V[:,j+1]=f/beta
        V = V.at[:,j+1].set(f.flatten()/beta)
    return V,H




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


seed=23498732
plot_eigs=True
rng=np.random.default_rng(seed)
m=128
diag=2.0
blocksize=32
batchsize=1
nepochs=500
inner=300
lr=1e-3
k=1
d=1
penalty=1e-6
#opt = optax.adam(lr)
opt = optax.sgd(lr)






@partial(jit, static_argnums=[1,2])
def loss(params,m,blocksize,A,b):
    k=10
    m,ncols=b.shape
    assert(ncols==1)
    @jit
    def Ac(x):
        return A @ eval_blr(params,m,blocksize,x.reshape((m,ncols)))
    _,H=arnoldi_dgks(Ac,b.reshape((m,)),k)
    return sum(H[j+1,j] for j in range(H.shape[0]-1))











A=make_banded_matrix(m,diag,[1,2,3,10,40,100],rng)
Aj = sparse.BCOO.from_scipy_sparse(A)
Afull=A.toarray()
eigA = la.eigvals(Afull)

Ab=make_block_precon(A,blocksize)
luAb=spla.splu(sp.csc_matrix(Ab))
b=np.ones((m,3))
#Ab=A@b
#blr=make_blr_random(A,blocksize,d=d)
#blr=make_blr(A,blocksize,d=d)
blr=make_blr_id(A,blocksize,d=d)


losses=[]
valids=[]
opt_state = opt.init(blr)
b=jnp.array(rng.uniform(size=(m,1)))

plot_eigs=True


for it in range(nepochs):
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
        istr=str(it).zfill(5)
        plt.savefig(f"eigs/{istr}.png")


    start=time.time()
    g = grad(loss)(blr,m,blocksize,Aj,b)
    updates,opt_state = opt.update(g,opt_state)
    blr = optax.apply_updates(blr,updates)
    err=loss(blr,m,blocksize,Aj,b)

    stop=time.time()
    print(f"it = {it},     elapsed = {stop-start : .4f},    loss = {err : 4f},    ")
    if losses and err<min(losses):
        f=open("blr_best.dat","wb")
        pickle.dump(blr,f)

    losses.append(err)



plt.close()
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

