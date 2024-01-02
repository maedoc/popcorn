import numpy as np
import scipy
import tqdm

def load_npz_to_csr(npz_fname):
    npz = np.load(npz_fname)
    csr = scipy.sparse.csr_matrix(
        (npz['data'], npz['indices'], npz['indptr']),
        shape=npz['shape'])
    return csr.astype('f')

# load the global and local connectivity matrices
G = load_npz_to_csr('vert2vert_gdist_mat_32k.npz')
L = load_npz_to_csr('vert2vert_lengths_32k_15M.npz')
W = load_npz_to_csr('vert2vert_weights_32k_15M.npz')

# take subset for benchmarking
nv = 10*1024
G = G[:nv][:,:nv]
L = L[:nv][:,:nv]
W = W[:nv][:,:nv]
assert G.shape == L.shape == W.shape

# some parameters
dt = np.float32(0.1)
r_noise_scale = 0.1

# make lc kernel from gdist
K = G.copy()
K.data = np.exp(-K.data/5.0).astype('f')

# prepare extra info for delays
local_velocity = 1.0
v2v_velocity = 10.0
iL = (L.data / v2v_velocity / dt).astype('i')
iG = (G.data / local_velocity / dt).astype('i')

# use next power of 2 of max delay
nh_r = 2**int(np.ceil(np.log2( iL.max() + 1 )))
nh_v = 2**int(np.ceil(np.log2( iG.max() + 1 )))
print('history len', nh_r, nh_v)

# allocate buffers
rbuf = np.zeros((nv, nh_r), 'f')
vbuf = np.zeros((nv, nh_v), 'f')
cr = np.zeros((2, nv), 'f')
cv = np.zeros((2, nv), 'f')

# compile C code
import os, ctypes
from ctypes import c_int, c_float
fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)
os.system('gcc -fopenmp -O3 -mavx2 -ffast-math -march=native -c delays2.c -o delays2.c.o')
os.system('gcc -shared -fopenmp delays2.c.o -o delays2.c.so')
lib = ctypes.CDLL('./delays2.c.so')
lib.delays1.restype = None
lib.delays1.argtypes = (
    c_int, c_int, c_int,
    fvec, fvec, fvec, ivec, ivec, ivec
)
lib.delays2.restype = None
lib.delays2.argtypes = (
    c_int, c_int, c_int,
    fvec, fvec, fvec, fvec, ivec, ivec, ivec
)
lib.delays2_upbuf.restype = None
lib.delays2_upbuf.argtypes = (
    c_int, c_int, c_int,
    fvec, fvec, fvec, fvec, ivec, ivec, ivec, fvec
)
lib.mpr_dfun.restype = None
lib.mpr_dfun.argtypes = (
    c_int, fvec, fvec, fvec, fvec, fvec,
    c_float, c_float, c_float, c_float, c_float, c_float
)

# numba impl for reference
import numba
@numba.njit(fastmath=True, parallel=True, boundscheck=False)
def nb_delays2(buf,nh,t,idelays,indices,weights,indptr,c):
    nhm = numba.int32(nh - 1)
    for i in numba.prange(c.shape[1]):
        acc1 = numba.float32(0.0)
        acc2 = numba.float32(0.0)
        for j in range(indptr[i], indptr[i+1]):
            roll_t = numba.int32(nh + t - idelays[j])
            acc1 += weights[j] * buf[indices[j], numba.int32(roll_t) & nhm]
            acc2 += weights[j] * buf[indices[j], numba.int32(roll_t+1) & nhm]
        c[0,i] = acc1
        c[1,i] = acc2

# check C code againt numpy impl
def np_delays1(buf,nh,t,idelays,indices,weights,indptr,c):
    xij = buf[indices, (nh + t - idelays) & (nh-1)]
    np.add.reduceat(xij*weights, indptr[:-1], axis=0, out=c)
    c[np.argwhere(np.diff(indptr)==0)] = 0
def np_delays2(buf,nh,t,idelays,indices,weights,indptr,c):
    xij = buf[indices, (nh + t + np.c_[0,1].T - idelays) & (nh-1)]
    np.add.reduceat(xij*weights, indptr[:-1], axis=1, out=c)
    c[:,np.argwhere(np.diff(indptr)==0)] = 0
# fill buffer with some thing
rbuf[:] = np.random.randn(*rbuf.shape).astype('f')
# test variant 1
cr_np = cr.copy()
np_delays1(rbuf, nh_r, 42, iL, W.indices, W.data, W.indptr, cr_np[0])
lib.delays1(nv, nh_r, 42, cr[0], rbuf, W.data, iL, W.indices, W.indptr)
np.testing.assert_allclose(cr_np[0], cr[0], 1e-3, 1e-3)
# test variant 2
cr[:] = cr_np[:] = 0
np_delays2(rbuf, nh_r, 42, iL, W.indices, W.data, W.indptr, cr_np)
lib.delays2(nv, nh_r, 42, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr)
np.testing.assert_allclose(cr_np, cr, 1e-3, 1e-3)
# test numba
cr[:] = cr_np[:] = 0
np_delays2(rbuf, nh_r, 42, iL, W.indices, W.data, W.indptr, cr_np)
nb_delays2(rbuf, nh_r, 42, iL, W.indices, W.data, W.indptr, cr)
np.testing.assert_allclose(cr_np, cr, 1e-3, 1e-3)

# benchmark implementations, numpy is slow
print('benchmarking numpy, numba, C1, C2')
for i in tqdm.trange(32):
    np_delays2(rbuf, nh_r, i, iL, W.indices, W.data, W.indptr, cr)
    np_delays2(vbuf, nh_v, i, iG, K.indices, K.data, K.indptr, cv)
# numba is a bit faster
for i in tqdm.trange(128):
    nb_delays2(rbuf, nh_r, i, iL, W.indices, W.data, W.indptr, cr)
    nb_delays2(vbuf, nh_v, i, iG, K.indices, K.data, K.indptr, cv)
# first variant
for i in tqdm.trange(256):
    lib.delays1(nv, nh_r, i,   cr[0], rbuf, W.data, iL, W.indices, W.indptr)
    lib.delays1(nv, nh_r, i+1, cr[1], rbuf, W.data, iL, W.indices, W.indptr)
    lib.delays1(nv, nh_v, i,   cv[0], vbuf, K.data, iG, K.indices, K.indptr)
    lib.delays1(nv, nh_v, i+1, cv[1], vbuf, K.data, iG, K.indices, K.indptr)
# 2nd variant
for i in tqdm.trange(256):
    lib.delays2(nv, nh_r, i, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr)
    lib.delays2(nv, nh_v, i, cv[0], cv[1], vbuf, K.data, iG, K.indices, K.indptr)

# now the rest of the simulation
Delta = 2.0
tau = 1.0
J = 15.0
I = 1.0
Delta_o_pi_tau = Delta / (np.pi * tau)
pi2tau2 = np.pi**2*tau**2
k = 1e-5

def np_dfun(drv, rv, cr, cv, eta):
    r, v = rv
    drv[0] = Delta_o_pi_tau + 2*v*r
    drv[1] = v*v - pi2tau2*r*r + eta + J * tau * r + I + k*(cr + cv)

def np_heun(i, drv, rv, dt, eta, use_c=False):
    # eval coupling for both stages in one pass
    lib.delays2_upbuf(nv, nh_r, i, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr, rv[0])
    lib.delays2_upbuf(nv, nh_v, i, cv[0], cv[1], vbuf, K.data, iG, K.indices, K.indptr, rv[1])
    # predictor stage
    if use_c:
        lib.mpr_dfun(nv, drv[0], rv, cr[0], cv[0], eta, Delta_o_pi_tau, pi2tau2, J, tau, I, k)
    else:
        np_dfun(drv[0], rv, cr[0], cv[0], eta)
    rvi = rv + dt*drv[0]
    rvi[0,rvi[0] < 0] = 0
    # corrector stage
    if use_c:
        lib.mpr_dfun(nv, drv[1], rvi, cr[1], cv[1], eta, Delta_o_pi_tau, pi2tau2, J, tau, I, k)
    else:
        np_dfun(drv[1], rvi, cr[1], cv[1], eta)
    rv += dt/2*(drv[0] + drv[1])
    rv[0,rv[0] < 0] = 0


@numba.njit(fastmath=True, parallel=True, boundscheck=True)
def _nb_heun_stages(nv, dt, rv, cr, cv, eta, Delta_o_pi_tau, pi2tau2, J, tau, I, k):
    dt2 = numba.float32(dt/2)
    for i in numba.prange(nv):
        # predictor stage
        r = rv[0,i]
        v = rv[1,i]
        dr1 = Delta_o_pi_tau + 2*v*r
        dv1 = v*v - pi2tau2*r*r + eta[i] + J * tau * r + I + k*(cr[0,i] + cv[0,i])
        r += dt*dr1
        v += dt*dv1
        if r < 0:
            r = 0
        # corrector stage
        dr2 = Delta_o_pi_tau + 2*v*r
        dv2 = v*v - pi2tau2*r*r + eta[i] + J * tau * r + I + k*(cr[1,i] + cv[1,i])
        r = rv[0,i] + dt2*(dr1 + dr2)
        v = rv[1,i] + dt2*(dv1 + dv1)
        if r < 0:
            r = 0
        # update outputs
        rv[0,i] = r
        rv[1,i] = v
# try make it faster with numba, only 3 passes O(nv)
def nb_heun(i, rv, dt, eta):
    lib.delays2_upbuf(nv, nh_r, i, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr, rv[0])
    lib.delays2_upbuf(nv, nh_v, i, cv[0], cv[1], vbuf, K.data, iG, K.indices, K.indptr, rv[1])
    _nb_heun_stages(nv, dt, rv, cr, cv, eta, Delta_o_pi_tau, pi2tau2, J, tau, I, k)


import pylab as pl
print('running full sim, np, np+c, numba+c')
steps = [
        lambda i: np_heun(i, drv, rv, dt, eta, use_c=False),
        lambda i: np_heun(i, drv, rv, dt, eta, use_c=True),
        lambda i: nb_heun(i, rv, dt, eta),
        ]
pl.figure()
for j, step in enumerate(steps):
    np.random.seed(42)
    rbuf[:] = np.random.randn(*rbuf.shape).astype('f') / 10 + 0.1
    rbuf[rbuf<0] = 0
    vbuf[:] = np.random.randn(*vbuf.shape).astype('f') - 2.0
    drv = np.zeros((2, 2, nv), 'f')
    rv = np.zeros((2, nv), 'f')
    dt = np.float32(0.1)
    eta = np.zeros((nv,), 'f') - 5.0
    r_trace = []
    for i in tqdm.trange(256):
        step(i)
        r_trace.append(rv[0].copy())
    r_trace = np.array(r_trace)
    pl.subplot(3, 1, j+1)
    pl.plot(r_trace[:, :10], 'k', alpha=0.1)
pl.show()
