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
bs = 8
rbuf = np.zeros((nv, nh_r, bs), 'f')
vbuf = np.zeros((nv, nh_v, bs), 'f')
cr = np.zeros((2, nv, bs), 'f')
cv = np.zeros((2, nv, bs), 'f')

# compile C code
import os, ctypes
from ctypes import c_int, c_float
fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)
os.system('gcc -fopenmp -O3 -mavx2 -ffast-math -march=native -c delays2.c -o delays2.c.o')
os.system('gcc -shared -fopenmp delays2.c.o -o delays2.c.so')
lib = ctypes.CDLL('./delays2.c.so')
lib.delays2.restype = None
lib.delays2.argtypes = (
    c_int, c_int, c_int,
    fvec, fvec, fvec, fvec, ivec, ivec, ivec
)
lib.delays2_batch.restype = None
lib.delays2_batch.argtypes = (
    c_int, c_int, c_int, c_int,
    fvec, fvec, fvec, fvec, ivec, ivec, ivec
)

# check C code againt numpy impl
def np_delays2(buf,nh,t,idelays,indices,weights,indptr,c):
    xij = buf[indices, (nh + t + np.c_[0,1].T - idelays) & (nh-1)] # (2, nnz, 8)
    np.add.reduceat(xij*weights.reshape(-1,1), indptr[:-1], axis=1, out=c)
    c[:,np.argwhere(np.diff(indptr)==0)] = 0

# fill buffer with some thing
rbuf[:] = np.random.randn(*rbuf.shape).astype('f')

# test variant 2
cr[:] = 0
cr_np = np.zeros_like(cr)
np_delays2(rbuf, nh_r, 42, iL, W.indices, W.data, W.indptr, cr_np)
lib.delays2_batch(bs, nv, nh_r, 42, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr)
np.testing.assert_allclose(cr_np[0], cr[0], 1e-3, 1e-3)
np.testing.assert_allclose(cr_np[1], cr[1], 1e-3, 1e-3)

# benchmark implementations, numpy is slow
print('benchmarking numpy, numba, C1, C2')
for i in tqdm.trange(32):
    np_delays2(rbuf, nh_r, i, iL, W.indices, W.data, W.indptr, cr)
    np_delays2(vbuf, nh_v, i, iG, K.indices, K.data, K.indptr, cv)

# 2nd variant
for i in tqdm.trange(256):
    lib.delays2(nv, nh_r, i, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr)
    lib.delays2(nv, nh_v, i, cv[0], cv[1], vbuf, K.data, iG, K.indices, K.indptr)

# 2nd variant
for i in tqdm.trange(256):
    lib.delays2_batch(bs, nv, nh_r, i, cr[0], cr[1], rbuf, W.data, iL, W.indices, W.indptr)
    lib.delays2_batch(bs, nv, nh_v, i, cv[0], cv[1], vbuf, K.data, iG, K.indices, K.indptr)

