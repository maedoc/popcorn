import tqdm
import time
import numpy as np

# load data
gdist = np.load('./vert2vert_gdist_mat_32k.npz')
gdist_indices = gdist['indices']
gdist_indptr = gdist['indptr']
gdist_data = gdist['data'].astype(np.float32)
gdist_shape = tuple(gdist['shape'])

f2v = np.load('./fibre2vertex_assignment_32k_15M.npz')['fibre2vertex_assignment']

v2vL = np.load('./vert2vert_lengths_32k_15M.npz')
v2vL_indices = v2vL['indices']
v2vL_indptr = v2vL['indptr']
v2vL_data = v2vL['data'].astype(np.float32)
v2vL_shape = tuple(v2vL['shape'])

v2vW = np.load('./vert2vert_weights_32k_15M.npz')
v2vW_indices = v2vW['indices']
v2vW_indptr = v2vW['indptr']
v2vW_data = v2vW['data'].astype(np.float32)
v2vW_shape = tuple(v2vW['shape'])

# sanity check
assert gdist_shape == v2vL_shape == v2vW_shape
nvtx = gdist_shape[0]
np.testing.assert_equal(v2vL_indices, v2vW_indices)
np.testing.assert_equal(v2vL_indptr, v2vW_indptr)
print(f'M nnz gdist {gdist_data.size/1e6:0.2f}, v2v {v2vL_data.size/1e6:0.2f}')
print(f'MB gdist {gdist_data.nbytes*2/1e6:0.2f}, v2v {v2vL_data.nbytes*2/1e6:0.2f}')
print(f'Gdist min {gdist_data.min():.2f} mm, max {gdist_data.max():.2f} mm')
print(f'v2v min {v2vL_data.min():.2f} mm, max {v2vL_data.max():.2f} mm')

# look at distribution of min distance per vertex
import scipy.sparse
gdmin = np.array([gdist_data[gdist_indptr[i]:gdist_indptr[i+1]].min() for i in tqdm.trange(nvtx)])
vLmin = np.array([
    v2vL_data[v2vL_indptr[i]:v2vL_indptr[i+1]].min()
    for i in tqdm.trange(nvtx) if v2vL_indptr[i+1] > v2vL_indptr[i]])

figure(figsize=(10,4))
subplot(121); hist(np.log(vLmin),50,log=True); xlabel('log mm'); title('v2v'); grid(1)
subplot(122); hist(np.log(gdmin),50,log=True); xlabel('log mm'); title('gdist'); grid(1)
tight_layout()

# compute delays
dt = 0.1 # ms
vel = 10.0 # mm/ms
nh = 256 # max history steps
gdist_idelays = (gdist_data / vel / dt).astype(np.int32)
v2vL_idelays = (v2vL_data / vel / dt).astype(np.int32)
# print delay min max
print(f'gdist delay min {gdist_idelays.min()}, max {gdist_idelays.max()}')
print(f'v2v delay min {v2vL_idelays.min()}, max {v2vL_idelays.max()}')
assert v2vL_idelays.max() < nh

# compute lc kernel
gd = np.r_[:10:100j]
figure(); plot(gd, np.exp(-(gd/3)**2)); ylim([0,1]); grid(1)
lc_kernel = np.exp(-(gdist_data/3)**2)

# benchmark numpy performance
buf = np.zeros((nh, nvtx), dtype=np.float32)
tic = time.time()
niter = 5
for _ in tqdm.trange(niter):
    gc = np.add.reduceat(buf[v2vL_idelays, v2vL_indices]*v2vW_data, v2vL_indptr[:-1])
    lc = np.add.reduceat(buf[gdist_idelays, gdist_indices]*lc_kernel, gdist_indptr[:-1])
toc = (time.time() - tic)/niter

# membw & flop count proportional to nnz
v2v_nnz = v2vL_data.size
gdist_nnz = gdist_data.size
flop = (v2v_nnz + gdist_nnz)*2 # mul+add, ~56 Mflop
membw = (v2v_nnz + gdist_nnz)*4*4 # load idelay,indices,buf,weight, forget write
print(f'{flop/toc/1e6:0.2f} Mflops, {membw/toc/1e9:0.2f} GB/s')

# numba
import numba as nb
@nb.njit(parallel=True, fastmath=True, boundscheck=False)
def compute_couplings(gc, lc, buf,
                      v2vL_idelays, v2vL_indices, v2vL_indptr, v2vW_data,
                      gdist_idelays, gdist_indices, gdist_indptr, lc_kernel):
    for i in nb.prange(nvtx):
        acc_gc = nb.float64(0.0)
        for j in range(v2vL_indptr[i], v2vL_indptr[i+1]):
            acc_gc += buf[v2vL_idelays[j], v2vL_indices[j]]*v2vW_data[j]
        gc[i] = acc_gc
        acc_lc = nb.float64(0.0)
        for j in range(gdist_indptr[i], gdist_indptr[i+1]):
            acc_lc += buf[gdist_idelays[j], gdist_indices[j]]*lc_kernel[j]
        lc[i] = acc_lc

buf = np.zeros((nh, nvtx), dtype=np.float32)
gc = np.zeros(nvtx, dtype=np.float32)
lc = np.zeros(nvtx, dtype=np.float32)
run_nb = lambda : compute_couplings(gc, lc, buf,
                      v2vL_idelays, v2vL_indices, v2vL_indptr, v2vW_data,
                      gdist_idelays, gdist_indices, gdist_indptr, lc_kernel)
run_nb()
tic = time.time()
niter = 10
for _ in tqdm.trange(niter):
    run_nb()
toc = (time.time() - tic)/niter
print(f'{flop/toc/1e6:0.2f} Mflops, {membw/toc/1e9:0.2f} GB/s')
# about 111 Mflops, 0.89 GB/s

# translate to ispc + opencl batched

# parallelize
