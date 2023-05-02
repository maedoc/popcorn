import popcorn
import pytest
import numpy as np
import scipy.io
try:
    import numba
except ImportError:
    numba = None


mat = scipy.io.loadmat('matrices.mat')
SC = mat['SC']
nvtx = SC.shape[0]


spmv_L = [1, 8]
ispc_chunks = [256, 512]

@pytest.mark.parametrize('chunks', ispc_chunks)
@pytest.mark.parametrize('use_simd', [False, True])
@pytest.mark.parametrize('L', spmv_L)
def test_correct_spmv_ispc(L, use_simd, chunks):
    spmv = popcorn.build_spmv_ispc()
    L = 16
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    out = np.zeros((nvtx, L), 'f')
    spmv(nvtx, L, out, vec, SC.data, SC.indices, SC.indptr, chunks, use_simd)
    assert np.allclose(out, SC@vec)


@pytest.mark.parametrize('chunks', ispc_chunks)
@pytest.mark.parametrize('use_simd', [False, True])
@pytest.mark.parametrize('L', spmv_L)
def test_perf_spmv_ispc(benchmark, L, use_simd, chunks):
    spmv = popcorn.build_spmv_ispc()
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    out = np.zeros((nvtx, L), 'f')
    run = lambda : spmv(nvtx, L, out, vec, SC.data, SC.indices, SC.indptr, chunks, use_simd)
    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_perf_spmv_scipy(benchmark, L):
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    run = lambda : SC@vec
    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_perf_spmv_numba(benchmark, L):
    #pytest.importorskip('numba')
    spmv = popcorn.build_spmv_numba()
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    out = np.zeros((nvtx, L), 'f')
    run = lambda : spmv(out, vec, SC.data, SC.indices, SC.indptr)
    benchmark.group = f'spmv L={L}'
    benchmark(run)

