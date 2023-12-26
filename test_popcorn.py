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


spmv_L = [1, 8, 256]
ispc_chunks = [256, 512]


@pytest.mark.parametrize('chunks', ispc_chunks)
@pytest.mark.parametrize('use_simd', [False, True])
@pytest.mark.parametrize('L', spmv_L)
def test_spmv_ispc(benchmark, L, use_simd, chunks):
    spmv = popcorn.build_spmv_ispc()
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    out = np.zeros((nvtx, L), 'f')
    run = lambda : spmv(nvtx, L, out, vec, SC.data, SC.indices, SC.indptr, chunks, use_simd)
    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_spmv_scipy(benchmark, L):
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    run = lambda : SC@vec
    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_spmv_numba(benchmark, L):
    spmv = popcorn.build_spmv_numba()
    vec = np.random.randn(nvtx, L).astype('f') + 1.0
    out = np.zeros((nvtx, L), 'f')
    run = lambda : spmv(out, vec, SC.data, SC.indices, SC.indptr)
    run()
    np.testing.assert_allclose(out, SC@vec)
    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_spmv_opencl(benchmark, L):
    # boilerplate to copy paste elsewhere
    import pyopencl as cl
    import pyopencl.array as ca
    import pyopencl.clrandom as cr

    try:
        device = cl.get_platforms()[0].get_devices(device_type=cl.device_type.GPU)[0]
    except:
        device = cl.get_platforms()[0].get_devices()[0]
    context = cl.Context([device])
    queue = cl.CommandQueue(context,
                         properties=cl.command_queue_properties.PROFILING_ENABLE)

    # useful functions
    def cl_randn(*shape, rng=cr.PhiloxGenerator(context)):
        return rng.normal(cq=queue, dtype='f', shape=shape, mu=0, sigma=1)

    def cl_move(arr):
        "Copy `arr` to device."
        m_arr = ca.Array(queue, shape=arr.shape, dtype=arr.dtype)
        m_arr.set(arr)
        return m_arr

    sc_data = cl_move(SC.data)
    sc_indices = cl_move(SC.indices)
    sc_indptr = cl_move(SC.indptr)

    spmv = popcorn.build_spmv_opencl(context)

    nvtx = SC.shape[0]
    vec = cl_randn(nvtx, L) + 1.0
    out = ca.zeros(queue, (nvtx, L), 'f')
    run = lambda : spmv(queue, (nvtx, L), (1,L),
              out.data, vec.data,
              sc_data.data, sc_indices.data, sc_indptr.data).wait()
    run()
    out_np = out.get()
    np.testing.assert_allclose(out_np, SC@vec.get(), 1e-6, 1e-6)

    benchmark.group = f'spmv L={L}'
    benchmark(run)


@pytest.mark.parametrize('L', spmv_L)
def test_spmv_cuda(benchmark, L):
    matX = np.random.randn(nvtx, L).astype('f') + 1.0
    benchmark.group = f'spmv cuda={L}'
    matY = benchmark(popcorn.build_spmv_cuda, SC, matX, L)
    if not np.allclose(matY, SC.dot(matX)):
        raise ValueError("Error outside tolerance for host-device vectors")
