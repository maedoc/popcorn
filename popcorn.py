import os
import ctypes
import subprocess
import numpy as np
from ctypes import c_int, c_bool
import mako.template
import numba
try:
     from cuda import cuda
except ImportError:
     cuda = None
from cuda.common import checkCudaErrors, KernelHandle


fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)


# TODO use temp lib directories for builds
def build_spmv_ispc(isa=None):
	os.system('ispc -g spmv.ispc --target=neon-i32x8 --pic -O3 -o spmv.ispc.o')
	os.system('g++ -std=c++11 -fPIC -c tasksys.cpp')
	os.system('g++ -shared tasksys.o spmv.ispc.o -o spmv.ispc.so -lpthread')
	lib = ctypes.CDLL('./spmv.ispc.so')
	lib.spmv.restype = None
	lib.spmv.argtypes = c_int, c_int, fvec, fvec, fvec, ivec, ivec, c_int, c_bool
	return lib.spmv


def build_spmv_opencl(context):
    import pyopencl as cl
    with open('spmv.opencl.c', 'r') as fd:
        prog = cl.Program(context, fd.read()).build()
    return prog.spmv


def build_spmv_numba():
    with open('spmv.py.mako', 'r') as fd:
        template = mako.template.Template(fd.read())
    code = template.render(Lvalues=[1, 8, 256])
    ns = dict()
    exec(code, ns)
    return ns['spmatvecn']


def build_spmv_cuda(sparseMat, matX, nColX):
    with open("cuda/spmv.cu", "r") as f:
         spmv_csr = f.read()
    # Initialize
    checkCudaErrors(cuda.cuInit(0))
    cuDevice = checkCudaErrors(cuda.cuDeviceGet(0))
    _ = checkCudaErrors(cuda.cuCtxCreate(0, cuDevice))

    kernelHandle = KernelHandle(spmv_csr, int(cuDevice))
    _spmv_csr_kernel = kernelHandle.getFunction(b'spmv_csr_kernel')

    nRowSC = sparseMat.shape[0]
    matY = np.zeros((nRowSC, nColX)).astype(dtype=np.float32)
    vecY = np.zeros((nRowSC, 1)).astype(dtype=np.float32)

    bufferSizeData = sparseMat.data.nbytes
    bufferSizeIndices = sparseMat.indices.nbytes
    bufferSizeIndPtr = sparseMat.indptr.nbytes
    bufferSizeVectorX = vecX.nbytes
    bufferSizeVectorY = vecY.nbytes

    _, dMatrix = cuda.cuMemAlloc(bufferSizeData)
    _, dIndices = cuda.cuMemAlloc(bufferSizeIndices)
    _, dPtrs = cuda.cuMemAlloc(bufferSizeIndPtr)
    _, dVecX = cuda.cuMemAlloc(bufferSizeVectorX)
    _, dVecY = cuda.cuMemAlloc(bufferSizeVectorY)

    _, stream = cuda.cuStreamCreate(0)

    cuda.cuMemcpyHtoDAsync(dMatrix, sparseMat.data.ctypes.data, bufferSizeData, stream)
    cuda.cuMemcpyHtoDAsync(dIndices, sparseMat.indices.ctypes.data, bufferSizeIndices, stream)
    cuda.cuMemcpyHtoDAsync(dPtrs, sparseMat.indptr.ctypes.data, bufferSizeIndPtr, stream)

    kernelArgs = ((dMatrix, dPtrs, dIndices, nRowSC, dVecX, dVecY), 
                  (None, None, None, ctypes.c_uint, None, None))

    NUM_THREADS = 256  # Threads per block
    NUM_BLOCKS = (nColX+255)/256 # Blocks per grid

    for col in range(nColX):
        cuda.cuMemsetD32Async(dVecY, 0, nRowSC, stream)
        vecX = matX[:, col].copy()
        cuda.cuMemcpyHtoD(dVecX, vecX.ctypes.data, bufferSizeVectorX)

        cuda.cuLaunchKernel(_spmv_csr_kernel,
                            NUM_BLOCKS, 1, 1,
                            NUM_THREADS, 1, 1,
                            0, stream,
                            kernelArgs,
                            0)

        cuda.cuMemcpyDtoH(vecY, dVecY, bufferSizeVectorY)
        matY[:, [col]] = vecY

    cuda.cuStreamDestroy(stream)
    cuda.cuMemFree(dMatrix)
    cuda.cuMemFree(dIndices)
    cuda.cuMemFree(dPtrs)
    cuda.cuMemFree(dVecX)
    cuda.cuMemFree(dVecY)
    kernelHandle.clear()

    return matY
