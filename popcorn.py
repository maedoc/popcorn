import os
import ctypes
import subprocess
import numpy as np
from ctypes import c_int, c_bool
import mako.template
import numba

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


def build_spmv_numba():
    template = mako.template.Template('''
import numba
@numba.njit(parallel=True, boundscheck=True, fastmath=True)
def spmatvecn(out, b, data, ir, jc):
    L = out.shape[1]

% for L in Lvalues:
    if L == ${L}:
        assert out.shape[1] == b.shape[1] == ${L}

        for c in numba.prange(jc.size - 1):
% for l in range(L):
            acc${l} = numba.float64(0.0)
% endfor
            for r in range(jc[c],jc[c+1]):
% for l in range(L):
                acc${l} += data[r] * b[ir[r],${l}]
% endfor
% for l in range(L):
            out[c,${l}] = acc${l}
% endfor
% endfor
''')
    code = template.render(Lvalues=[1, 2, 4, 8, 16])
    ns = dict()
    exec(code, ns)
    return ns['spmatvecn']
