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
	os.system('ispc -g spmv.ispc --pic -O3 -o spmv.ispc.o')
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
