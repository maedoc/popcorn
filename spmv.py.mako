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
