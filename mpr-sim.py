"""
A hi-res MPR w/ two sparse coupling matrices.

"""

import numpy as np
import scipy.sparse
import tqdm
from clutils import *
import pyopencl.array as ca


def make_step(nvtx, num_sims, W, iL, K, iG, dt, r_noise_scale):
    "Setup functions for time stepping."

    # CL kernels expect float32 scalars
    dt = np.float32(dt)
    nh_r = iL.max() + 1
    nh_v = iG.max() + 1

    # max delay determines buffer size
    print((nh_r + nh_v) * nvtx * num_sims * 4 >> 20, 'min buf size')

    util = Util()
    # allocate required memory for states & Heun integration
    util.init_vectors('nr nV r V ri Vi cr cV dr1 dV1 dr2 dV2 zr zV bold', (nvtx, num_sims))
    # alloc delay buffers
    util.init_vectors('rbuf', (nh_r, nvtx, num_sims))
    util.init_vectors('vbuf', (nh_v, nvtx, num_sims))
    util.vbuf[:] = np.float32(-2.0)
    # alloc balloon state
    util.init_vectors('dsfvq1 dsfvq2 sfvqi sfvq zsfvq', (4, nvtx, num_sims))
    util.sfvq[1:] = np.float32(1.0)
    # eta is the only varying parameter for now
    util.init_vectors('params', (nvtx, num_sims), 'randn', mu=-6.0, sigma=0.1)
    # move sparse matrix data to GPU
    util.move_csr(W=W, K=K)
    util.move(iL=iL, iG=iG)
    # load kernels
    util.load_kernel('./delays.opencl.c', 'delays_batched', N=32, B=num_sims)
    util.load_kernel('./mpr.opencl.c', 'mpr_dfun')
    util.load_kernel('./heun.opencl.c', 'heun_pred', 'heun_corr')
    util.load_kernel('./balloon.opencl.c', 'balloon_dfun', 'balloon_readout')

    # handle boilerplate for kernel launches
    def do(f, *args, nvtx=nvtx):
        args = [(arg.data if hasattr(arg, 'data') else arg) for arg in args]
        f(util.queue, (nvtx, num_sims), (1, num_sims), *args)

    def coupling(t, cr, cV, r, V):
        "Compute coupling terms."

        # delays_batched(nvtx, nh, t, out, buf, weights, idelays, indices, indptr)
        t = np.int32(t)

        # global coupling transmits rate information
        do(util.delays_batched, np.int32(nvtx), np.int32(nh_r), t, cr, util.rbuf,
                util.W_data, util.iL, util.W_indices, util.W_indptr)
        # local coupling transmits potential
        do(util.delays_batched, np.int32(nvtx), np.int32(nh_v), t, cV, util.vbuf,
                util.K_data, util.iG, util.K_indices, util.K_indptr)

    def dfun(t, dr, dV, r, V):
        "Compute MPR derivatives."
        coupling(t, util.cr, util.cV, r, V)
        do(util.mpr_dfun, dr, dV, r, V, util.cr, util.cV, util.params)

    def step(t):
        "Do one Heun step."

        # update rbuf & vbuf
        util.rbuf[t%nh_r] = util.r
        util.vbuf[t%nh_v] = util.V

        # sample noise
        util.rng.fill_normal(util.zr, sigma=r_noise_scale)

        # predictor step computes dr1,dV1 from states r,V
        dfun(t, util.dr1, util.dV1, util.r, util.V)

        # and puts Euler result into intermediate arrays ri,Vi
        do(util.heun_pred, dt, util.ri, util.r, util.dr1, util.zr)
        do(util.heun_pred, dt, util.Vi, util.V, util.dV1, util.zV)

        # corrector step computes dr2,dV2 from intermediate states ri,Vi
        dfun(t, util.dr2, util.dV2, util.ri, util.Vi)

        # and writes Heun result into arrays r,V
        do(util.heun_corr, dt, util.r, util.r, util.dr1, util.dr2, util.zr)
        do(util.heun_corr, dt, util.V, util.V, util.dV1, util.dV2, util.zV)


    def bold_step(dt):
        "Do one step of the balloon model."
        dt = np.float32(dt)
        # do Heun step on balloon model, using r as neural input
        do(util.balloon_dfun, util.dsfvq1, util.sfvq, util.r)
        do(util.heun_pred, dt, util.sfvqi, util.sfvq, util.dsfvq1, util.zsfvq, nvtx=4*nvtx)
        do(util.balloon_dfun, util.dsfvq2, util.sfvqi, util.r)
        do(util.heun_corr, dt, util.sfvq, util.sfvq, util.dsfvq1, util.dsfvq2, util.zsfvq, nvtx=4*nvtx)
        # update bold signal
        do(util.balloon_readout, util.sfvq, util.bold)

    return util, step, bold_step


def load_npz_to_csr(npz_fname):
    npz = np.load(npz_fname)
    csr = scipy.sparse.csr_matrix(
        (npz['data'], npz['indices'], npz['indptr']),
        shape=npz['shape'])
    return csr.astype('f')


def main():

    # load the global and local connectivity matrices
    G = load_npz_to_csr('vert2vert_gdist_mat_32k.npz')
    L = load_npz_to_csr('vert2vert_lengths_32k_15M.npz')
    W = load_npz_to_csr('vert2vert_weights_32k_15M.npz')
    assert G.shape == L.shape == W.shape

    if True:
        # for testing, can run just a subset of the network, like first 512
        # vertices, but could also be a mask selecting just 5 regions, etc.
        nvtx = 512
        G = G[:nvtx][:,:nvtx]
        L = L[:nvtx][:,:nvtx]
        W = W[:nvtx][:,:nvtx]
        print('reduced network shape to', G.shape, L.shape, W.shape)

    dt = 0.1
    r_noise_scale = 0.1
    nvtx = L.shape[0]
    num_sims = 32

    # make lc kernel from gdist
    K = G.copy()
    K.data = np.exp(-K.data/5.0).astype('f')

    # prepare extra info for delays
    local_velocity = 1.0
    v2v_velocity = 10.0
    iG = (G.data / local_velocity / dt).astype('i') # rounds down
    iL = (L.data / v2v_velocity / dt).astype('i')

    # prepare the GPU arrays and stepping function
    util, step, bold_step = make_step(nvtx, num_sims, W, iL, K, iG, dt, r_noise_scale)

    # set initial conditions
    util.rng.fill_normal(util.r, mu=0.1, sigma=0.2)
    util.rng.fill_normal(util.V, mu=-2.1, sigma=0.2)

    # simulation times
    import time
    tic = time.time()
    minute = int(60e3 / dt)

    # neural field iterations & output storage
    niter = 15*minute  # this many iterations
    nskip = int(10/dt)       # but only save every nskip iterations to file
    rs_shape = niter//nskip+1, nvtx, num_sims
    print(f'output rs.npy ~{(np.prod(rs_shape)*4) >> 30} GB')
    rs = np.lib.format.open_memmap(
            'rs.npy', mode='w+', dtype='f',
            shape=rs_shape)

    # bold iterations & output storage 
    bold_dtskip = 180
    bold_nskip = int(180/dt)         # sample bold every 180 ms
    bolds_shape = niter//bold_nskip+1, nvtx, num_sims
    print(f'output bolds.npy ~{(np.prod(bolds_shape)*4) >> 30} GB')
    bolds = np.lib.format.open_memmap(
            'bolds.npy', mode='w+', dtype='f',
            shape=bolds_shape)

    # do time stepping
    for i in tqdm.trange(niter):
        step(i)
        # bold is slow, don't step it every time
        if i % bold_dtskip == 0:
            bold_step(dt*1e-3*bold_dtskip)
        # save bold & states every few steps
        if i % bold_nskip == 0:
            util.bold.get_async(util.queue, bolds[i//bold_nskip])
        if i % nskip == 0:
            util.r.get_async(util.queue, rs[i//nskip])

    # opencl operations are asynchronous, wait for everything to finish & report time
    print('finishing...')
    util.queue.finish()
    toc = time.time() - tic
    print(f'done in {toc:0.2f}s, {toc/(niter)*1e3:0.3f} ms/iter of {num_sims}')

if __name__ == '__main__':
    main()

    import numpy as np
    import pylab as pl
    rs = np.lib.format.open_memmap('rs.npy')
    bolds = np.lib.format.open_memmap('bolds.npy')

    pl.figure()
    for i in range(25):
        pl.subplot(5, 5, i + 1)
        pl.plot(rs[::100, ::100, i], 'k', alpha=0.1)

    pl.suptitle('r')
    pl.tight_layout()
    pl.figure()
    for i in range(25):
        pl.subplot(5, 5, i + 1)
        pl.plot(bolds[:, ::100, i], 'k', alpha=0.1)

    pl.suptitle('bold')
    pl.tight_layout()
    pl.show()
