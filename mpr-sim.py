"""
A hi-res MPR w/ two sparse coupling matrices.

"""

import numpy as np
import scipy.io
import tqdm
import pyopencl as cl
import pyopencl.array as ca
import pyopencl.clrandom as cr


class Util:

    def __init__(self):
        self.init_cl()
        self._progs = []

    def init_cl(self):
        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices(device_type=cl.device_type.GPU)[0]
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context)
        self.rng = cr.PhiloxGenerator(self.context)

    def randn(self, *shape, mu=0, sigma=1):
        return self.rng.normal(cq=self.queue, dtype='f', shape=shape, mu=mu, sigma=sigma)

    def zeros(self, *shape):
        return ca.zeros(self.queue, shape, dtype='f')

    def init_vectors(self, names, shape, method='zeros', **kwargs):
        for name in names.split(' '):
            setattr(self, name, getattr(self, method)(*shape, **kwargs))

    def load_kernel(self, fname, *knames):
        with open(fname, 'r') as f:
            src = f.read()
        prog = cl.Program(self.context, src).build()
        self._progs.append(prog)
        for kname in knames:
            setattr(self, kname, getattr(prog, kname))

    def move_csr(self, name, csr):
        setattr(self, name + '_data', ca.to_device(self.queue, csr.data))
        setattr(self, name + '_indices', ca.to_device(self.queue, csr.indices))
        setattr(self, name + '_indptr', ca.to_device(self.queue, csr.indptr))


def make_step(nvtx, num_sims, SC, LC, dt, r_noise_scale):
    "Setup functions for time stepping."

    # CL kernels expect float32 scalars
    dt = np.float32(dt)

    util = Util()
    # allocate required memory for states & Heun integration
    util.init_vectors('nr nV r V ri Vi cr cV dr1 dV1 dr2 dV2 zr zV bold', (nvtx, num_sims))
    # alloc balloon state
    util.init_vectors('dsfvq1 dsfvq2 sfvqi sfvq zsfvq', (4, nvtx, num_sims))
    util.sfvq[1:] = np.float32(1.0)
    # eta is the only varying parameter for now
    util.init_vectors('params', (nvtx, num_sims), 'randn', mu=-6.0, sigma=0.1)
    # move sparse matrix data to GPU
    util.move_csr('sc', SC)
    util.move_csr('lc', LC)
    # load kernels
    util.load_kernel('./spmv.opencl.c', 'spmv')
    util.load_kernel('./mpr.opencl.c', 'mpr_dfun')
    util.load_kernel('./heun.opencl.c', 'heun_pred', 'heun_corr')
    util.load_kernel('./balloon.opencl.c', 'balloon_dfun', 'balloon_readout')

    # handle boilerplate for kernel launches
    def do(f, *args, nvtx=nvtx):
        args = [(arg.data if hasattr(arg, 'data') else arg) for arg in args]
        f(util.queue, (nvtx, num_sims), (1, num_sims), *args)

    def coupling(cr, cV, r, V):
        "Compute coupling terms."
        # global coupling transmits rate information
        do(util.spmv, cr, r, util.sc_data, util.sc_indices, util.sc_indptr)
        # local coupling transmits potential
        do(util.spmv, cV, V, util.lc_data, util.lc_indices, util.lc_indptr)

    def dfun(dr, dV, r, V):
        "Compute MPR derivatives."
        coupling(util.cr, util.cV, r, V)
        do(util.mpr_dfun, dr, dV, r, V, util.cr, util.cV, util.params)

    def step():
        "Do one Heun step."
        # sample noise
        util.rng.fill_normal(util.zr, sigma=r_noise_scale)

        # predictor step computes dr1,dV1 from states r,V
        dfun(util.dr1, util.dV1, util.r, util.V)

        # and puts Euler result into intermediate arrays ri,Vi
        do(util.heun_pred, dt, util.ri, util.r, util.dr1, util.zr)
        do(util.heun_pred, dt, util.Vi, util.V, util.dV1, util.zV)

        # corrector step computes dr2,dV2 from intermediate states ri,Vi
        dfun(util.dr2, util.dV2, util.ri, util.Vi)

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


def main():

    # load the global and local connectivity matrices
    mat = scipy.io.loadmat('matrices.mat')
    SC = mat['SC']
    LC = mat['LC']

    if False:
        # for testing, can run just a subset of the network, like first 512
        # vertices, but could also be a mask selecting just 5 regions, etc.
        nvtx = 512
        SC = SC[:nvtx, :nvtx]
        LC = LC[:nvtx, :nvtx]
        print('reduced network shape to', SC.shape, LC.shape)

    dt = 0.1
    r_noise_scale = 0.1
    nvtx = SC.shape[0]
    num_sims = 256

    # prepare the GPU arrays and stepping function
    util, step, bold_step = make_step(nvtx, num_sims, SC, LC, dt, r_noise_scale)

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
        step()
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
