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
    util.init_vectors('nr nV r V ri Vi cr cV dr1 dV1 dr2 dV2 zr zV', (nvtx, num_sims))
    # eta is the only varying parameter for now
    util.init_vectors('params', (nvtx, num_sims), 'randn', mu=-5.0, sigma=0.1)
    # move sparse matrix data to GPU
    util.move_csr('sc', SC)
    util.move_csr('lc', LC)
    # load kernels
    util.load_kernel('./spmv.opencl.c', 'spmv')
    util.load_kernel('./mpr.opencl.c', 'mpr_dfun')
    util.load_kernel('./heun.opencl.c', 'heun_pred', 'heun_corr')

    # handle boilerplate for kernel launches
    def do(f, *args):
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

    return util, step


def main():
    mat = scipy.io.loadmat('matrices.mat')
    SC = mat['SC']
    LC = mat['LC']

    dt = 0.1
    r_noise_scale = 0.1
    nvtx = SC.shape[0]
    num_sims = 32

    # prepare the GPU arrays and stepping function
    util, step = make_step(nvtx, num_sims, SC, LC, dt, r_noise_scale)

    # set initial conditions
    util.rng.fill_normal(util.r, mu=0.1, sigma=0.2)
    util.rng.fill_normal(util.V, mu=-2.1, sigma=0.2)

    # run the simulation
    import time
    tic = time.time()
    niter = 200
    nskip = 1
    rs = np.lib.format.open_memmap(
            'rs.npy', mode='w+', dtype='f',
            shape=(niter//nskip, nvtx, num_sims))
    for i in tqdm.trange(niter):
        step()
        if i % nskip == 0:
            util.r.get_async(util.queue, rs[i//nskip])
    print('finishing...')
    util.queue.finish()
    toc = time.time() - tic
    print(f'done in {toc:0.2f}s, {toc/(niter)*1e3:0.3f} ms/iter of {num_sims}')

if __name__ == '__main__':
    main()

    import pylab as pl
    rs = np.lib.format.open_memmap('rs.npy')
    for i in range(25):
        pl.subplot(5, 5, i + 1)
        pl.plot(rs[:, ::100, i], 'k', alpha=0.1)
    pl.tight_layout()
    pl.show()
