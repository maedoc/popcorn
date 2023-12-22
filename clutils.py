import pyopencl as cl
import pyopencl.array as ca
import pyopencl.clrandom as cr


class Util:

    def __init__(self):
        self.init_cl()
        self._progs = []

    def init_cl(self):
        # TODO multi device
        self.context = cl.create_some_context() # cl.Context([self.device])
        print(self.context)
        self.queue = cl.CommandQueue(self.context)
        self.rng = cr.PhiloxGenerator(self.context)

    def randn(self, *shape, mu=0, sigma=1):
        return self.rng.normal(cq=self.queue, dtype='f', shape=shape, mu=mu, sigma=sigma)

    def zeros(self, *shape):
        return ca.zeros(self.queue, shape, dtype='f')

    def init_vectors(self, names, shape, method='zeros', **kwargs):
        for name in names.split(' '):
            setattr(self, name, getattr(self, method)(*shape, **kwargs))

    def load_kernel(self, fname, *knames, **buildopts):
        with open(fname, 'r') as f:
            src = f.read()
        buildoptlist = []
        for key, val in buildopts.items():
            buildoptlist.extend(['-D', f'{key}={val}'])
        prog = cl.Program(self.context, src).build(buildoptlist)
        self._progs.append(prog)
        for kname in knames:
            setattr(self, kname, getattr(prog, kname))

    def move(self, **names):
        for name, csr in names.items():
            setattr(self, name, ca.to_device(self.queue, csr))


    def move_csr(self, **names):
        for name, csr in names.items():
            setattr(self, name + '_data', ca.to_device(self.queue, csr.data))
            setattr(self, name + '_indices', ca.to_device(self.queue, csr.indices))
            setattr(self, name + '_indptr', ca.to_device(self.queue, csr.indptr))

