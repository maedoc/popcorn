
struct sim {
    const int chunksize;
    const int nvtx;
    const int nh;
    int t;
    
    float *buf;
    float *V;
    const float *weights;
    const int *idelays;
    const int *indices;
    const int *indptr;
    
    const float dt;
    const float Delta;
    const float pi;
    const float tau;
    const float J;
    const float eta;
    const float I;
    const float k;
};


float get_aff_ij(const uniform struct sim &s, const uniform int t, const varying int j) {
    const varying int dij = (s.nh + t - s.idelays[j]) & (s.nh-1);
    return s.weights[j] * s.buf[s.indices[j]*s.nh + dij];
}

task void delays_task(const uniform int start, const uniform int chunksize,
                      const uniform struct sim &s)
{
    // loop over nodes
    for (uniform int i=start; i<(start + chunksize); i++)
    {
        // SIMD the sparse coupling terms
        // we can issue both at once due to delays
        varying float acc1 = 0;
        varying float acc2 = 0;
        foreach (j = s.indptr[i] ... s.indptr[i+1]) {
            acc1 += get_aff_ij(s, s.t, j);
            acc2 += get_aff_ij(s, s.t+1, j);
        }
        uniform float cr1 = reduce_add(acc1);
        uniform float cr2 = reduce_add(acc2);
        
        // rest is not SIMD but hardly matters for performance
        uniform float r = s.buf[i*s.nh + ((s.t) & (s.nh - 1))];
        uniform float V = s.V[i];
        
        // Heun predictor
        uniform float dr1 = s.Delta / (s.pi * s.tau) + 2*V*r;
        uniform float dV1 = V*V - s.pi*s.pi * s.tau*s.tau * r*r + s.eta + s.J * s.tau * r + s.I + s.k*cr1;
        uniform float ri = r + s.dt*dr1;
        ri = ri>0 ? ri : 0;
        uniform float Vi = V + s.dt*dV1;
        
        // Heun corrector
        uniform float dr2 = s.Delta / (s.pi * s.tau) + 2*Vi*ri;
        uniform float dV2 = Vi*Vi - s.pi*s.pi * s.tau*s.tau * ri*ri + s.eta + s.J * s.tau * ri + s.I + s.k*cr2;
        uniform float nr = r + s.dt*0.5*(dr1 + dr2);
        uniform float nV = V + s.dt*0.5*(dV1 + dV2);
        nr = nr>0 ? nr : 0;
        
        // update buffer
        s.buf[i*s.nh + ((s.t + 1) & (s.nh - 1))] = nr;
        s.V[i] = nV;
    }
}

// this function is called by Python and distributes vertices over CPUs
export void delays(uniform struct sim &s)
{
    uniform int chunks = s.nvtx / s.chunksize;
    for (uniform int start=0; start<chunks; start++)
        launch delays_task(start*s.chunksize, s.chunksize, s);
    launch delays_task(chunks*s.chunksize, s.nvtx-chunks*s.chunksize, s);
    sync;
}
