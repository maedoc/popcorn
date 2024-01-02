void delays1(int nv, int nh, int t,
             float *out,
             float *buf, float *weights, int *idelays, int *indices, int *indptr)
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;
    #pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
        float acc1 = 0.0f;
        #pragma omp simd reduction(+:acc1)
        for (int j=indptr[i]; j<indptr[i+1]; j++) {
            float *b = buf + indices[j]*nh;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            acc1 += w * b[roll_t & nhm];
        }
        out[i] = acc1;
    }
}

void delays2(int nv, int nh, int t,
             float *out1, float *out2,
             float *buf, float *weights, int *idelays, int *indices, int *indptr)
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;
    #pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
        // compute coupling terms for both Heun stages
        float acc1 = 0.0f, acc2 = 0.0f;
        #pragma omp simd reduction(+:acc1,acc2)
        for (int j=indptr[i]; j<indptr[i+1]; j++) {
            float *b = buf + indices[j]*nh;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            acc1 += w * b[(roll_t+0) & nhm];
            acc2 += w * b[(roll_t+1) & nhm];
        }
        out1[i] = acc1;
        out2[i] = acc2;
    }
}

// variant which updates the buf with current state
void delays2_upbuf(int nv, int nh, int t,
             float *out1, float *out2,
             float *buf, float *weights, int *idelays, int *indices, int *indptr,
             float *x)
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;
    #pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
        // update buffer
        buf[i*nh + ((nh + t) & nhm)] = x[i];
        // compute coupling terms for both Heun stages
        float acc1 = 0.0f, acc2 = 0.0f;
        #pragma omp simd reduction(+:acc1,acc2)
        for (int j=indptr[i]; j<indptr[i+1]; j++) {
            float *b = buf + indices[j]*nh;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            acc1 += w * b[(roll_t+0) & nhm];
            acc2 += w * b[(roll_t+1) & nhm];
        }
        out1[i] = acc1;
        out2[i] = acc2;
    }
}

void mpr_dfun(
                int nv, float *drv, float *rv, float *cr, float *cv, float *eta,
                float Delta_o_pi_tau, float pi2tau2, float J, float tau, float I, float k
              )
{
    float *dr=drv, *dv=drv+nv, *r=rv, *v=rv+nv;
    #pragma omp simd
    for (int i=0; i<nv; i++)
    {
        dr[i] = Delta_o_pi_tau + 2*v[i]*r[i];
        dv[i] = v[i]*v[i] - pi2tau2*r[i]*r[i] + eta[i] + J * tau * r[i] + I + k*(cr[i] + cv[i]);
    }
}
