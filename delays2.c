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

// batch variant
void delays2_batch(int bs, int nv, int nh, int t,
             float *out1, float *out2,
             float *buf, float *weights, int *idelays, int *indices, int *indptr,
             float *x
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // update buffer
            
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l];
            
    // compute coupling
#pragma omp simd
        for (int l=0; l<bs; l++)
            out1[bs*i+l] = out2[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            float *b1 = b + ((roll_t+0) & nhm)*bs;
            float *b2 = b + ((roll_t+1) & nhm)*bs;

            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out1[bs*i + l] += w * b1[l];
                out2[bs*i + l] += w * b2[l];
            }
        }
    }
}
// batch variant
void update_buffer_heaviside(int bs, int nv, int nh, int t, float theta, float *buf,float *x)
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;
#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l] > theta;
    }
}


void upbuf(int bs, int nv, int nh, int t,
             float *buf, float *x
            )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // update buffer  
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l];
    }
            
}

// batch variant
void delays1_batch(int bs, int nv, int nh, int t,
             float *out1,
             float *buf, float *weights, int *idelays, int *indices, int *indptr
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {  
    // compute coupling
#pragma omp simd
        for (int l=0; l<bs; l++)
            out1[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            float *b1 = b + ((roll_t) & nhm)*bs; 

            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out1[bs*i + l] += w * b1[l];
                // out1[bs*i + l] += indices[j]*nh*bs + ((roll_t) & nhm)*bs + l;
            }
        }
    }
}

// batch variant
void delays1_batch_heavi(int bs, int nv, int nh, int t, float theta,
             float *out1,
             float *buf, float *weights, int *idelays, int *indices, int *indptr,
             float *x
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // update buffer with heaviside step function
            
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l]>theta;
            
    // compute coupling
#pragma omp simd
        for (int l=0; l<bs; l++)
            out1[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            float *b1 = b + ((roll_t+1) & nhm)*bs; // the "+1" to get the coupling for the heun corrector step
            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out1[bs*i + l] += w * b1[l];
            }
        }
    }
}

// batch variant
void delays1_batch_heuncorr_upbuf_heavi(int bs, int nv, int nh, int t, float theta,
             float *out1,
             float *buf, float *weights, int *idelays, int *indices, int *indptr,
             float *x
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // update buffer with heaviside step function
            
#pragma omp simd
        for (int l=0; l<bs; l++)
            buf[(i*nh + ((nh + t) & nhm))*bs + l] = x[i*bs + l]>theta;
            
    // compute coupling
#pragma omp simd
        for (int l=0; l<bs; l++)
            out1[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + (t+1) - idelays[j]; // t+1 for heun corrector step, not to be applied for updating buffer
            float *b1 = b + (roll_t & nhm)*bs; 
            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out1[bs*i + l] += w * b1[l];
            }
        }
    }
}


// batch variant
void delays1_batch_heuncorr_noupbuf(int bs, int nv, int nh, int t,
             float *out, float *buf, float *weights, int *idelays, int *indices, int *indptr
             )
{
    // nh is power of two, so x&(nh-1) is faster way to compute x%nh
    int nhm = nh - 1;

    // time for heun corrector step
    t += 1;
#pragma omp parallel for
    for (int i=0; i<nv; i++)
    {
    // compute coupling
#pragma omp simd
        for (int l=0; l<bs; l++)
            out[bs*i+l] = 0.0f;
        
        for (int j=indptr[i]; j<indptr[i+1]; j++)
        {
            float *b = buf + indices[j]*nh*bs;
            float w = weights[j];
            int roll_t = nh + t - idelays[j];
            float *b1 = b + (roll_t & nhm)*bs;

            
#pragma omp simd
            for (int l=0; l<bs; l++)
            {
                out[bs*i + l] += w * b1[l];
            }
        }
    }
}




void epi2d_dfun_batch( 
                int bs, int nv, 
                float *dxz, float *xz, float *lc, float *gc, float *x_0,
                float *k_lc, float *k_gc,
                float eps, float I)
{
    float *dx=dxz, *dz=dxz+nv*bs, *x=xz, *z=xz+nv*bs;
    #pragma omp simd
    for (int i=0; i<nv; i++)
    {
        #pragma omp simd
        for (int l=0; l<bs; l++){
            int idx = bs*i+l;
            dx[idx] = -x[idx]*x[idx]*x[idx] - 2 * x[idx]*x[idx] - z[idx] + I + k_lc[l] * lc[idx] + k_gc[l] * gc[idx];
            dz[idx] = eps * ( 4 * ( x[idx] - x_0[idx] ) - z[idx] );
        }
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

void mpr_dfun_batch( 
                int bs, int nv, float *drv, float *rv, float *cr, float *cv, float *eta,
                float Delta_o_pi_tau, float pi2tau2, float J, float tau, float I, float k
              )
{
    float *dr=drv, *dv=drv+nv*bs, *r=rv, *v=rv+nv*bs;
    #pragma omp simd
    for (int i=0; i<nv; i++)
    {
        #pragma omp simd
        for (int l=0; l<bs; l++){
            int idx = bs*i+l;
            dr[idx] = Delta_o_pi_tau + 2*v[idx]*r[idx];
            dv[idx] = v[idx]*v[idx] - pi2tau2*r[idx]*r[idx] + eta[idx] + J * tau * r[idx] + I + k*(cr[idx] + cv[idx]);
        }
    }
}
