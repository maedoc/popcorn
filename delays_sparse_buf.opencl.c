
__kernel void upbuf_heavi(
    const int nvtx, const int nsim, const int t, 
    const float theta, const int nh, 
    __global float *buf, __global float *state)
{
    int gi = get_global_id(0); 
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer

    // update buffer with current state
    int nhm = nh - 1;
    int idx = (t & nhm) * nsim * nvtx + gi;
    buf[idx] =  state[gi] > theta;
}

__kernel void delays1(
    const int nvtx, const int nsim, const int t, const int nh, 
    __global float *out1, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr)
{
    int gi = get_global_id(0); 
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer
    int nhm = nh - 1;

    // compute coupling
    float acc1 = 0.0f;
    for (int j=indptr[vtx_i]; j<indptr[vtx_i+1]; j++) {
        int roll_t = nh + t - idelays[j];
        tau1 = ((roll_t+0) & nhm) * nsim * nvtx + indices[j] * nsim + sim_i;
        acc1 += weights[j] * buf[tau1];
    }
    out1[gi] = acc1;
};


__kernel void delays1_heuncorr_noupbuf(
    const int nvtx, const int nsim, const int t, const int nh, 
    __global float *out1, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr)
{
    int gi = get_global_id(0); 
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer
    int nhm = nh - 1;
    // compute coupling
    int tp1 = t + 1; // we need the coupling for the heun corrector step so t+1, not to be used when updating the buffer
    float acc1 = 0.0f;
    for (int j=indptr[vtx_i]; j<indptr[vtx_i+1]; j++) {
        int roll_t = nh + tp1 - idelays[j];
        tau1 = ((roll_t+0) & nhm) * nsim * nvtx + indices[j] * nsim + sim_i;
        acc1 += weights[j] * buf[tau1];
    }
    out1[gi] = acc1;
};


__kernel void delays1_heuncorr_upbuf_heaviside(
    const int nvtx, const int nsim, const int t, 
    const float theta, const int nh, 
    __global float *out1,
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{
    int gi = get_global_id(0); 
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer

    // update buffer with current state
    int nhm = nh - 1;
    int idx = (t & nhm) * nsim * nvtx + gi;
    buf[idx] =  state[gi] > theta;

    // compute coupling
    int tp1 = t+1; // we need the coupling for the heun corrector step so t+1, not to be used when updating the buffer
    float acc1 = 0.0f;
    for (int j=indptr[vtx_i]; j<indptr[vtx_i+1]; j++) {
        int roll_t = nh + tp1 - idelays[j];
        tau1 = ((roll_t+0) & nhm) * nsim * nvtx + indices[j] * nsim + sim_i;
        acc1 += weights[j] * buf[tau1];
    }
    out1[gi] = acc1;
};


__kernel void delays2_upbuf_heaviside_tlocal(
    const int nvtx, const int nsim, const int t, const float theta,
    const int nh, 
    __global float *out1, __global float *out2, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{
    int gi = get_global_id(0); 
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer
    int tau2 = 0;              // index2 into buffer

    // update buffer with current state
    int nhm = nh - 1;
    int idx = (t & nhm) * nsim * nvtx + gi;
    buf[idx] =  state[gi] > theta;

    // compute coupling
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j=indptr[vtx_i]; j<indptr[vtx_i+1]; j++) {
        int roll_t = nh + t - idelays[j];
        tau1 = ((roll_t+0) & nhm) * nsim * nvtx + indices[j] * nsim + sim_i;
        tau2 = ((roll_t+1) & nhm) * nsim * nvtx + indices[j] * nsim + sim_i;
        acc1 += weights[j] * buf[tau1];
        acc2 += weights[j] * buf[tau2];
    }
    out1[gi] = acc1;
    out2[gi] = acc2;
};



__kernel void delays2_upbuf_heaviside_simlocal(
    const int nsim, const int t, const float theta,
    const int nh,
    __global float *out1, __global float *out2, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{
    int gi = get_global_id(0); //* get_global_size(1) + get_global_id(1);
    int sim_i = gi % nsim;     // sim id for a single vertex
    int vtx_i = gi / nsim;     // vertex id within connectome
    int tau1 = 0;              // index1 into buffer
    int tau2 = 0;              // index2 into buffer

    // update buffer with current state
    int nhm = nh - 1;
    int idx = nh*gi + (t & nhm);
    buf[idx] =  state[gi] > theta;

    // compute coupling
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j=indptr[vtx_i]; j<indptr[vtx_i+1]; j++) {
        int roll_t = nh + t - idelays[j];
        tau1 = (indices[j] * nsim + sim_i) * nh +  ((roll_t+0) & nhm);
        tau2 = (indices[j] * nsim + sim_i) * nh +  ((roll_t+1) & nhm);
        acc1 += weights[j] * buf[tau1];
        acc2 += weights[j] * buf[tau2];
    }
    out1[gi] = acc1;
    out2[gi] = acc2;
};


__kernel void delays2_upbuf_heaviside(
    const int nvtx, const int t, const float theta,
    const int nh,
    __global float *out1, __global float *out2, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{

    int vtx_gi = get_global_id(0); //* get_global_size(1) + get_global_id(1);
    // int v tx_gi = get_global_id(0); // vertex global id
    int vtx_li = vtx_gi % nvtx;    // vertex local id within one simulation
    int sim_i  = vtx_gi / nvtx;    // simulation id
    int tau1 = 0;                   // index1 into buffer
    int tau2 = 0;                   // index2 into buffer

    // update buffer with current state
    int nhm = nh - 1;
    int idx = nh*vtx_gi + (t & nhm);
    buf[idx] =  state[vtx_gi] > theta;

    // compute coupling
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    for (int j=indptr[vtx_li]; j<indptr[vtx_li+1]; j++) {
        int roll_t = nh + t - idelays[j];
        tau1 = indices[j] * nh + sim_i * nvtx * nh +  ((roll_t+0) & nhm);
        tau2 = indices[j] * nh + sim_i * nvtx * nh +  ((roll_t+1) & nhm);
        acc1 += weights[j] * buf[tau1];
        acc2 += weights[j] * buf[tau2];
    }
    out1[vtx_gi] = acc1;
    out2[vtx_gi] = acc2;
};

__kernel void delays2_sparse_upbuf_heaviside(
    const int nvtx, const int t, const float theta,
    __global int *nh_per_vtx, __global int *buf_indptr,
    __global float *out1, __global float *out2, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{

    int vtx_gi = get_global_id(0); // vertex global id
    int vtx_li = vtx_gi % nvtx;    // vertex local id within one simulation
    int sim_i  = vtx_gi / nvtx;    // simulation id
    int tau1 = 0;                   // index into sparse history buffer
    int tau2 = 0;                   // index into sparse history buffer

    // update buffer with current state using heaviside step function
    int idx = buf_indptr[vtx_gi] + (nh_per_vtx[vtx_li]+t) % nh_per_vtx[vtx_li];
    buf[idx] =  state[vtx_gi] > theta;

    // compute coupling
    out1[vtx_gi] = 0.0;
    out2[vtx_gi] = 0.0;
    for (int j=indptr[vtx_li]; j<indptr[vtx_li+1]; j++) {
        int nh  = nh_per_vtx[indices[j]];
        int nhm = nh - 1;
        int roll_t = nh + t - idelays[j];
        tau1 = buf_indptr[indices[j] + sim_i * nvtx] +  ((roll_t+0) & nhm);
        tau2 = buf_indptr[indices[j] + sim_i * nvtx] +  ((roll_t+1) & nhm);
        out1[vtx_gi] += weights[j] * buf[tau1];
        out2[vtx_gi] += weights[j] * buf[tau2];
    }
};

__kernel void delays2_sparse_upbuf(
    const int nvtx, const int t,
    __global int *nh_per_vtx, __global int *buf_indptr,
    __global float *out1, __global float *out2, 
    __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, 
    __global int *indptr, __global float *state)
{

    int vtx_gi = get_global_id(0); // vertex global id
    int vtx_li = vtx_gi % nvtx;    // vertex local id within one simulation
    int sim_i  = vtx_gi / nvtx;    // simulation id
    int tau1 = 0;                   // index into sparse history buffer
    int tau2 = 0;                   // index into sparse history buffer

    // update buffer with current state using heaviside step function
    int idx = buf_indptr[vtx_gi] + (nh_per_vtx[vtx_li]+t) % nh_per_vtx[vtx_li];
    buf[idx] =  state[vtx_gi];

    // compute coupling
    out1[vtx_gi] = 0.0;
    out2[vtx_gi] = 0.0;
    for (int j=indptr[vtx_li]; j<indptr[vtx_li+1]; j++) {
        int nh  = nh_per_vtx[indices[j]];
        int nhm = nh - 1;
        int roll_t = nh + t - idelays[j];
        tau1 = buf_indptr[indices[j] + sim_i * nvtx] +  ((roll_t+0) & nhm);
        tau2 = buf_indptr[indices[j] + sim_i * nvtx] +  ((roll_t+1) & nhm);
        out1[vtx_gi] += weights[j] * buf[tau1];
        out2[vtx_gi] += weights[j] * buf[tau2];
    }
};