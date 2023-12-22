
float local_reduce_add(
    const unsigned n,
    __local float *x
)
{
    const unsigned j = get_local_id(0);
    unsigned l = n;
    while (l > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l & 0x1) { // handle odd l
            if (j == 0)
                x[l - 2] += x[l - 1];
            l--;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        l >>= 1;
        if (j < l)
            x[j] += x[j + l];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return x[0];
}

__kernel void delays(
    const int nvtx, const int nh, const int t,
    __global float *out, __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, __global int *indptr)
{
    __local float acc[N];
    int i = get_global_id(0) / N;
    int li = get_local_id(0);
    int nchunks = (indptr[i+1] - indptr[i]) / N;
    acc[li] = 0.0;
    for (int j=indptr[i]+li; j<indptr[i+1]; j+=N) {
        acc[li] += weights[j] * buf[(nh+t-idelays[j])*nvtx+indices[j]];
    }
    acc[li] = local_reduce_add(N, acc);
    if (li==0)
        out[i] = acc[0];
}


float local_reduce_add_batch(
    const unsigned n,
    __local float *x
)
{
    const unsigned j = get_local_id(0);
    int bi = get_local_id(1);
    unsigned l = n;
    while (l > 1) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (l & 0x1) { // handle odd l
            if (j == 0)
                x[(l - 2)*B+bi] += x[(l - 1)*B+bi];
            l--;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        l >>= 1;
        if (j < l)
            x[j*B+bi] += x[(j + l)*B+bi];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return x[bi];
}

__kernel void delays_batched(
    const int nvtx, const int nh, const int t,
    __global float *out, __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, __global int *indptr)
{
    __local float acc[N*B];
    int i = get_global_id(0) / N;
    int li = get_local_id(0);
    int bi = get_local_id(1);
    
    int nchunks = (indptr[i+1] - indptr[i]) / N;
    acc[li*B+bi] = 0.0;
    for (int j=indptr[i]+li; j<indptr[i+1]; j+=N) {
        int roll_t = (nh + t - idelays[j]) % nh;
        acc[li*B+bi] += weights[j] * buf[(roll_t*nvtx+indices[j])*B+bi];
    }
    acc[li*B+bi] = local_reduce_add_batch(N, acc);
    if (li==0)
        out[i*B+bi] = acc[bi];
}

// nvidia driver borked, so there's a problem with this one
__kernel void delays_batched_simple(
    const int nvtx, const int nh, const int t,
    __global float *out, __global float *buf, __global float *weights,
    __global int *idelays, __global int *indices, __global int *indptr)
{
    int i = get_global_id(0);
    int bi = get_local_id(1);
    float acc = 0.0;
    for (int j=indptr[i]; j<indptr[i+1]; j++) {
        int tau = nh+t-idelays[j];
        acc += weights[j] * buf[(tau*nvtx + indices[j])*B+bi];
    }
    out[i*B + bi] = acc;
}
