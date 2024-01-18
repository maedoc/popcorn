// predictor stage for Heun, same as Euler
__kernel void heun_pred_determ(
    const float dt,
    __global float *out,
    __global float *x,
    __global float *dx
)
{
    int idx = get_global_id(0); // node id
    out[idx] = x[idx] + dt*dx[idx];
}

// corrector stage for Heun
__kernel void heun_corr_determ(
    const float dt,
    __global float *out,
    __global float *x,
    __global float *dx1,
    __global float *dx2
)
{
    int idx = get_global_id(0); // node id
    out[idx] = x[idx] + dt/2*(dx1[idx] + dx2[idx]);
}
