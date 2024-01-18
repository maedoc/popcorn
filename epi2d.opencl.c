__kernel void epi2d_dfun(
    __global float *dx, __global float *dz,
    __global float *x, __global float *z,
    __global float *x_0, 
    __global float *k_lc, __global float *lc,
    __global float *k_gc, __global float *gc,
    const float eps, const float I)
{
    int idx = get_global_id(0); // vertex global id
    dx[idx] = -x[idx]*x[idx]*x[idx] - 2 * x[idx]*x[idx] - z[idx] + I + k_lc[idx] * lc[idx] + k_gc[idx] * gc[idx];
    dz[idx] = eps * ( 4 * ( x[idx] - x_0[idx] ) - z[idx] );   
};
