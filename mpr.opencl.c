#define pi 3.141592653589793f

// assume constant for now
#define Delta 2.0f
#define tau 1.0f
#define J 15.0f
#define I 1.0f

// but they can be passed in as parameters like so
#define eta (params[i*L + l])

__kernel void mpr_dfun(
    __global float *g_dr,
    __global float *g_dV,
    __global float *g_r,
    __global float *g_V,
    __global float *g_cr,
    __global float *g_cV,
    __global float *params
)
{
    int i = get_global_id(0); // node id
    int N = get_global_size(0); // num nodes
    int l = get_global_id(1); // lane / gpu thread
    int L = get_global_size(1); // num lanes

    float r = g_r[i*L + l];
    float V = g_V[i*L + l];
    float cr = g_cr[i*L + l];
    float cV = g_cV[i*L + l];
    
    float dr = Delta / (pi * tau) + 2*V*r;
    float dV = V*V - pi*pi * tau*tau * r*r + eta + J * tau * r + I + cr + cV;

    g_dr[i*L + l] = dr / tau;
    g_dV[i*L + l] = dV / tau;
}
