task void delays_task(
    const uniform int start,
    const uniform int chunksize, const uniform int nvtx, const uniform int nh, const uniform int t,
    uniform float out[], const uniform float buf[], const uniform float weights[], const uniform int idelays[], const uniform int indices[], const uniform int indptr[] )
{
    for (uniform int i=start; i<(start+chunksize); i++)
    {
        varying float acc = 0;
        foreach (j = indptr[i] ... indptr[i+1])
            acc += weights[j] * buf[(nh+t-idelays[j])*nvtx+indices[j]];
        out[i] = reduce_add(acc);
    }
}

export void delays(
    const uniform int chunksize, const uniform int nvtx, const uniform int nh, const uniform int t,
    uniform float out[], const uniform float buf[], const uniform float weights[], const uniform int idelays[], const uniform int indices[], const uniform int indptr[] )
{
    uniform int chunks = nvtx / chunksize;
    for (uniform int start=0; start<chunks; start++)
        launch delays_task(start*chunksize, chunksize,
                         nvtx,nh,t,out,buf,weights,idelays,indices,indptr);
    launch delays_task(chunks*chunksize, nvtx-chunks*chunksize,
                     nvtx,nh,t,out,buf,weights,idelays,indices,indptr);
    sync;
}
