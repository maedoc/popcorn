task void spmv_task(
    const uniform int start,
    const uniform int chunksize,
    const uniform int num_lanes,
    uniform float out[],
    const uniform float vec[],
    const uniform float data[],
    const uniform int indices[],
    const uniform int indptr[],
    const uniform bool use_simd
    )
{
    for (uniform int i=start; i<(start+chunksize); i++)
    {
	// manual unrolling for 1, 2, 4 lanes
	if (use_simd && num_lanes == 1)
	{
		varying float acc = 0;
		foreach (j = indptr[i] ... indptr[i+1])
			acc += data[j] * vec[indices[j]];
		out[i] = reduce_add(acc);
	}
	else if (use_simd && num_lanes == 2)
	{
		varying float acc1 = 0;
		varying float acc2 = 0;
		foreach (j = indptr[i] ... indptr[i+1])
		{
			acc1 += data[j] * vec[indices[j]*2];
			acc2 += data[j] * vec[indices[j]*2+1];
		}
		out[i*2] = reduce_add(acc1);
		out[i*2+1] = reduce_add(acc2);
	}
	else if (use_simd && num_lanes == 4)
	{
		varying float acc1 = 0;
		varying float acc2 = 0;
		varying float acc3 = 0;
		varying float acc4 = 0;
		foreach (j = indptr[i] ... indptr[i+1])
		{
			acc1 += data[j] * vec[indices[j]*4];
			acc2 += data[j] * vec[indices[j]*4+1];
			acc3 += data[j] * vec[indices[j]*4+2];
			acc4 += data[j] * vec[indices[j]*4+3];
		}
		out[i*4] = reduce_add(acc1);
		out[i*4+1] = reduce_add(acc2);
		out[i*4+2] = reduce_add(acc3);
		out[i*4+3] = reduce_add(acc4);
	}
	else if (use_simd && num_lanes == 8)
	{
		varying float acc1 = 0;
		varying float acc2 = 0;
		varying float acc3 = 0;
		varying float acc4 = 0;
		varying float acc5 = 0;
		varying float acc6 = 0;
		varying float acc7 = 0;
		varying float acc8 = 0;
		foreach (j = indptr[i] ... indptr[i+1])
		{
			acc1 += data[j] * vec[indices[j]*8];
			acc2 += data[j] * vec[indices[j]*8+1];
			acc3 += data[j] * vec[indices[j]*8+2];
			acc4 += data[j] * vec[indices[j]*8+3];
			acc5 += data[j] * vec[indices[j]*8+4];
			acc6 += data[j] * vec[indices[j]*8+5];
			acc7 += data[j] * vec[indices[j]*8+6];
			acc8 += data[j] * vec[indices[j]*8+7];
		}
		out[i*8] = reduce_add(acc1);
		out[i*8+1] = reduce_add(acc2);
		out[i*8+2] = reduce_add(acc3);
		out[i*8+3] = reduce_add(acc4);
		out[i*8+4] = reduce_add(acc5);
		out[i*8+5] = reduce_add(acc6);
		out[i*8+6] = reduce_add(acc7);
		out[i*8+7] = reduce_add(acc8);
	}
	else
	{
		foreach (l = 0 ... num_lanes)
		{
		    varying float acc = 0;
		    for (int j=indptr[i]; j<indptr[i+1]; j++)
			acc += data[j] * vec[indices[j]*num_lanes + l];
		    out[i*num_lanes + l] = acc;
		}
	}
        
    }
}

export void spmv(
    const uniform int num_nodes,
    const uniform int num_lanes,
    uniform float out[],
    const uniform float vec[],
    const uniform float data[],
    const uniform int indices[],
    const uniform int indptr[],
    const uniform int chunksize,
    const uniform bool use_simd
    )
{
    uniform int chunks = num_nodes / chunksize;
    for (uniform int start=0; start<chunks; start++)
        launch spmv_task(start*chunksize, chunksize,num_lanes,out,vec,data,indices,indptr,use_simd);
    launch spmv_task(chunks*chunksize, num_nodes-chunks*chunksize,num_lanes,out,vec,data,indices,indptr,use_simd);
    sync;
}
