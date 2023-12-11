extern "C" __global__
void spmv_csr_kernel(float* matrixVal, int* matRowPtrs, int* matColIdx, const size_t nRow, float* x, float* outY) {
    unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nRow) {
        float sum = 0.0f;
        for (unsigned int i = matRowPtrs[row]; i < matRowPtrs[row + 1]; i++) {
            sum += matrixVal[i] * x[matColIdx[i]];
        }
        outY[row] += sum;
    }
}