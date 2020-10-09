#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "matrix_k_minimums_gpu.h"
#include "cuda_utils.h"


__global__ void matrix_k_minimums_kernel(int n, int m, float radius, int nsample,
                        const float *__restrict__ matrix,
                        int *__restrict__ idx){
    int batch_index = blockIdx.x;
    matrix += batch_index * n * m;
    idx += batch_index * n * nsample;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for(int i = index; i < n; i += stride) {
        for(int j = 0, cnt = 0; j < m && cnt < nsample; ++j) {
            float v = matrix[i * m + j];
            if (v < radius) {
                if (cnt == 0) {
                    for(int k = 0; k < nsample; ++k) {
                        idx[i * nsample + k] = j;
                    }
                }
                idx[i * nsample + cnt] = j;
                ++cnt;
            }
        }
    }
}

void matrix_k_minimums_kernel_wrapper(int b, int n, int m, float radius, int nsample, 
                     const float *matrix,
                     int *idx,
                     cudaStream_t stream) {

    cudaError_t err;
    matrix_k_minimums_kernel<<<b, opt_n_threads(n), 0, stream>>>(
    n, m, radius, nsample, matrix, idx);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
    }
}
