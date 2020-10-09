#include <THC/THC.h>

#include "matrix_k_minimums_gpu.h"

extern THCState *state;

int matrix_k_minimums_wrapper(int b, int n, int m, float radius, int nsample,
               THCudaTensor *matrix_tensor,
               THCudaIntTensor *idx_tensor) {

    const float *matrix = THCudaTensor_data(state, matrix_tensor);
    int *idx = THCudaIntTensor_data(state, idx_tensor);

    cudaStream_t stream = THCState_getCurrentStream(state);

    matrix_k_minimums_kernel_wrapper(b, n, m, radius, nsample, matrix, idx,
                    stream);
    return 1;
}
