#ifndef _BALL_QUERY_GPU
#define _BALL_QUERY_GPU

#ifdef __cplusplus
extern "C" {
#endif

void matrix_k_minimums_kernel_wrapper(int b, int n, int m, float radius,
                     int nsample, const float *matrix,
                     int *idx,
                     cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
