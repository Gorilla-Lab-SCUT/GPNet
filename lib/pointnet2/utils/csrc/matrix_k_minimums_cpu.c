#include <TH/TH.h>
#include <math.h>

int matrix_k_minimums_cpu(int b, int n, int m, float radius, int nsample,
               THFloatTensor *matrix, 
               THIntTensor *idx)
{
    float * matrix_flat = THFloatTensor_data(matrix);
    int * idx_flat = THIntTensor_data(idx);
    for (int i = 0; i < b; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0, cnt = 0; k < m && cnt < nsample; ++k)
            {
                float v = matrix_flat[i * n * m + j * m + k];
                if (v < radius)
                {
                    if (cnt == 0)
                    {
                        for (int l = 0; l < nsample; ++l)
                        {
                            idx_flat[i * n * nsample + j * nsample + l] = k;
                        }
                    }
                    idx_flat[i * n * nsample + j * nsample + cnt] = k;
                    ++cnt;
                }
            }
        }
    }
    return 1;
}




