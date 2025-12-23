#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

__global__ void AdjustThreadAllocation(int* M, int NNZ_max, int nrows, int* G, int NNZA) {
    int iter_thread = NNZ_max / (*M);

    while (iter_thread > 2 * nrows || iter_thread < 2 * nrows) {
        if (iter_thread > 2 * nrows) {
            *M = static_cast<int>(*M * (static_cast<float>(nrows) / (2 * iter_thread)));
        } else if (iter_thread < 2 * nrows) {
            *M = static_cast<int>(*M * (static_cast<float>(iter_thread) / (2 * nrows)));
        }

        iter_thread = NNZ_max / (*M);

        if (*G > NNZA) {
            *G = NNZA;
        }
    }
}


#define THREADS_PER_BLOCK 256

// Kernel for dynamic binning and SpGEMM computation
__global__ void SpGEMM_DynamicBinning(const int* Ap, const int* Aj, const float* Ax,
                                      const int* Bp, const int* Bj, const float* Bx,
                                      int* Cp, int* Cj, float* Cx, int numRowsA,
                                      int sharedMemorySize) {
    // Thread block ID and thread ID within a block
    int blockId = blockIdx.x;
    int threadId = threadIdx.x;

    extern __shared__ int sharedMem[]; // Shared memory for dynamic binning

    // Each thread block processes a bin of rows
    int startRow = blockId * sharedMemorySize;
    int endRow = min((blockId + 1) * sharedMemorySize, numRowsA);

    for (int row = startRow; row < endRow; ++row) {
        if (threadId == 0) {
            sharedMem[row - startRow] = 0; // Initialize shared memory for row
        }
        __syncthreads();

        for (int i = Ap[row]; i < Ap[row + 1]; ++i) {
            int colA = Aj[i];
            float valA = Ax[i];

            for (int j = Bp[colA]; j < Bp[colA + 1]; ++j) {
                int colB = Bj[j];
                float valB = Bx[j];
                atomicAdd(&sharedMem[row - startRow], valA * valB);
            }
        }
        __syncthreads();

        if (threadId == 0) {
            Cp[row + 1] = sharedMem[row - startRow];
        }
    }
}


