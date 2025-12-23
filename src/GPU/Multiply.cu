#include <bitset>
#include <memory>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "Multiply.h"
#include "GPU/Configuration.h"
#include "meta_utils.h"
#include <thrust/scan.h>
#include <thrust/extrema.h>
#include "Config.h"
#include "common.h"

using IndexType = uint32_t;

namespace ApSpGEMM
{

template <typename T>
__host__ __forceinline__ T divup(T a, T b)
{
    return (a + b - 1) / b;
}

void startTimerVar(cudaEvent_t &start, CUstream stream = 0)
{
    HANDLE_ERROR(cudaEventRecord(start, stream));
    HANDLE_ERROR(cudaEventSynchronize(start));
}

float recordTimerVar(cudaEvent_t &start, cudaEvent_t &end, CUstream stream = 0)
{
    float time;
    HANDLE_ERROR(cudaEventRecord(end, stream));
    HANDLE_ERROR(cudaEventSynchronize(end));
    HANDLE_ERROR(cudaEventElapsedTime(&time, start, end));
    return time;
}

template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
void MultiplyApSpGEMMImplementation(const dCSR<DataType> &matA_Dealloc, const dCSR<DataType> &matB_Dealloc, dCSR<DataType> &matOut, ApSpGEMMConfig &config, Timings &timings)
{
    dCSRNoDealloc<DataType> matA(matA_Dealloc), matB(matB_Dealloc);

    if (matB.cols > 1 << 27)
    {
        printf("ERROR: matrix B has more than %d columns (%lu)\n", 1 << 27, matB.cols);
        return;
    }
    if (matA.rows > 1 << 27)
    {
        printf("ERROR: matrix A has more than %d rows (%lu)\n", 1 << 27, matB.rows);
        return;
    }
    if (matA.nnz * matB.nnz == 0) {
        matOut.nnz = 0;
        return;
    }

    if (MAX_DYNAMIC_SHARED != config.maxDynamicSharedMemoryPerBlock || MAX_STATIC_SHARED != config.maxStaticSharedMemoryPerBlock) {
        if (MAX_DYNAMIC_SHARED > config.maxDynamicSharedMemoryPerBlock) {
            printf("ERROR: ApSpGEMM was compiled with %d maximum dynamic shared memory, but device limit is %d.\n",
                MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
            return;
        } else {
            printf("WARNING: ApSpGEMM was compiled with %d maximum dynamic shared memory, but device limit is %d.\n",
                MAX_DYNAMIC_SHARED, config.maxDynamicSharedMemoryPerBlock);
        }
        if (MAX_STATIC_SHARED > MAX_DYNAMIC_SHARED)
        {
            printf("ERROR: ApSpGEMM was compiled with smaller dynamic than static shared memory.\n");
            return;
        }
        if (MAX_STATIC_SHARED > config.maxStaticSharedMemoryPerBlock)
        {
            printf("ERROR: ApSpGEMM was compiled with %d maximum static shared memory, but device limit is %d.\n",
                MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
            return;
        }
        else if (MAX_STATIC_SHARED < config.maxStaticSharedMemoryPerBlock) {
            printf("WARNING: ApSpGEMM was compiled with %d maximum static shared memory, but device limit is %d.\n",
                MAX_STATIC_SHARED, config.maxStaticSharedMemoryPerBlock);
        }
    }

    ApSpGEMMKernels spgemm(1024);

    const int kernelCountNumeric = 6;
    const int kernelCountCounting = 6;
    const int maxRowsPerBlock = 32;
    const int warpsCounting = THREADS_PER_BLOCK / 32;
    const int warpsNumeric = THREADS_PER_BLOCK / 32;
    const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;

    const int sharedBytesPerWarpCounting = MAX_STATIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting;
    const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(IndexType);
    const int sharedBytesPerBlockCounting = sharedBytesPerWarpCounting * warpsCounting;

    const int dynamicSharedBytesPerWarpCounting = MAX_DYNAMIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting;
    const int dynamicEntriesPerWarpCounting = dynamicSharedBytesPerWarpCounting / sizeof(IndexType);
    const int dynamicSharedBytesPerBlockCounting = dynamicSharedBytesPerWarpCounting * warpsCounting;

    const int sharedBytesPerWarpNumeric = MAX_STATIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric;
    const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
    const int sharedBytesPerBlockNumeric = sharedBytesPerWarpNumeric * warpsNumeric;

    const int dynamicSharedBytesPerWarpNumeric = MAX_DYNAMIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric;
    const int dynamicEntriesPerWarpNumeric = dynamicSharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
    const int dynamicSharedBytesPerBlockNumeric = dynamicSharedBytesPerWarpNumeric * warpsNumeric;

    assert(kernelCountCounting <= kernelCountNumeric);

    bool supportGlobalFallback = true;
    const uint32_t minimumDensityForDenseModeCounting = 999;
    const uint32_t denseModeRowThresholdInternalSorting = 999;
    const uint32_t denseModeRowThresholdExternalSorting = 18;
    const uint32_t sm = config.sm;
    const ui
