
#pragma once
#include "Timings.h"
#include "spECKConfig.h"

// REPLACE THESE VALUES WITH YOUR ACTUAL DEVICE SPECIFICATIONS

static constexpr int ApSpGEMM_STATIC_MEM_PER_BLOCK {49152};
static constexpr int ApSpGEMM_DYNAMIC_MEM_PER_BLOCK{49152};

namespace ApSpGEMM
{
    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyApSpGEMM(const dCSR<DataType> &A, const dCSR<DataType> &B, dCSR<DataType> &matOut, ApSpGEMMConfig &config, Timings &timings);

    template <typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
    void MultiplyApSpGEMMImplementation(const dCSR<DataType> &A, const dCSR<DataType> &B, dCSR<DataType> &matOut, ApSpGEMMConfig &config, Timings &timings = Timings());
}
