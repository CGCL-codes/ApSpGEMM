#pragma once
#include <vector>
#include "stdio.h"
#include "Config.h"

namespace ApSpGEMM {
    struct ApSpGEMMConfig {
        int sm;
        int maxStaticSharedMemoryPerBlock;
        int maxDynamicSharedMemoryPerBlock;
        std::vector<CUstream> streams;
        cudaEvent_t completeStart = 0, completeEnd = 0, individualStart = 0, individualEnd = 0;

        static ApSpGEMMConfig initialize(int cudaDeviceNumber) {
			ApSpGEMMConfig config;
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, cudaDeviceNumber);
            config.sm = prop.multiProcessorCount;
            config.maxStaticSharedMemoryPerBlock = prop.sharedMemPerBlock;
            config.maxDynamicSharedMemoryPerBlock = std::max(prop.sharedMemPerBlockOptin, prop.sharedMemPerBlock);

            for (int i = 0; i < 6; i++) {
                config.streams.push_back(0);
                cudaStreamCreate(&config.streams[i]);
            }
            cudaEventCreate(&config.completeStart);
            cudaEventCreate(&config.completeEnd);
            cudaEventCreate(&config.individualStart);
            cudaEventCreate(&config.individualEnd);
            return config;
        }

        void cleanup() {
            for (auto s : streams) {
                cudaStreamDestroy(s);
            }
            cudaEventDestroy(completeStart);
            cudaEventDestroy(completeEnd);
            cudaEventDestroy(individualStart);
            cudaEventDestroy(individualEnd);
            streams.clear();
        }

        ~ApSpGEMMConfig() {
            // cleanup();
        }

    private:
		ApSpGEMMConfig() {

        }
    };
}


class ApSpGEMMKernels
{
public:
	ApSpGEMMKernels(uint32_t blockDim=128):
	blockDim{blockDim}
	{}

	void setLaunchDimensions(uint32_t _gridDim, cudaStream_t _stream = 0, uint32_t _blockDim = 128, uint32_t _sharedMem = 0)
	{
		gridDim = _gridDim;
		blockDim = _blockDim;
		stream = _stream;
		sharedMem = _sharedMem;
	}


	 template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_HashSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,

	template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalHashMap, class GlobalRowOffsetMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	void h_SpGEMMNumericLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC,
		GlobalHashMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, Config::SortModes sortColumns, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
		INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, bool setSortedBit, uint32_t rowsPerBlock);



	 template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_DenseSpGEMMNumeric(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, dCSRNoDealloc<VALUE_TYPE> matC, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
		INDEX_TYPE *rowMaxOperations, bool setSortedBit, uint32_t rowsPerBlock);


	template <typename INDEX_TYPE, typename VALUE_TYPE, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	void h_DenseSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, GlobalMap *maps, INDEX_TYPE mapCount,
		INDEX_TYPE *matCRowOffsets, INDEX_TYPE *blockStartRow, INDEX_TYPE *rowOperations, uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
		INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock);

	
	 template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_HashSpGEMMCount(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB, GlobalMap *maps, INDEX_TYPE mapCount, INDEX_TYPE *matCNnzRow,
		 INDEX_TYPE* rowOperations, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, const INDEX_TYPE* rowColMinMax,
		  INDEX_TYPE *rowMaxOperations, uint32_t *maxNnzPerRow, uint32_t rowsPerBlock);


	 template <typename INDEX_TYPE, typename VALUE_TYPE, unsigned MAX_ROWS_PER_BLOCK, class GlobalMap, class GlobalRowOffsetsMap, uint32_t SHARED_HASH_SIZE, bool SUPPORT_GLOBAL, uint32_t THREADS>
	 void h_SpGEMMCountLauncher(dCSRNoDealloc<VALUE_TYPE> matA, dCSRNoDealloc<VALUE_TYPE> matB,
								GlobalMap *hashMaps, INDEX_TYPE hashMapCount, GlobalRowOffsetsMap *rowOffsetMaps, INDEX_TYPE rowOffsetMapsCount,
								INDEX_TYPE *matCNnzRow, INDEX_TYPE *rowOperations, INDEX_TYPE *blockStartRow,
								uint32_t numberBlocks, const INDEX_TYPE *rowColMinMax,
								INDEX_TYPE *rowMaxOperations, uint32_t minimumDensity, INDEX_TYPE *maxNnzPerRow, uint32_t rowsPerBlock);


	 template <typename INDEX_TYPE, typename VALUE_TYPE, uint32_t THREADS, uint32_t entriesPerBlock>
	 void h_HashSpGEMMSorting(dCSRNoDealloc<VALUE_TYPE> matC, INDEX_TYPE *blockStartRow, uint32_t numberBlocks, bool bitShiftNumRows);

	 template <typename Map, typename INDEX_TYPE, typename VALUE_TYPE>
	 void h_InitializeGlobalMaps(Map *maps, int count, INDEX_TYPE *ids, VALUE_TYPE *values, size_t elementsPerMap);

	 template <typename Map, typename INDEX_TYPE>
	 void h_InitializeGlobalMapsNoVal(Map *maps, int count, INDEX_TYPE *ids, size_t elementsPerMap, uint32_t maxRowsPerBlock);


	 template <typename INDEX_TYPE, typename VALUE_TYPE, typename ROW_COUNT_TYPE, uint8_t KERNEL_COUNT>
	 void h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations(dCSRNoDealloc<VALUE_TYPE> &matA, dCSRNoDealloc<VALUE_TYPE> &matB, uint32_t *rowOperations,
		 INDEX_TYPE *blockStartRows, INDEX_TYPE *numBlockStarts, INDEX_TYPE (&h_numBlockStarts)[KERNEL_COUNT], INDEX_TYPE *blockStartRowsCombined,
		 uint32_t maxNnzPerBlock, uint32_t maxNnzPerBlockDynamicSharedMem, uint32_t maxRowsPerBlock, uint32_t actualKernelCount, uint32_t &h_rowsRequiringGlobal);

private:
	uint32_t blockDim;
	uint32_t gridDim;
	uint32_t sharedMem;
	cudaStream_t stream;
};

