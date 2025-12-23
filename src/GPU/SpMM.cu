#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void spmm_kernel(CSRMatrix A, DenseMatrix B, DenseMatrix Y, int M, int K, int S) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < A.rows) {
        int row_start = A.row_offsets[row];
        int row_end = A.row_offsets[row + 1];


        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0;


            for (int i = row_start; i < row_end; ++i) {
                int col = A.col_indices[i];  

                if (A.cols>900) {
                    __shared__ float shared_B[/* shared memory size */];
                    if (threadIdx.x == 0) {
                        for (int k = 0; k < B.cols; ++k) {
                            shared_B[k] = B.values[col * B.cols + k];  
                        }
                    }
                    __syncthreads();  
                    sum += a_value * shared_B[j];
                } else {

                    sum += a_value * B.values[col * B.cols + j];
                }
            }

            Y.values[row * Y.cols + j] = sum;
        }
    }
}



void SpMM(CSRMatrix &A, DenseMatrix &B, DenseMatrix &Y, int M, int K, int S) {

    dim3 threadsPerBlock(M);
    dim3 blocksPerGrid((A.rows + M - 1) / M);


    spmm_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, Y, M, K, S);
}

int main() {

    A B, Y;

    int M = 8; 
    int K = cols_B;  
    int S = 1024;  
    SpMM(A, B, Y, M, K, S);


    std::cout << "Result matrix Y:" << std::endl;
    for (int i = 0; i < Y.rows; ++i) {
        for (int j = 0; j < Y.cols; ++j) {
            std::cout << "Y[" << i << "][" << j << "] = " << Y.values[i * Y.cols + j] << std::endl;
        }
    }

    // 释放内存
    delete[] A.row_offsets;
    delete[] A.col_indices;
    delete[] A.values;
    delete[] B.values;
    delete[] Y.values;

    return 0;
}
