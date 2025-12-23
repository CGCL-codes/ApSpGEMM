#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

typedef vector<vector<int>> Matrix;

struct RowPair {
    int rowA;
    int rowB;
    int sparsity;
};


int calculateSparsityMean(const vector<int>& rowA, const vector<int>& rowB) {
    return (count(rowA.begin(), rowA.end(), 0) + count(rowB.begin(), rowB.end(), 0)) / 2;
}


bool compareBySparsity(const RowPair& a, const RowPair& b) {
    return a.sparsity > b.sparsity;
}


void adaptivePanelAllocation(const Matrix& A, const Matrix& B, Matrix& C, double K) {
    int numRowsA = A.size();
    int numRowsB = B.size();

    vector<RowPair> rowPairs;


    for (int i = 0; i < numRowsA; ++i) {
        for (int j = 0; j < numRowsB; ++j) {
            RowPair rp;
            rp.rowA = i;
            rp.rowB = j;
            rp.sparsity = calculateSparsityMean(A[i], B[j]);
            rowPairs.push_back(rp);
        }
    }

  
    sort(rowPairs.begin(), rowPairs.end(), compareBySparsity);

 
    double R = K / (K + 1);
    int gpuThreshold = static_cast<int>(R * rowPairs.size());

    for (int G = 0; G < gpuThreshold; ++G) {
        int rowAIndex = rowPairs[G].rowA;
        int rowBIndex = rowPairs[G].rowB;


        for (int k = 0; k < A[rowAIndex].size(); ++k) {
            C[rowAIndex][k] += A[rowAIndex][k] * B[rowBIndex][k];
        }
    }


    for (int C_idx = gpuThreshold; C_idx < rowPairs.size(); ++C_idx) {
        int rowAIndex = rowPairs[C_idx].rowA;
        int rowBIndex = rowPairs[C_idx].rowB;


        for (int k = 0; k < A[rowAIndex].size(); ++k) {
            C[rowAIndex][k] += A[rowAIndex][k] * B[rowBIndex][k];
        }
    }

}

