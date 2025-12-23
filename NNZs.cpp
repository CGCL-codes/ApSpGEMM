#include <iostream>
#include <vector>


std::vector<int> computeRowNNZ(const std::vector<int>& rowptr) {
    std::vector<int> nnz_per_row;
    for (size_t i = 0; i < rowptr.size() - 1; ++i) {
        nnz_per_row.push_back(rowptr[i + 1] - rowptr[i]);
    }
    return nnz_per_row;
}


std::vector<int> computeColNNZ(const std::vector<int>& colptr) {
    std::vector<int> nnz_per_col;
    for (size_t i = 0; i < colptr.size() - 1; ++i) {
        nnz_per_col.push_back(colptr[i + 1] - colptr[i]);
    }
    return nnz_per_col;
}


