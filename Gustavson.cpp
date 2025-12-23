#include <iostream>
#include <vector>
#include <unordered_map>

// Function to perform sparse matrix multiplication using Gustavson's algorithm
void GustavsonSpGEMM(const std::vector<int>& Ap, const std::vector<int>& Aj, const std::vector<float>& Ax,
                     const std::vector<int>& Bp, const std::vector<int>& Bj, const std::vector<float>& Bx,
                     std::vector<int>& Cp, std::vector<int>& Cj, std::vector<float>& Cx) {
    int rowsA = Ap.size() - 1; // Number of rows in A
    Cp.resize(rowsA + 1, 0);

    // Temporary storage for results in a row
    std::vector<std::unordered_map<int, float>> rowResults(rowsA);

    // Perform multiplication row by row
    for (int i = 0; i < rowsA; ++i) {
        for (int k = Ap[i]; k < Ap[i + 1]; ++k) {
            int colA = Aj[k];
            float valA = Ax[k];

            for (int l = Bp[colA]; l < Bp[colA + 1]; ++l) {
                int colB = Bj[l];
                float valB = Bx[l];

                // Accumulate the result for C[i, colB]
                rowResults[i][colB] += valA * valB;
            }
        }

        // Populate Cp for the row
        Cp[i + 1] = Cp[i] + rowResults[i].size();
    }

    // Flatten the results into CSR format
    Cj.reserve(Cp[rowsA]);
    Cx.reserve(Cp[rowsA]);

    for (int i = 0; i < rowsA; ++i) {
        for (const auto& [col, value] : rowResults[i]) {
            Cj.push_back(col);
            Cx.push_back(value);
        }
    }
}

int main() {
    // Example sparse matrices A and B in CSR format
    std::vector<int> Ap = {0, 2, 4};
    std::vector<int> Aj = {0, 1, 1, 2};
    std::vector<float> Ax = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<int> Bp = {0, 2, 3};
    std::vector<int> Bj = {0, 1, 2};
    std::vector<float> Bx = {5.0f, 6.0f, 7.0f};

    // Result matrix C in CSR format
    std::vector<int> Cp;
    std::vector<int> Cj;
    std::vector<float> Cx;

    // Perform SpGEMM
    GustavsonSpGEMM(Ap, Aj, Ax, Bp, Bj, Bx, Cp, Cj, Cx);

    // Output the resulting matrix C
    std::cout << "Cp: ";
    for (int v : Cp) std::cout << v << " ";
    std::cout << "\nCj: ";
    for (int v : Cj) std::cout << v << " ";
    std::cout << "\nCx: ";
    for (float v : Cx) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
