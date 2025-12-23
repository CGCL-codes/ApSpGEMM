#include <stdio.h>
#include <stdlib.h>

void add(int n, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] + B[i][j];
}

void sub(int n, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = A[i][j] - B[i][j];
}

void normal_mul(int n, int **A, int **B, int **C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
}

int **alloc_matrix(int n) {
    int **M = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        M[i] = (int *)malloc(n * sizeof(int));
    return M;
}

void free_matrix(int n, int **M) {
    for (int i = 0; i < n; i++)
        free(M[i]);
    free(M);
}



void strassen(int n, int **A, int **B, int **C) {
    if (n <= 2) {
        normal_mul(n, A, B, C);
        return;
    }

    int k = n / 2;


    int **A11 = alloc_matrix(k);
    int **A12 = alloc_matrix(k);
    int **A21 = alloc_matrix(k);
    int **A22 = alloc_matrix(k);
    int **B11 = alloc_matrix(k);
    int **B12 = alloc_matrix(k);
    int **B21 = alloc_matrix(k);
    int **B22 = alloc_matrix(k);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + k];
            A21[i][j] = A[i + k][j];
            A22[i][j] = A[i + k][j + k];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + k];
            B21[i][j] = B[i + k][j];
            B22[i][j] = B[i + k][j + k];
        }

 
    int **M1 = alloc_matrix(k);
    int **M2 = alloc_matrix(k);
    int **M3 = alloc_matrix(k);
    int **M4 = alloc_matrix(k);
    int **M5 = alloc_matrix(k);
    int **M6 = alloc_matrix(k);
    int **M7 = alloc_matrix(k);

    int **T1 = alloc_matrix(k);
    int **T2 = alloc_matrix(k);

 
    add(k, A11, A22, T1);
    add(k, B11, B22, T2);
    strassen(k, T1, T2, M1);


    add(k, A21, A22, T1);
    strassen(k, T1, B11, M2);

    sub(k, B12, B22, T2);
    strassen(k, A11, T2, M3);


    sub(k, B21, B11, T2);
    strassen(k, A22, T2, M4);


    add(k, A11, A12, T1);
    strassen(k, T1, B22, M5);

 
    sub(k, A21, A11, T1);
    add(k, B11, B12, T2);
    strassen(k, T1, T2, M6);


    sub(k, A12, A22, T1);
    add(k, B21, B22, T2);
    strassen(k, T1, T2, M7);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            C[i][j] = M1[i][j] + M4[i][j] - M5[i][j] + M7[i][j];
            C[i][j + k] = M3[i][j] + M5[i][j];
            C[i + k][j] = M2[i][j] + M4[i][j];
            C[i + k][j + k] = M1[i][j] - M2[i][j] + M3[i][j] + M6[i][j];
        }

    
    free_matrix(k, A11); free_matrix(k, A12);
    free_matrix(k, A21); free_matrix(k, A22);
    free_matrix(k, B11); free_matrix(k, B12);
    free_matrix(k, B21); free_matrix(k, B22);

    free_matrix(k, M1); free_matrix(k, M2);
    free_matrix(k, M3); free_matrix(k, M4);
    free_matrix(k, M5); free_matrix(k, M6);
    free_matrix(k, M7);

    free_matrix(k, T1); free_matrix(k, T2);
}
