#include "matrix.h"

/* ----------- Access operations ----------- */

Matrix matrixCreate(int rows, int cols, float fill) {
    Matrix matrix;

    matrix.data = (float*)calloc(rows*cols, sizeof(float));
    matrix.rows = rows;
    matrix.cols = cols;

    for(int i = 0; i < rows*cols; i++)
        matrix.data[i] = fill;

    return matrix;
}

float matrixGet(Matrix* matrix, int row, int col) {
    return matrix->data[row * matrix->cols + col];
}

void matrixSet(Matrix* matrix, int row, int col, float val) {
    matrix->data[row * matrix->cols + col] = val;
}

Matrix matrixGetRow(Matrix* matrix, int row) {
    Matrix result = matrixCreate(1, matrix->cols, 0);

    for(int i = 0; i < matrix->cols; i++) {
        matrixSet(&result, 0, i, matrixGet(matrix, row, i));
    }

    return result;
}

Matrix matrixCopy(Matrix* matrix) {
    Matrix result = matrixCreate(matrix->rows, matrix->cols, 0);
    for(int i = 0; i < matrix->rows*matrix->cols; i++)
        result.data[i] = matrix->data[i];
    return result;
}

void matrixDestroy(Matrix* matrix) {
    free(matrix->data);
}

/* ------------ Math operations ------------ */

Matrix matrixAdd(Matrix* m1, Matrix* m2) {
    if((m1->cols != m2->cols) || (m1->rows != m2->rows)) {
        fprintf(stderr, "Matrix shape mismatch in matrixAdd\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrixCreate(m1->rows, m1->cols, 0);

    for (int i = 0; i < m1->rows; i++)
        for (int j = 0; j < m1->cols; j++)
            matrixSet(&result, i, j, matrixGet(m1,i,j)+matrixGet(m2, i,j));

    return result;
}

Matrix matrixSub(Matrix* m1, Matrix* m2) {
    if((m1->cols != m2->cols) || (m1->rows != m2->rows)) {
        fprintf(stderr, "Matrix shape mismatch in matrixSub\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrixCreate(m1->rows, m1->cols, 0);

    for (int i = 0; i < m1->rows; i++)
        for (int j = 0; j < m1->cols; j++)
            matrixSet(&result, i, j, matrixGet(m1,i,j)-matrixGet(m2, i,j));

    return result;
}

Matrix matrixMul(Matrix* m1, Matrix* m2) {
    if((m1->cols != m2->cols) || (m1->rows != m2->rows)) {
        fprintf(stderr, "Matrix shape mismatch in matrixMul\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrixCreate(m1->rows, m1->cols, 0);

    for (int i = 0; i < m1->rows; i++)
        for (int j = 0; j < m1->cols; j++)
            matrixSet(&result, i, j, matrixGet(m1,i,j)*matrixGet(m2, i,j));

    return result;
}

Matrix matrixMulf(Matrix* matrix, float value) {
    Matrix result = matrixCreate(matrix->rows, matrix->cols, 0);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrixSet(&result, i, j, matrixGet(matrix,i,j)*value);

    return result;
}

Matrix matrixDot(Matrix* m1, Matrix* m2) {
    if(m1->cols != m2->rows) {
        printf("m1: %dx%d, m2: %dx%d\n", m1->rows, m1->cols, m2->rows, m2->cols);
        fprintf(stderr, "Matrix shape mismatch in matrixDot\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrixCreate(m1->rows, m2->cols, 0);

    for(int i = 0; i < m1->rows; i++) {
        for(int j = 0; j < m2->cols; j++) {
            float ij = 0;
            for(int k = 0; k < m1->cols; k++)
                ij += matrixGet(m1, i, k) * matrixGet(m2, k, j);
            matrixSet(&result, i, j, ij);
        }
    }

    return result;
}

Matrix matrixT(Matrix* matrix) {
    Matrix result = matrixCreate(matrix->cols, matrix->rows, 0);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrixSet(&result, j, i, matrixGet(matrix, i, j));

    return result;
}