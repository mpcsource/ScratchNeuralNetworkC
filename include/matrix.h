#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float* data;
    int rows, cols;
} Matrix;

/* ----------- Access operations ----------- */
Matrix matrixCreate(int rows, int cols, float fill);
float matrixGet(Matrix* matrix, int row, int col);
void matrixSet(Matrix* matrix, int row, int col, float val);
Matrix matrixGetRow(Matrix* matrix, int row);
Matrix matrixCopy(Matrix* matrix);
void matrixDestroy(Matrix* matrix);
/* ------------ Math operations ------------ */
Matrix matrixAdd(Matrix* m1, Matrix* m2);
Matrix matrixSub(Matrix* m1, Matrix* m2);
Matrix matrixMul(Matrix* m1, Matrix* m2);
Matrix matrixMulf(Matrix* matrix, float value);
Matrix matrixDot(Matrix* m1, Matrix* m2);
Matrix matrixT(Matrix* matrix);