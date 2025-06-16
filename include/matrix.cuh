#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define HOST_AND_DEVICE   1 // Allocate both on GPU and CPU
#define DEVICE_ONLY       0 // Allocate on GPU only

#define EMPTY_MATRIX_STATE -1
#define EMPTY_MATRIX ((Matrix){ .rows = EMPTY_MATRIX_STATE, .cols = EMPTY_MATRIX_STATE, .data = NULL, .d_data = NULL, .owns_data = GPU_ONLY })

#ifdef __cplusplus
extern "C" {
#endif


typedef struct {
    float* data;              // Host data
    float* d_data;            // Device data
    int rows, cols;           // Amount of rows and cols
    int allocation_type;      // HOST_AND_DEVICE or DEVICE_ONLY
} Matrix;

static cublasHandle_t cublasHandle = NULL;
void initCublas();
void destroyCublas();



/* ----------- Access operations ----------- */
Matrix matrixCreate(int rows, int cols, float fill, int allocation_flag);
void matrixUpload(Matrix* matrix); // From CPU to GPU
void matrixDownload(Matrix* matrix); // From GPU to CPU
float matrixGet(Matrix* matrix, int row, int col);
void matrixSet(Matrix* matrix, int row, int col, float val);
Matrix matrixGetRow(Matrix* matrix, int row);// Needs refactoring for cublas
Matrix matrixCopy(Matrix* matrix); // Needs refactoring for cublas
void matrixPrint(const Matrix* m);
void matrixDestroy(Matrix* matrix);
/* ------------ Math operations ------------ */
void matrixAdd(const Matrix* A, const Matrix* B, Matrix* out);
Matrix matrixSub(Matrix* m1, Matrix* m2);// Needs refactoring for cublas
Matrix matrixMul(Matrix* m1, Matrix* m2);// Needs refactoring for cublas
Matrix matrixMulf(Matrix* matrix, float value);// Needs refactoring for cublas
void matrixDot(const Matrix* A, const Matrix* B, Matrix* out);
Matrix matrixT(Matrix* matrix);// Needs refactoring for cublas

#ifdef __cplusplus
}
#endif