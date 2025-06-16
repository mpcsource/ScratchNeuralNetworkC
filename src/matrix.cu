#include "matrix.cuh"

__global__ void matrixFillKernel(float* data, int size, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size) data[idx] = value;
}

void initCublas() {
    if (cublasHandle == NULL) {
        cublasCreate(&cublasHandle);
    }
}

void destroyCublas() {
    if (cublasHandle != NULL) {
        cublasDestroy(cublasHandle);
        cublasHandle = NULL;
    }
}

/* ----------- Access operations ----------- */
Matrix matrixCreate(int rows, int cols, float fill, int allocation_flag) {
    Matrix matrix; // Structure to hold matrix metadata and pointers.
    matrix.data         = NULL;
    matrix.d_data       = NULL;
    matrix.rows         = rows;
    matrix.cols         = cols;
    matrix.allocation_type    = allocation_flag;

    int size     = rows * cols;
    size_t bytes = size * sizeof(float);

    // Allocate host memory if requested
    if(allocation_flag == HOST_AND_DEVICE) {
        matrix.data = (float*)malloc(bytes);

        if(!matrix.data) {
            fprintf(stderr, "Host malloc failed in matrixCreate\n");
            exit(EXIT_FAILURE);
        }

        // Fill host data
        for(int i = 0; i < rows * cols; i++) {
            matrix.data[i] = fill;
        }
    }

    // Allocate device memory always.
    cudaError_t err = cudaMalloc((void**)&matrix.d_data, bytes);
    if(err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed in matrixCreate: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // This part of the code deals with filling the matrix.

    // If on CPU and GPU, copy from CPU to GPU.
    if(allocation_flag) {
        matrixUpload(&matrix);
    }

    // If only on GPU and fill is 0, use cudaMemset function.
    else if (fill == 0.0f) {
        cudaMemset(matrix.d_data, 0, bytes);
    }

    // If only on GPU and fill is another value, use fill kernel.
    else {
        int blockSize = 256;
        int numBlocks = (size + blockSize - 1) / blockSize;
        matrixFillKernel<<<numBlocks, blockSize>>>(matrix.d_data, size, fill);
    }

    return matrix;
}

void matrixUpload(Matrix* matrix) {
    if (matrix->data == NULL) {
        fprintf(stderr, "matrixUpload called on matrix without host memory\n");
        return;
    }
    size_t bytes = matrix->rows * matrix->cols * sizeof(float);
    cudaMemcpy(matrix->d_data, matrix->data, bytes, cudaMemcpyHostToDevice);
}

// makes it host and device
void matrixDownload(Matrix* matrix) {
    size_t bytes = matrix->rows * matrix->cols * sizeof(float);

    switch (matrix->allocation_type)
    {
    case HOST_AND_DEVICE:
        cudaMemcpy(matrix->data, matrix->d_data, bytes, cudaMemcpyDeviceToHost);
        break;
    
    case DEVICE_ONLY:
        fprintf(stderr, "matrixDownload warning: calling matrixDownload on a DEVICE_ONLY matrix will cause it to become HOST_AND_DEVICE.\n");
        matrix->data = (float*)malloc(matrix->rows * matrix->cols * sizeof(float));
        matrix->allocation_type = HOST_AND_DEVICE;
        cudaMemcpy(matrix->data, matrix->d_data, bytes, cudaMemcpyDeviceToHost);
        break;

    default:
        fprintf(stderr, "matrixDownload error: unknown allocation type.\n");
        exit(EXIT_FAILURE);
    }
}

float matrixGet(Matrix* matrix, int row, int col) {
    int idx = row * matrix->cols + col;

    switch (matrix->allocation_type)
    {
    case HOST_AND_DEVICE:
        return matrix->data[idx];
    
    case DEVICE_ONLY:
        float val;
        cudaMemcpy(&val, matrix->d_data + idx, sizeof(float), cudaMemcpyDeviceToHost);
        return val;

    default:
        fprintf(stderr, "matrixGet error: unknown allocation type.\n");
        exit(EXIT_FAILURE);
    }
}

void matrixSet(Matrix* matrix, int row, int col, float val) {
    int idx = row * matrix->cols + col;
    switch (matrix->allocation_type)
    {
        case HOST_AND_DEVICE: {
            cudaMemcpy(matrix->d_data + idx, &val, sizeof(float), cudaMemcpyHostToDevice);
            matrix->data[idx] = val;
            break;
        }
        
        case DEVICE_ONLY: {
            cudaMemcpy(matrix->d_data + idx, &val, sizeof(float), cudaMemcpyHostToDevice);
            break;
        }

        default: {
            fprintf(stderr, "matrixSet error: unknown allocation type.\n");
            exit(EXIT_FAILURE);
        }
    }
}

Matrix matrixGetRow(Matrix* matrix, int row) {
    Matrix result = matrixCreate(1, matrix->cols, 0, DEVICE_ONLY);

    for(int i = 0; i < matrix->cols; i++) {
        matrixSet(&result, 0, i, matrixGet(matrix, row, i));
    }

    return result;
}

Matrix matrixCopy(Matrix* matrix) {
    Matrix result = matrixCreate(matrix->rows, matrix->cols, 0, DEVICE_ONLY);
    for(int i = 0; i < matrix->rows*matrix->cols; i++)
        result.data[i] = matrix->data[i];
    return result;
}

void matrixPrint(const Matrix* m) {
    switch (m->allocation_type)
    {
    case HOST_AND_DEVICE:
        for (int i = 0; i < m->rows; ++i) {
            for (int j = 0; j < m->cols; ++j) {
                printf("%8.3f ", m->data[i * m->cols + j]);
            }
            printf("\n");
        }
        break;
    
    case DEVICE_ONLY:
        fprintf(stderr, "matrixPrint warning: printing large matrices from device can be expensive.\n");
        for (int i = 0; i < m->rows*m->cols; i++) {
            float val;
            cudaMemcpy(&val, m->d_data + i, sizeof(float), cudaMemcpyDeviceToHost);
            if(i % m->rows-1 == 0) {
                printf("%8.3f\n", val);
            } else {
                printf("%8.3f", val);
            }
            
        }
        break;

    default:
        break;
    }
}

void matrixDestroy(Matrix* matrix) {
    if(matrix->data) free(matrix->data);
    if(matrix->d_data) cudaFree(matrix->d_data);
    matrix->data = NULL;
    matrix->d_data = NULL;
}

/* ------------ Math operations ------------ */

void matrixAdd(const Matrix* A, const Matrix* B, Matrix* out) {

    // Shape check
    if ((A->cols != B->cols) || (A->rows != B->rows)) {
        fprintf(stderr, "matrixAdd shape mismatch: A is (%d x %d), B is (%d x %d)\n",
                A->rows, A->cols, B->rows, B->cols);
        exit(EXIT_FAILURE);
    }

    int rows = A->rows;
    int cols = A->cols;

    // Reallocate if placeholder
    if (out->rows != rows || out->cols != cols || out->d_data == NULL) {
        matrixDestroy(out);
       *out = matrixCreate(rows, cols, 0, out->allocation_type);
    }
    float alpha = 1.0f;

    cublasSgeam(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        cols, rows,
        &alpha,
        A->d_data, cols,
        &alpha,
        B->d_data, cols,
        out->d_data, cols
    );

    if (out->data && out->allocation_type == HOST_AND_DEVICE) {
        matrixDownload(out);
    }
    
}

Matrix matrixSub(Matrix* m1, Matrix* m2) {
    if((m1->cols != m2->cols) || (m1->rows != m2->rows)) {
        fprintf(stderr, "Matrix shape mismatch in matrixSub\n");
        exit(EXIT_FAILURE);
    }

    Matrix result = matrixCreate(m1->rows, m1->cols, 0, DEVICE_ONLY);

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

    Matrix result = matrixCreate(m1->rows, m1->cols, 0, DEVICE_ONLY);

    for (int i = 0; i < m1->rows; i++)
        for (int j = 0; j < m1->cols; j++)
            matrixSet(&result, i, j, matrixGet(m1,i,j)*matrixGet(m2, i,j));

    return result;
}

Matrix matrixMulf(Matrix* matrix, float value) {
    Matrix result = matrixCreate(matrix->rows, matrix->cols, 0, DEVICE_ONLY);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrixSet(&result, i, j, matrixGet(matrix,i,j)*value);

    return result;
}

void matrixDot(const Matrix* A, const Matrix* B, Matrix* out) {

    // Shape check
    if(A->cols != B->rows) {
        fprintf(stderr, "matrixDot shape mismatch: (%d x %d) x (%d x %d) -> (%d x %d)\n",
                A->rows, A->cols, B->rows, B->cols, out->rows, out->cols);
        exit(EXIT_FAILURE);
    }

    int rows = A->rows;
    int cols = B->cols;

    // Reallocate if placeholder
    if (out->rows != rows || out->cols != cols || out->d_data == NULL) {
        /*if (out->d_data) cudaFree(out->d_data);
        out->rows = rows;
        out->cols = cols;
        cudaMalloc(&out->d_data, rows * cols * sizeof(float));

        if (out->allocation_type == HOST_AND_DEVICE && out->data == NULL) {
            out->data = (float*)malloc(rows * cols * sizeof(float));
        }*/
       matrixDestroy(out);
       *out = matrixCreate(rows, cols, 0, out->allocation_type);
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasSgemm(
        cublasHandle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        B->cols, A->rows, A->cols,
        &alpha,
        B->d_data, B->cols,
        A->d_data, A->cols,
        &beta,
        out->d_data, out->cols
    );

    if(out->data && out->allocation_type == HOST_AND_DEVICE) {
        matrixDownload(out);
    }
}

Matrix matrixT(Matrix* matrix) {
    Matrix result = matrixCreate(matrix->cols, matrix->rows, 0, DEVICE_ONLY);

    for (int i = 0; i < matrix->rows; i++)
        for (int j = 0; j < matrix->cols; j++)
            matrixSet(&result, j, i, matrixGet(matrix, i, j));

    return result;
}