#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1024  // Matrix size (N x N)

int main() {
    size_t size = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS uses column-major by default, so we tell it we're using row-major
    float alpha = 1.0f;
    float beta = 0.0f;

    // Time cuBLAS operation
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    // C = alpha * A * B + beta * C
    cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose
        N, N, N,
        &alpha,
        d_B, N,  // Note: order of B and A is reversed in row-major
        d_A, N,
        &beta,
        d_C, N
    );

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Copy result to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    std::cout << "cuBLAS Time: " << duration.count() << " seconds\n";
    std::cout << "C[0][0] = " << h_C[0] << "\n";

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
