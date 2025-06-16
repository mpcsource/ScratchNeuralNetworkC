#include <stdio.h>
//#include "model.cuh"
#include "matrix.cuh"


int C1Matrix_T1Access() {

    // Create matrix.
    Matrix matrix = matrixCreate(2, 2, 1, DEVICE_ONLY);

    // Write number.
    int SOLUTION = 20;
    matrixSet(&matrix, 0, 0, SOLUTION);

    // Read number.
    float num = matrixGet(&matrix, 0, 0);

    // Destroy matrix.
    matrixDestroy(&matrix);

    // Evaluate results.
    if (num == 20) {
        printf("%s successful.\n", __func__); 
        return 1;
    } else {
        printf("%s failed.\n", __func__);
        printf("Reason: %.0f != %d.\n", num, SOLUTION);
        return 0;
    }
    
}

int C1Matrix_T2Dot() {
    // Create matrices.
    Matrix mat1 = matrixCreate(2, 2, 2, DEVICE_ONLY);
    Matrix mat2 = matrixCreate(2, 2, 6, DEVICE_ONLY);

    // Dot product.
    Matrix mat3 = matrixCreate(mat1.rows, mat2.cols, 0, HOST_AND_DEVICE);
    matrixDot(&mat1, &mat2, &mat3);

    // Read number.
    float num = matrixGet(&mat3, 0, 0);

    // Destroy matrix.
    matrixDestroy(&mat1);
    matrixDestroy(&mat2);
    matrixDestroy(&mat3);

    // Evaluate results.
    if (num == 24) {
        printf("%s successful.\n", __func__); 
        return 1;
    } else {
        printf("%s failed.\n", __func__);
        printf("Reason: %.0f != 24.\n", num);
        return 0;
    }
    
}
/*
int C2Layer_T1Access() {

    Layer l1 = layerCreate(1, 1, SIGMOID_ACTIVATION);

    // Evaluate results.
    if (matrixGet(&l1.weights, 0, 0) == 1) {
        printf("%s successful.\n", __func__); 
        return 1;
    } else {
        printf("%s failed.\n", __func__);
        printf("Reason: %.0f != 1.\n", matrixGet(&l1.weights, 0, 0));
        return 0;
    }

    layerDestroy(&l1);

    return 1;
}

int C3Model_T1Backprop() {

    Matrix x = matrixCreate(1, 1, 5);
    Matrix y = matrixCreate(1, 1, 1);
    Layer l1 = layerCreate(1, 1, SIGMOID_ACTIVATION);
    Model model = modelCreate(MSE_LOSS);
    modelAppendLayer(&model, &l1);

    printf("First prediction: %f\n", matrixGet(&l1.a, 0, 0));
    //printf("%f\n", matrixGet(&l1.da, 0, 0));
    //printf("%f\n", matrixGet(&l1.z, 0, 0));
    //printf("%f\n", matrixGet(&l1.dz, 0, 0));

    modelForward(&model, x);
    modelSGD(&model, &x, &y, 1000, 1, SILENT_TRAINING);

    printf("Second prediction: %f\n", matrixGet(&l1.a, 0, 0));
    //printf("%f\n", matrixGet(&l2.da, 0, 0));
    //printf("%f\n", matrixGet(&l2.z, 0, 0));
    //printf("%f\n", matrixGet(&l2.dz, 0, 0));

    // Evaluate results.
    if (matrixGet(&l1.a, 0, 0) > 0.85) {
        printf("%s successful.\n", __func__); 
        return 1;
    } else {
        printf("%s failed.\n", __func__);
        printf("Reason: Accuracy too low: %.0f <= 0.\n", matrixGet(&l1.a, 0, 0));
        return 0;
    }

    modelDestroy(&model);

    return 1;
}

int C3Model_T2XOR() {

    Matrix x = matrixCreate(4, 2, 0);
    matrixSet(&x, 0, 0, 0); // (0, 0) = 0
    matrixSet(&x, 0, 1, 0); // (0, 1) = 0
    matrixSet(&x, 1, 0, 0); // (1, 0) = 0
    matrixSet(&x, 1, 1, 1); // (1, 1) = 1
    matrixSet(&x, 2, 0, 1); // (2, 0) = 1
    matrixSet(&x, 2, 1, 0); // (2, 1) = 0
    matrixSet(&x, 3, 0, 1); // (3, 0) = 1
    matrixSet(&x, 3, 1, 1); // (3, 1) = 1


    Matrix y = matrixCreate(4, 1, 0);
    matrixSet(&y, 0, 0, 0); // (0, 0) = 0
    matrixSet(&y, 1, 0, 1); // (1, 0) = 1
    matrixSet(&y, 2, 0, 1); // (2, 0) = 1
    matrixSet(&y, 3, 0, 0); // (3, 0) = 0

    Layer l1 = layerCreate(2, 128*2, SIGMOID_ACTIVATION);
    Layer l2 = layerCreate(128*2, 1, SIGMOID_ACTIVATION);
    Model model = modelCreate(MSE_LOSS);
    modelAppendLayer(&model, &l1);
    modelAppendLayer(&model, &l2);

    printf("First prediction:\n");
    for(int i = 0; i < 4; i++)
        printf("%.2f\n", matrixGet(&l2.a, i, 0));

    //modelForward(&model, x);
    modelSGD(&model, &x, &y, 1000, 0.1, SILENT_TRAINING);

    printf("Second prediction:\n");
    for(int i = 0; i < 4; i++) {
        Matrix xi = matrixGetRow(&x, i);
        xi = matrixT(&xi);
        modelForward(&model, xi);
        printf("%f\n", matrixGet(&l2.a, 0, 0));
    }

    matrixDestroy(&x);
    matrixDestroy(&y);
    modelDestroy(&model);

    return 1;
}*/

int main() {  

    int test_count = 5;
    int successful_count = 0;

    printf("Executing tests...\n");
    
    initCublas();

    successful_count += C1Matrix_T1Access();
    successful_count += C1Matrix_T2Dot();
    //successful_count += C2Layer_T1Access();
    //successful_count += C3Model_T1Backprop();
    //successful_count += C3Model_T2XOR();


    // Inline temporary test
    int N = 4096;
    Matrix mat1 = matrixCreate(N, N, 10, DEVICE_ONLY);
    Matrix mat2 = matrixCreate(N, N, 10, DEVICE_ONLY);
    Matrix mat3 = matrixCreate(mat1.rows, mat2.cols, 0, DEVICE_ONLY);
    matrixDot(&mat1, &mat2, &mat3);
    matrixDownload(&mat3);
    //matrixPrint(&mat3);
    matrixDestroy(&mat1);
    matrixDestroy(&mat2);
    matrixDestroy(&mat3);
    
    printf("%d out of %d tests were successful (%.1f%%).\n", successful_count, test_count, (float)successful_count/test_count*100);
    
    destroyCublas();
    return 0;
}