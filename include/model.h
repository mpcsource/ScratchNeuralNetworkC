#include "layer.h"

#define MSE_LOSS 0

#define VERBOSE_TRAINING 0
#define SILENT_TRAINING 1

typedef struct {
    Matrix input;
    Matrix output;
    Layer** layers;
    int layers_amount;
    int loss_type;
} Model;

Model modelCreate(int loss_type);
void modelAppendLayer(Model* model, Layer* layer);
Layer* modelGetLayer(Model* model, int index);
void modelDestroy(Model* model);
Matrix modelForward(Model* model, Matrix input);
void modelBackward(Model* model, Matrix* labels, float learning_rate);
void modelSGD(Model* model, Matrix* data, Matrix* labels, int epochs, float learning_rate, int silent_training); // Batch gradient descent
void modelBGD(Model* model, Matrix* data, Matrix* labels, int epochs, float learning_rate, int silent_training); // Batch gradient descent
