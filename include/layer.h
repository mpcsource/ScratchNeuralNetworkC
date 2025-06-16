#include "matrix.h"

#define SIGMOID_ACTIVATION 0
#define LINEAR_ACTIVATION 1

typedef struct {
    Matrix weights;
    Matrix biases;
    int activation;

    Matrix input;   // Input to layer (from previous layer)
    Matrix a;       // Output after activation (layer output)
    Matrix z;       // Linear combination before activation (z = W dot x + b)
    Matrix da;      // Derivative of the activation function (for backprop)
    Matrix dz;      // Derivative of the linear combination (for backprop)
} Layer;

Layer layerCreate(int nin, int nout, int activation);
void layerForward(Layer* layer, Matrix* input);
void layerApplyActivation(Layer* layer);
void layerApplyActivationDerivative(Layer* layer);
void layerBackward(Layer* layer, float learning_rate);
void layerDestroy(Layer* layer);