#include "layer.h"

Layer layerCreate(int nin, int nout, int activation) {
    Layer layer;

    layer.weights = matrixCreate(nout, nin, 1);
    for(int i = 0; i < layer.weights.cols*layer.weights.rows; i++)
        layer.weights.data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2 * 0.1;
    layer.biases = matrixCreate(nout, 1, 0);
    layer.activation = activation;

    layer.a = matrixCreate(nout, 1, 0);
    layer.z = matrixCreate(nout, 1, 0);
    layer.da = matrixCreate(nout, 1, 0);
    layer.dz = matrixCreate(nout, 1, 0);

    return layer;
}

void layerForward(Layer* layer, Matrix* input) {
    layer->input = matrixCopy(input);
    
    Matrix wx = matrixDot(&layer->weights, input);
    Matrix z = matrixAdd(&wx, &layer->biases);

    layer->z = z;
    layerApplyActivation(layer);
    layerApplyActivationDerivative(layer);
}

void layerApplyActivation(Layer* layer) {
    switch (layer->activation)
    {
        case SIGMOID_ACTIVATION: { // Fast sigmoid actually! I lied to you
            Matrix* a_ = &layer->a;
            for(int i = 0; i < a_->rows*a_->cols; i++) {
                float z = layer->z.data[i];
                float z_abs = (z < 0) ? -z : z;
                //a_->data[i] = z / (1.0f + z_abs);
                a_->data[i] = 0.5 + (z / (2.0f + 2 * z_abs));
            }
            break;
        }
    
        default: {
            //layer->a = layer->z;
            matrixDestroy(&layer->a);
            layer->a = matrixCopy(&layer->z);
            layer->da = matrixCreate(layer->z.rows, layer->z.cols, 0);
            break;
        }
    }
}

void layerApplyActivationDerivative(Layer* layer) {
    switch (layer->activation)
    {
        case SIGMOID_ACTIVATION: {
            Matrix* da_ = &layer->da;
            for(int i = 0; i < da_->rows*da_->cols; i++) {
                float a = layer->a.data[i];
                da_->data[i] = a * ( 1.0f - a );
            }
            break;
        }
    
        default: {
            break;

        }
    }
}

void layerBackward(Layer* layer, float learning_rate) {
    Matrix input_transposed = matrixT(&layer->input);
    Matrix grad_weights = matrixDot(&layer->dz, &input_transposed);
    Matrix grad_biases = layer->dz;

    grad_weights = matrixMulf(&grad_weights, learning_rate);
    grad_biases = matrixMulf(&grad_biases, learning_rate);

    Matrix weights_updated = matrixSub(&layer->weights, &grad_weights);
    Matrix biases_updated = matrixSub(&layer->biases, &grad_biases);

    layer->weights = weights_updated;
    layer->biases = biases_updated;
}

void layerDestroy(Layer* layer) {
    matrixDestroy(&layer->weights);
    matrixDestroy(&layer->biases);
    matrixDestroy(&layer->input);
    matrixDestroy(&layer->a);
    matrixDestroy(&layer->z);
    matrixDestroy(&layer->da);
    matrixDestroy(&layer->dz);
}