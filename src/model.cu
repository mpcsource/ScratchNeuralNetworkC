#include "model.cuh"

Model modelCreate(int loss_type) {
    Model model;

    model.layers = (Layer**)malloc(sizeof(Layer*));
    model.layers_amount = 0;
    model.loss_type = loss_type;

    return model;
}

void modelAppendLayer(Model* model, Layer* layer) {
    model->layers_amount++;
    model->layers = (Layer**)realloc(model->layers, model->layers_amount*sizeof(Layer*));
    
    model->layers[model->layers_amount-1] = layer;
}

Layer* modelGetLayer(Model* model, int index) {
    return model->layers[index];
}

void modelDestroy(Model* model) {
    for(int i = 0; i < model->layers_amount; i++) {
        layerDestroy(modelGetLayer(model, i));
    }
}

Matrix modelForward(Model* model, Matrix input) {
    
    model->input = input;
    for(int i = 0; i < model->layers_amount; i++) {
        Layer* layer = modelGetLayer(model, i);
        layerForward(layer, &input);
        //input = layer->input;
        input = matrixCopy(&layer->a);
    }

    return input;
}

void modelBackward(Model* model, Matrix* labels, float learning_rate) {
    //Layer* output_layer = &model->layers[model->layers_amount-1];
    Layer* output_layer = modelGetLayer(model, model->layers_amount-1);
    Matrix dloss = matrixSub(&output_layer->a, labels);
    Matrix doutput = matrixCreate(1, 1, 0, DEVICE_ONLY);
    matrixDot(&dloss, &output_layer->da, &doutput);

    output_layer->dz = doutput;
    layerBackward(output_layer, learning_rate);

    for(int i = model->layers_amount-2; i > 0; i--) {
        //Layer* next_layer = &*model->layers[i+1];
        //Layer* layer = &*model->layers[i];
        Layer* next_layer = modelGetLayer(model, i+1);
        Layer* layer = modelGetLayer(model, i);

        Matrix dh = next_layer->weights;
        dh = matrixT(&dh);
        matrixDot(&dh, &next_layer->dz, &dh);
        dh = matrixMul(&dh, &layer->da);

        layer->dz = dh;
        layerBackward(layer, learning_rate);
    }
}

void modelSGD(Model* model, Matrix* data, Matrix* labels, int epochs, float learning_rate, int silent_training) {
    int size = data->rows;

    for(int epochi = 0; epochi < epochs; epochi++) {
        if(!silent_training)
            printf("Epoch: %d\n", epochi+1);
        for(int rowi = 0; rowi < size; rowi++) {
            Matrix image = matrixGetRow(data, rowi);
            image = matrixT(&image);

            Matrix label = matrixGetRow(labels, rowi);
            label = matrixT(&label);

            modelForward(model, image);
            modelBackward(model, &label, learning_rate);
        }
    }
}


void modelBGD(Model* model, Matrix* data, Matrix* labels, int epochs, float learning_rate, int silent_training) {
    
    Matrix dataT = matrixT(data);
    Matrix labelsT = matrixT(labels);

    for(int epochi = 0; epochi < epochs; epochi++) {
        if(!silent_training)
            printf("Epoch: %d\n", epochi+1);

        modelForward(model, dataT);
        modelBackward(model, &labelsT, learning_rate);
    }
}