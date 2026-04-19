/*
 * Example: mlp_mnist_stress_host  (HOST-only)
 *
 * Stresstest für Boilerplate-Messung: 5-Hidden-Layer MLP 784 -> 256 -> 128 ->
 * 64 -> 32 -> 10, ReLU zwischen allen, Softmax am Ende, CrossEntropy Loss,
 * FLOAT32 durchgehend. Trainiert auf MNIST-Full aus .npy.
 *
 * Ziel dieses Examples ist NICHT Convenience oder Produktions-Stil. Es ist
 * explizit so geschrieben, dass jede Layer Zeile für Zeile ohne Helper
 * aufgebaut wird — genau so sieht User-Code aus, wenn die USERAPI keine
 * Convenience-Layer hat. Der dadurch entstehende Boilerplate ist das, was
 * Pass 1 des USERAPI-Audits quantifiziert.
 *
 * Host-only: ~244K Parameter × 4 B = ~970 KB passt nicht in Pico2 W (520 KB
 * SRAM). Passt bequem in Host-BSS.
 *
 * Boilerplate-Metrik (Stand 2026-04-19): 60 Zeilen für die 5 Linear-Layer-Konstruktion,
 * davon 54 identisch mechanisch (alle Param-Inits + linearLayerInit + 4× reluLayerInit).
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "hardware_init.h"

#include "Layer.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "SgdApi.h"
#include "InferenceApi.h"
#include "TrainingLoopApi.h"
#include "CalculateGradsSequential.h"
#include "DataLoaderApi.h"
#include "DataLoader.h"
#include "NPYLoaderApi.h"
#include "Dataset.h"
#include "LossFunction.h"
#include "StorageApi.h"

#ifndef MNIST_DATA_DIR
#define MNIST_DATA_DIR "."
#endif
#define MNIST_TRAIN_X MNIST_DATA_DIR "/mnist_train_x.npy"
#define MNIST_TRAIN_Y MNIST_DATA_DIR "/mnist_train_y.npy"
#define MNIST_TEST_X  MNIST_DATA_DIR "/mnist_test_x.npy"
#define MNIST_TEST_Y  MNIST_DATA_DIR "/mnist_test_y.npy"

#define INPUT_DIM     (28 * 28)
#define H1_DIM        256
#define H2_DIM        128
#define H3_DIM        64
#define H4_DIM        32
#define OUTPUT_DIM    10
#define NUM_CLASSES   10
#define BATCH_SIZE    32
#define NUM_EPOCHS    5
#define LEARNING_RATE 0.001f
/* Model-Kette: 5 Linear + 4 ReLU (zwischen Linears) + 1 Softmax = 10 Layer. */
#define MODEL_SIZE    10

static dataset_t trainDataset;
static dataset_t testDataset;

static sample_t *getTrainSample(size_t id) { return npyGetSample(&trainDataset, id); }
static sample_t *getTestSample (size_t id) { return npyGetSample(&testDataset,  id); }
static size_t getTrainSize(void) { return trainDataset.items->size; }
static size_t getTestSize (void) { return testDataset.items->size;  }

static void flattenItems(tensorArray_t *arr) {
    for (size_t i = 0; i < arr->size; i++) {
        shape_t *shape = arr->array[i]->shape;
        size_t *newDims  = *reserveMemory(2 * sizeof(size_t));
        size_t *newOrder = *reserveMemory(2 * sizeof(size_t));
        newDims[0] = shape->dimensions[0];
        newDims[1] = shape->dimensions[1] * shape->dimensions[2];
        newOrder[0] = 0;
        newOrder[1] = 1;
        freeReservedMemory(shape->dimensions);
        freeReservedMemory(shape->orderOfDimensions);
        shape->dimensions = newDims;
        shape->orderOfDimensions = newOrder;
        shape->numberOfDimensions = 2;
    }
}

static void onEpochEnd(size_t epoch, float trainLoss, float evalLoss) {
    printf("  epoch %zu: train_loss=%.4f eval_loss=%.4f\n",
           epoch + 1, (double)trainLoss, (double)evalLoss);
}

int main(void) {
    init();
    printf("mlp_mnist_stress_host: 784->256->128->64->32->10 MLP (stress-test)\n");

    trainDataset.items  = npyLoad(MNIST_TRAIN_X);
    trainDataset.labels = npyLoad(MNIST_TRAIN_Y);
    testDataset.items   = npyLoad(MNIST_TEST_X);
    testDataset.labels  = npyLoad(MNIST_TEST_Y);
    if (!trainDataset.items || !trainDataset.labels ||
        !testDataset.items  || !testDataset.labels) {
        fprintf(stderr, "Could not load MNIST .npy files.\n");
        return 1;
    }
    flattenItems(trainDataset.items);
    flattenItems(testDataset.items);

    dataLoader_t *trainDL = dataLoaderInit(getTrainSample, getTrainSize, BATCH_SIZE,
                                            NULL, NULL, true, 42, true);
    dataLoader_t *testDL  = dataLoaderInit(getTestSample,  getTestSize,  1,
                                            NULL, NULL, false, 0, true);

    quantization_t *q = quantizationInitFloat();
    layer_t *model[MODEL_SIZE];

    /* =========================================================================
     * LAYER 1: Linear 784 -> 256  (ABSICHTLICH OHNE HELPER — Boilerplate sichtbar)
     * ========================================================================= */
    static float w0[H1_DIM * INPUT_DIM] = {0};
    size_t w0Dims[] = {H1_DIM, INPUT_DIM};
    tensor_t *w0P = tensorInitWithDistribution(XAVIER_UNIFORM, w0, w0Dims, 2, q, NULL, INPUT_DIM, H1_DIM);
    tensor_t *w0G = gradInitFloat(w0P, NULL);
    parameter_t *w0Pm = parameterInit(w0P, w0G);

    static float b0[H1_DIM] = {0};
    size_t b0Dims[] = {1, H1_DIM};
    tensor_t *b0P = tensorInitWithDistribution(ZEROS, b0, b0Dims, 2, q, NULL, 1, H1_DIM);
    tensor_t *b0G = gradInitFloat(b0P, NULL);
    parameter_t *b0Pm = parameterInit(b0P, b0G);

    model[0] = linearLayerInit(w0Pm, b0Pm, q, q, q, q);
    model[1] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 2: Linear 256 -> 128
     * ========================================================================= */
    static float w1[H2_DIM * H1_DIM] = {0};
    size_t w1Dims[] = {H2_DIM, H1_DIM};
    tensor_t *w1P = tensorInitWithDistribution(XAVIER_UNIFORM, w1, w1Dims, 2, q, NULL, H1_DIM, H2_DIM);
    tensor_t *w1G = gradInitFloat(w1P, NULL);
    parameter_t *w1Pm = parameterInit(w1P, w1G);

    static float b1[H2_DIM] = {0};
    size_t b1Dims[] = {1, H2_DIM};
    tensor_t *b1P = tensorInitWithDistribution(ZEROS, b1, b1Dims, 2, q, NULL, 1, H2_DIM);
    tensor_t *b1G = gradInitFloat(b1P, NULL);
    parameter_t *b1Pm = parameterInit(b1P, b1G);

    model[2] = linearLayerInit(w1Pm, b1Pm, q, q, q, q);
    model[3] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 3: Linear 128 -> 64
     * ========================================================================= */
    static float w2[H3_DIM * H2_DIM] = {0};
    size_t w2Dims[] = {H3_DIM, H2_DIM};
    tensor_t *w2P = tensorInitWithDistribution(XAVIER_UNIFORM, w2, w2Dims, 2, q, NULL, H2_DIM, H3_DIM);
    tensor_t *w2G = gradInitFloat(w2P, NULL);
    parameter_t *w2Pm = parameterInit(w2P, w2G);

    static float b2[H3_DIM] = {0};
    size_t b2Dims[] = {1, H3_DIM};
    tensor_t *b2P = tensorInitWithDistribution(ZEROS, b2, b2Dims, 2, q, NULL, 1, H3_DIM);
    tensor_t *b2G = gradInitFloat(b2P, NULL);
    parameter_t *b2Pm = parameterInit(b2P, b2G);

    model[4] = linearLayerInit(w2Pm, b2Pm, q, q, q, q);
    model[5] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 4: Linear 64 -> 32
     * ========================================================================= */
    static float w3[H4_DIM * H3_DIM] = {0};
    size_t w3Dims[] = {H4_DIM, H3_DIM};
    tensor_t *w3P = tensorInitWithDistribution(XAVIER_UNIFORM, w3, w3Dims, 2, q, NULL, H3_DIM, H4_DIM);
    tensor_t *w3G = gradInitFloat(w3P, NULL);
    parameter_t *w3Pm = parameterInit(w3P, w3G);

    static float b3[H4_DIM] = {0};
    size_t b3Dims[] = {1, H4_DIM};
    tensor_t *b3P = tensorInitWithDistribution(ZEROS, b3, b3Dims, 2, q, NULL, 1, H4_DIM);
    tensor_t *b3G = gradInitFloat(b3P, NULL);
    parameter_t *b3Pm = parameterInit(b3P, b3G);

    model[6] = linearLayerInit(w3Pm, b3Pm, q, q, q, q);
    model[7] = reluLayerInit(q, q);

    /* =========================================================================
     * LAYER 5: Linear 32 -> 10 + Softmax
     * ========================================================================= */
    static float w4[OUTPUT_DIM * H4_DIM] = {0};
    size_t w4Dims[] = {OUTPUT_DIM, H4_DIM};
    tensor_t *w4P = tensorInitWithDistribution(XAVIER_UNIFORM, w4, w4Dims, 2, q, NULL, H4_DIM, OUTPUT_DIM);
    tensor_t *w4G = gradInitFloat(w4P, NULL);
    parameter_t *w4Pm = parameterInit(w4P, w4G);

    static float b4[OUTPUT_DIM] = {0};
    size_t b4Dims[] = {1, OUTPUT_DIM};
    tensor_t *b4P = tensorInitWithDistribution(ZEROS, b4, b4Dims, 2, q, NULL, 1, OUTPUT_DIM);
    tensor_t *b4G = gradInitFloat(b4P, NULL);
    parameter_t *b4Pm = parameterInit(b4P, b4G);

    model[8] = linearLayerInit(w4Pm, b4Pm, q, q, q, q);
    model[9] = softmaxLayerInit(q, q);

    /* Ende Boilerplate-Sektion. Ab hier: gewöhnliches Training. */

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, MODEL_SIZE, FLOAT32);

    clock_t t0 = clock();
    trainingRunResult_t res = trainingRun(
        model, MODEL_SIZE, CROSS_ENTROPY,
        trainDL, testDL, sgd, NUM_EPOCHS,
        calculateGradsSequential, inferenceWithLoss, onEpochEnd);
    clock_t t1 = clock();

    float accuracy = evaluationEpochAccuracy(model, MODEL_SIZE, testDL, NUM_CLASSES, inference);
    printf("Done in %.2fs. final_train_loss=%.4f final_eval_loss=%.4f accuracy=%.2f%%\n",
           (double)(t1 - t0) / CLOCKS_PER_SEC,
           (double)res.finalTrainLoss, (double)res.finalEvalLoss, (double)accuracy * 100.0);
    return 0;
}
