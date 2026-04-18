/*
 * Example: linear_regression
 *
 * Trains a single Linear layer (3 inputs -> 2 outputs) on three fixed
 * input/label pairs using plain SGD + MSE. The reference PyTorch run converges
 * to weights (5, -1, 9, 22, -100, 18); the assertion at the end checks that we
 * reach within 3% of those values on the WEIGHT row.
 *
 * This is the most minimal training example in the repo. It uses the manual
 * optimizer API (optimizerFunctions[SGD_M].step/zero) rather than trainingRun
 * so every step of the loop is visible. Data is embedded in the source, so the
 * program is portable to every target without any storage/filesystem layer.
 *
 * Target compatibility: HOST, PICO1, PICO2_W, STM32*.
 */

#include <stdio.h>

#include "hardware_init.h"
#include "debug_lib.h"

#include "Layer.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "SgdApi.h"
#include "OptimizerApi.h"
#include "CalculateGradsSequential.h"
#include "TrainingLoopApi.h"
#include "LossFunction.h"
#include "Linear.h"

#define IN_FEATURES    3
#define OUT_FEATURES   2
#define NUM_SAMPLES    3
#define NUM_ITERATIONS 100
#define LEARNING_RATE  0.01f

int main(void) {
    init();
    debug_sleep(1000);
    printf("linear_regression: begin\n");

    quantization_t q;
    initFloat32Quantization(&q);

    float weightData[OUT_FEATURES * IN_FEATURES] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    float weightGradData[OUT_FEATURES * IN_FEATURES] = {0};
    size_t weightDims[] = {OUT_FEATURES, IN_FEATURES};
    tensor_t *weightParam = tensorInitFloat(weightData, weightDims, 2, NULL);
    tensor_t *weightGrad  = tensorInitFloat(weightGradData, weightDims, 2, NULL);
    parameter_t *weights  = parameterInit(weightParam, weightGrad);

    float biasData[OUT_FEATURES] = {-1.f, 3.f};
    float biasGradData[OUT_FEATURES] = {0};
    size_t biasDims[] = {1, OUT_FEATURES};
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, 2, NULL);
    tensor_t *biasGrad  = tensorInitFloat(biasGradData, biasDims, 2, NULL);
    parameter_t *bias   = parameterInit(biasParam, biasGrad);

    layer_t *linear = linearLayerInit(weights, bias, &q, &q, &q, &q);
    layer_t *model[] = {linear};
    const size_t modelSize = 1;

    float input0Data[] = {-4.f, 1.f, 9.f};
    float input1Data[] = { 5.f,-1.f, 2.f};
    float input2Data[] = {-7.f,-5.f, 6.f};
    size_t inputDims[] = {1, IN_FEATURES};
    tensor_t *inputs[NUM_SAMPLES] = {
        tensorInitFloat(input0Data, inputDims, 2, NULL),
        tensorInitFloat(input1Data, inputDims, 2, NULL),
        tensorInitFloat(input2Data, inputDims, 2, NULL),
    };

    float label0Data[] = { 59.f, -23.f};
    float label1Data[] = { 43.f, 249.f};
    float label2Data[] = { 23.f, 457.f};
    size_t labelDims[] = {1, OUT_FEATURES};
    tensor_t *labels[NUM_SAMPLES] = {
        tensorInitFloat(label0Data, labelDims, 2, NULL),
        tensorInitFloat(label1Data, labelDims, 2, NULL),
        tensorInitFloat(label2Data, labelDims, 2, NULL),
    };

    optimizer_t *sgd = sgdMCreateOptim(LEARNING_RATE, 0.f, 0.f, model, modelSize, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    // Freeze bias updates so we converge to the same weights the reference
    // PyTorch run produced (PyTorch used a bias-frozen SGD too). The optimizer
    // stores [weights, bias] per linear layer; truncating sizeStates skips the
    // trailing bias entry.
    sgd->sizeStates = 1;

    for (size_t iter = 0; iter < NUM_ITERATIONS; iter++) {
        for (size_t s = 0; s < NUM_SAMPLES; s++) {
            trainingStats_t *stats = calculateGradsSequential(
                model, modelSize, MSE, inputs[s], labels[s]);
            freeTrainingStats(stats);
        }
        sgdFns.step(sgd);
        sgdFns.zero(sgd);
    }

    const float expectedWeights[OUT_FEATURES * IN_FEATURES] = {
        5.f, -1.f, 9.f, 22.f, -100.f, 18.f,
    };
    const float errorPercent = 0.03f;

    linearConfig_t *linearConfig = linear->config->linear;
    const float *actualWeights = (const float *)linearConfig->weights->param->data;

    size_t failures = 0;
    for (size_t i = 0; i < OUT_FEATURES * IN_FEATURES; i++) {
        const float threshold = expectedWeights[i] * errorPercent;
        const float diff = actualWeights[i] - expectedWeights[i];
        const float abs_diff = diff < 0 ? -diff : diff;
        const float abs_thr = threshold < 0 ? -threshold : threshold;
        printf("  w[%zu] = %f (expected %f, diff %f)\n",
               i, (double)actualWeights[i], (double)expectedWeights[i], (double)diff);
        if (abs_diff > abs_thr) failures++;
    }

    if (failures == 0) {
        printf("linear_regression: PASS (all weights within 3%% of reference)\n");
    } else {
        printf("linear_regression: FAIL (%zu weights outside tolerance)\n", failures);
    }

    // No explicit teardown: ODT's free functions assume every tensor's data
    // buffer came from its internal allocator, so feeding them stack/static
    // arrays (as this example does) crashes. MCU targets never return from
    // main(); host process exit lets the OS reclaim everything in one shot.

    return failures == 0 ? 0 : 1;
}
