#include <stdio.h>
#include "hardware_init.h"
#include "debug_lib.h"
#include "Layer.h"
#include "TensorAPI.h"
#include "InferenceAPI.h"
#include "LinearAPI.h"
#include "ReluAPI.h"
#include "QuantizationAPI.h"
#include "TensorConversion.h"
#include "LossFunction.h"
#include "OptimizerAPI.h"
#include "SgdAPI.h"
#include "TrainingAPI.h"
#include "Linear.h"


int main() {
    init();
    debug_sleep(1000);
    printf("Begin Calculations\n");

    float weightData[] = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
    size_t weightDims[] = {2, 3};
    size_t weightNumberOfDims = 2;
    tensor_t *weightsParam = tensorInitFloat(weightData, weightDims, weightNumberOfDims, NULL);

    float weightGradData[] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    tensor_t *weightsGrad = tensorInitFloat(weightGradData, weightDims, weightNumberOfDims, NULL);

    parameter_t *weights = parameterInit(weightsParam, weightsGrad);

    float biasData[] = {-1.f, 3.f};
    size_t biasDims[] = {2, 1};
    size_t biasNumberOfDims = 2;
    tensor_t *biasParam = tensorInitFloat(biasData, biasDims, biasNumberOfDims, NULL);

    float biasGradData[] = {0.f, 0.f};
    tensor_t *biasGrad = tensorInitFloat(biasGradData, biasDims, biasNumberOfDims, NULL);

    parameter_t *bias = parameterInit(biasParam, biasGrad);

    float input0Data[] = {-4.f, 1.f, 9.f,};
    size_t input0Dims[] = {1, 3};
    size_t input0NumberOfDims = 2;
    tensor_t *input0 = tensorInitFloat(input0Data, input0Dims, input0NumberOfDims, NULL);

    float input1Data[] = {5.f, -1.f, 2.f};
    size_t input1Dims[] = {1, 3};
    size_t input1NumberOfDims = 2;
    tensor_t *input1 = tensorInitFloat(input1Data, input1Dims, input1NumberOfDims, NULL);

    float input2Data[] = {-7.f, -5.f, 6.f};
    size_t input2Dims[] = {1, 3};
    size_t input2NumberOfDims = 2;
    tensor_t *input2 = tensorInitFloat(input2Data, input2Dims, input2NumberOfDims, NULL);

    quantization_t testQ;
    initFloat32Quantization(&testQ);
    layer_t *linear = linearLayerInit(weights, bias, &testQ, &testQ, &testQ, &testQ);

    layer_t *model[] = {linear};
    size_t sizeModel = 1;

    float label0Data[] = {59.f, -23.f};
    size_t label0Dims[] = {2, 1};
    size_t label0NumberOfDims = 2;
    tensor_t *label0 = tensorInitFloat(label0Data, label0Dims, label0NumberOfDims, NULL);

    float label1Data[] = {43.f, 249.f};
    size_t label1Dims[] = {2, 1};
    size_t label1NumberOfDims = 2;
    tensor_t *label1 = tensorInitFloat(label1Data, label1Dims, label1NumberOfDims, NULL);

    float label2Data[] = {23.f, 457.f};
    size_t label2Dims[] = {2, 1};
    size_t label2NumberOfDims = 2;
    tensor_t *label2 = tensorInitFloat(label2Data, label2Dims, label2NumberOfDims, NULL);

    optimizer_t *sgd = sgdMCreateOptim(0.01f, 0.f, 0.f, model, sizeModel, FLOAT32);
    optimizerFunctions_t sgdFns = optimizerFunctions[SGD_M];

    // IMPORTANT: we want only the weights to be trainable, to check against pytorch learned weights!
    // That's why we set sizeStates to 1 (bias states are ignored).
    sgd->sizeStates = 1;

    for (size_t i = 0; i < 100; i++) {
        trainingStats_t *trainingStats0 = calculateGrads(model, sizeModel, MSE, input0, label0);
        trainingStats_t *trainingStats1 = calculateGrads(model, sizeModel, MSE, input1, label1);
        trainingStats_t *trainingStats2 = calculateGrads(model, sizeModel, MSE, input2, label2);

        sgdFns.step(sgd);
        sgdFns.zero(sgd);

        freeTrainingStats(trainingStats0);
        freeTrainingStats(trainingStats1);
        freeTrainingStats(trainingStats2);
    }

    float expectedWeights[] = {5.f, -1.f, 9.f, 22.f, -100.f, 18.f};
    linearConfig_t *linearConfig = linear->config->linear;
    float *actualWeights = (float *)linearConfig->weights->param->data;

    const float errorPercent = 0.03f;
    for (size_t i = 0; i < 6; i++) {
        float currentThreshold = actualWeights[i] * errorPercent;
        printf("Value: %f", currentThreshold);
    }

    printf("Done\n");
  return 0;
}
