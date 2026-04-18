/*
 * Example: mnist_inference
 *
 * Loads the weights produced by mlp_mnist_float32_host into a fresh
 * 784 -> 20 -> 10 MLP and runs pure forward-pass inference on every sample in
 * the embedded MNIST test subset. Prints per-sample argmax predictions
 * alongside the ground truth, plus a final accuracy summary.
 *
 * Target compatibility: HOST, PICO1, PICO2_W, STM32*.
 *
 * Prerequisite: mlp_mnist_float32_host must have been built and run once to
 * produce src/examples/data/mnist_pretrained_float32.h. A stub version of
 * that file is checked in so this example compiles out of the box, but the
 * accuracy will be ~10% (random) until a real training run overwrites it.
 */

#include <stdio.h>
#include <stddef.h>

#include "hardware_init.h"
#include "debug_lib.h"

#include "Layer.h"
#include "Tensor.h"
#include "TensorApi.h"
#include "QuantizationApi.h"
#include "LinearApi.h"
#include "ReluApi.h"
#include "SoftmaxApi.h"
#include "InferenceApi.h"

#include "data/mnist_test_subset.h"
#include "data/mnist_pretrained_float32.h"

#define INPUT_DIM   MNIST_PRETRAINED_INPUT_DIM
#define HIDDEN_DIM  MNIST_PRETRAINED_HIDDEN_DIM
#define OUTPUT_DIM  MNIST_PRETRAINED_OUTPUT_DIM
#define MODEL_SIZE  4

static size_t argmax(const float *values, size_t n) {
    size_t best = 0;
    float bestVal = values[0];
    for (size_t i = 1; i < n; i++) {
        if (values[i] > bestVal) { bestVal = values[i]; best = i; }
    }
    return best;
}

int main(void) {
    init();
    debug_sleep(1000);
    printf("mnist_inference: loading pretrained float32 MLP\n");

    quantization_t *q = quantizationInitFloat();

    /* Copy the const pretrained arrays into mutable buffers — tensorInit*
     * wraps the pointer, it does not copy, and the layer code writes into
     * those buffers internally. Flash-resident const arrays can't be
     * mutated, so we stage them in RAM once. */
    static float w0[HIDDEN_DIM * INPUT_DIM];
    static float b0[HIDDEN_DIM];
    static float w1[OUTPUT_DIM * HIDDEN_DIM];
    static float b1[OUTPUT_DIM];
    for (size_t i = 0; i < HIDDEN_DIM * INPUT_DIM; i++) w0[i] = mnist_pretrained_w0[i];
    for (size_t i = 0; i < HIDDEN_DIM; i++)             b0[i] = mnist_pretrained_b0[i];
    for (size_t i = 0; i < OUTPUT_DIM * HIDDEN_DIM; i++) w1[i] = mnist_pretrained_w1[i];
    for (size_t i = 0; i < OUTPUT_DIM; i++)              b1[i] = mnist_pretrained_b1[i];

    size_t w0Dims[] = {HIDDEN_DIM, INPUT_DIM};
    size_t b0Dims[] = {1, HIDDEN_DIM};
    size_t w1Dims[] = {OUTPUT_DIM, HIDDEN_DIM};
    size_t b1Dims[] = {1, OUTPUT_DIM};

    tensor_t *w0t = tensorInitFloat(w0, w0Dims, 2, NULL);
    tensor_t *b0t = tensorInitFloat(b0, b0Dims, 2, NULL);
    tensor_t *w1t = tensorInitFloat(w1, w1Dims, 2, NULL);
    tensor_t *b1t = tensorInitFloat(b1, b1Dims, 2, NULL);

    layer_t *model[MODEL_SIZE] = {
        linearLayerInitNonTrainable(w0t, b0t, q),
        reluLayerInit(q, q),
        linearLayerInitNonTrainable(w1t, b1t, q),
        softmaxLayerInit(q, q),
    };

    /* Inference loop over the embedded test subset. */
    static float itemBuf[INPUT_DIM];
    size_t itemDims[] = {1, INPUT_DIM};

    size_t correct = 0;
    for (size_t id = 0; id < MNIST_TEST_SUBSET_SIZE; id++) {
        for (size_t i = 0; i < INPUT_DIM; i++) {
            itemBuf[i] = (float)mnist_test_subset_images[id * INPUT_DIM + i] / 255.0f;
        }
        tensor_t *input = tensorInitFloat(itemBuf, itemDims, 2, NULL);
        tensor_t *out   = inference(model, MODEL_SIZE, input);

        const float *probs = (const float *)out->data;
        size_t pred  = argmax(probs, OUTPUT_DIM);
        size_t label = mnist_test_subset_labels[id];
        if (pred == label) correct++;

        printf("  [%3zu] true=%zu pred=%zu %s\n", id, label, pred,
               pred == label ? "OK" : "X");

        /* The output tensor's data buffer was allocated by the framework, so
         * freeTensor() is safe. The input tensor wraps our static itemBuf;
         * freeing it would call free() on BSS and abort. The minor tensor +
         * shape leak is bounded by MNIST_TEST_SUBSET_SIZE and reclaimed at
         * process exit / MCU reset. */
        freeTensor(out);
    }

    printf("Accuracy on subset: %zu/%d = %.2f%%\n",
           correct, MNIST_TEST_SUBSET_SIZE,
           100.0 * (double)correct / (double)MNIST_TEST_SUBSET_SIZE);
    return 0;
}
