
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

        //CONFIG
#define TOTAL_LAYERS 3
#define PIXEL_COUNT 784
#define NUM_OUTPUTS 10
#define INPUT_SCALE 255
#define LAYER_SIZE_MAX 32
#define LAYER0_SIZE 32
#define LAYER1_SIZE 32
#define ETA 0.4 // learning speed (0.01 to 0.9)
#define DESIRED_TRUE 1
#define DESIRED_FALSE 0
#define TEST_CYCLES 1000
#define ACTIVE_THRESH 0 //threshold at which neuron is active and feeds data forward
#define EPOCHS 6


enum
{
    e_layer_input = 0,
    e_layer_zero = 1,
    e_layer_one = 2
};

typedef float neuron;

typedef struct layer
{
    neuron n[LAYER_SIZE_MAX];
    float gradient[LAYER_SIZE_MAX];
}layer;

typedef struct input
{
    neuron mat[PIXEL_COUNT]; // 28x28 image (MNIST)
}input;

typedef struct brain
{
    input* in;
    layer* L0;
    layer* L1;
    int result;
    int truth;
    float out[NUM_OUTPUTS];
    float out_grad[NUM_OUTPUTS];
    uint8_t softmax[NUM_OUTPUTS]; //decimal digits 0 - 9
}brain;

//function defs
input* normalize_inputs(input* in); // scale input to 0-1
brain* ff(uint8_t curr_layer, brain*);   // feed forward
brain* init_brain(void);
input* init_input(void);
layer* init_layer(void);
void init_weights(void);
brain* softmax(brain* b);
brain* bp(brain* b); //backprop
brain* test(brain* b, int t);
brain* randomize_input(brain* b);
brain* mnist_train(brain* b);
void accuracy(int m);
void test_accuracy(brain *b);
