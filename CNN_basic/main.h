#ifndef MAIN_H_INCLUDED
#define MAIN_H_INCLUDED


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

//config
#define TOTAL_LAYERS 3
#define PIXEL_COUNT 784
#define MAX_OUTPUT 30000 // this will change with CNN size, more MAC operations = higher final output
#define NUM_OUTPUTS 10
#define INPUT_SCALE 255
#define LAYER_SIZE_GENERIC 64
#define LAYER0_SIZE 64
#define LAYER1_SIZE 64
#define ETA 0.9 // learning speed
#define DESIRED_TRUE 1
#define DESIRED_FALSE 0
#define TEST_CYCLES 1000
#define ACTIVE_THRESH 0.05 //threshold at which neuron is active and feeds data forward


enum
{
    e_layer_input = 0,
    e_layer_zero = 1,
    e_layer_one = 2
};

typedef float neuron;

typedef struct layer
{
    neuron n[LAYER_SIZE_GENERIC];
    float gradient[LAYER_SIZE_GENERIC];
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

// weight arrays initialized randomly and dynamically, used for training then output as a model, need an option to import a model as well
float Wij[PIXEL_COUNT][LAYER0_SIZE]; //first matrix of weights, PIXEL_COUNT rows by LAYER0_SIZE columns
float Wjk[LAYER0_SIZE][LAYER1_SIZE]; //second matrix of weights, LAYER0_SIZE rows by LAYER1_SIZE columns
float Wkl[LAYER1_SIZE][NUM_OUTPUTS]; //third matrix of weights, LAYER1_SIZE rows by NUM_OUTPUTS columns

//bias arrays for each layer
float bias_L0[LAYER0_SIZE] = {0.0}; //all biases should be initialized to zero per machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
float bias_L1[LAYER1_SIZE] = {0.0};
float bias_out[NUM_OUTPUTS] = {0.0};



#endif // MAIN_H_INCLUDED
