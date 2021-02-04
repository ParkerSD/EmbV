
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include "config.h"

 #define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

//#define DEBUG

enum
{
    e_one_d= 0,
    e_two_d
};

enum
{
    e_layer_input = 0,
    e_layer_zero = 1,
    e_layer_one = 2
};

enum
{
    e_conv_zero = 0,
    e_conv_one = 1,
    e_conv_two = 2
};

typedef float neuron;

typedef struct input
{
    neuron mat[PIXEL_COUNT];// 28x28 image (MNIST)
    neuron mat2D[IMAGE_HEIGHT+(2*PAD_IN)][IMAGE_WIDTH+(2*PAD_IN)]; // add room for padding
}input;

typedef struct layer
{
    neuron n[LAYER_SIZE_MAX];
    float gradient[LAYER_SIZE_MAX];
}layer;


typedef struct conv_layer // holds output of convolution
{
    neuron L0[FILT_A_NUM][H2A+(2*PAD_A)][W2A+(2*PAD_A)]; //3D stacks of filter outputs
    neuron L0_POOL[FILT_A_NUM][L0_POOL_Y+(2*PAD_A)][L0_POOL_X+(2*PAD_A)];
    neuron L1[FILT_B_NUM][H2B+(2*PAD_B)][W2B+(2*PAD_B)];
    neuron L1_POOL[FILT_B_NUM][L1_POOL_Y+(2*PAD_B)][L1_POOL_X+(2*PAD_B)];
    neuron L2[FILT_C_NUM][H2C][W2C];
    neuron L2_POOL[FILT_C_NUM][L2_POOL_Y][L2_POOL_X];
    neuron flat[FLAT_SIZE]; // flat buffer
}conv_layer;



typedef struct brain
{
    input* in;
    conv_layer* c;
    layer* L0;
    layer* L1;
    int result;
    int truth;
    neuron out[NUM_OUTPUTS];
    float out_grad[NUM_OUTPUTS]; //output gradient
    uint8_t softmax[NUM_OUTPUTS]; //decimal digits 0 - 9
}brain;



//function declarations
input* normalize_inputs(input* in, bool two_d); // scale input to 0-1
brain* ff(uint8_t curr_layer, brain*);   // feed forward
brain* init_brain(void);
input* init_input(void);
conv_layer* init_conv(void);
layer* init_layer(void);
void init_filters(void);
void init_weights(void);
brain* softmax(brain* b);
brain* bp(brain* b); //backprop
brain* test(brain* b, int t);
brain* randomize_input(brain* b);
brain* mnist_train(brain* b);
void accuracy(int m);
void test_accuracy(brain *b);
brain* conv(brain* b, uint8_t curr_layer);
brain* max_pool(brain* b, uint8_t curr_layer);
brain* unroll(brain* b);
int* flat_to_2D(uint8_t side, int inc, int* coord);



