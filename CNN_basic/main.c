

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "main.h"

//#define DEBUG
#define TEST

input* normalize_inputs(input* in)
{
    for(int i = 0; i < PIXEL_COUNT; i++)
    {
        in->mat[i] = in->mat[i]/(INPUT_SCALE+1);
        //printf("%f \n", in->mat[i]); // for test
    }
    return in;
}

float ff_calc(uint8_t curr_layer, brain* b, int n_num, float temp) //multiply weights with activations, add bias, apply relu
{
    temp = 0.0;
    switch(curr_layer)
    {
        case e_layer_input:
            for(int i=0; i<PIXEL_COUNT; i++)
            {
                if(b->in->mat[i] > 0) // relu, was !=0, only for active neurons
                    temp += b->in->mat[i] * Wij[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_L0[n_num]; //one bias per neuron
            temp = 1/(1+temp); //basic sigmoid
            #ifdef DEBUG
            printf("%f \n", temp);
            #endif // DEBUG
            break;

        case e_layer_zero:
            for(int i=0; i<LAYER0_SIZE; i++)
            {
                if(b->L0->n[i] > 0) // relu, was !=0, only for active neurons
                    temp += b->L0->n[i] * Wjk[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_L1[n_num]; //one bias per neuron
            temp = 1/(1+temp); //basic sigmoid
            #ifdef DEBUG
            printf("%f \n", temp);
            #endif // DEBUG
            break;

        case e_layer_one:
            for(int i=0; i<LAYER1_SIZE; i++)
            {
                if(b->L1->n[i] > 0) // was !=0, only for active neurons
                    temp += b->L1->n[i] * Wkl[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_out[n_num]; //outputs, no activation function
            temp = 1/(1+temp); //basic sigmoid
            #ifdef DEBUG
            printf("%f \n", temp);
            #endif // DEBUG
            break;

        default:
            break;
    }
    return temp;
}

brain* ff(uint8_t curr_layer, brain* b) //feed forward to next layer, calculate next layers activations
{
    float temp = 0.0;
    switch(curr_layer)
    {
        case e_layer_input:
            #ifdef DEBUG
            printf("%s \n", "____________layer 0");
            #endif // DEBUG
            for(int i=0; i<LAYER0_SIZE; i++)
                b->L0->n[i] = ff_calc(curr_layer, b, i, temp);
            break;

        case e_layer_zero:
            #ifdef DEBUG
            printf("%s \n", "____________layer 1");
            #endif // DEBUG
            for(int i=0; i<LAYER1_SIZE; i++)
                b->L1->n[i] = ff_calc(curr_layer, b, i, temp);
            break;

        case e_layer_one:
            #ifdef DEBUG
            printf("%s \n", "____________output");
            #endif // DEBUG
            for(int i=0; i<NUM_OUTPUTS; i++)
                b->out[i] = ff_calc(curr_layer, b, i, temp);
            break;

        default:
            break;
    }
    return b;
}

layer* init_layer(void)
{
    layer* LX = malloc(sizeof(layer));
    return LX;
}

input* init_input(void)
{
    input* inX = malloc(sizeof(input));
    return inX;
}

brain* init_brain(void) //TODO add asserts
{
    brain* brainX = malloc(sizeof(brain));

    brainX->in = init_input();
    brainX->L0 = init_layer();
    brainX->L1 = init_layer();
    for(int i =0; i<NUM_OUTPUTS; i++)
        brainX->out[i] = 0.0; //zero outputs
    return brainX;
}

void init_weights(void) //init weights to a random value between 0-1
{
    //first layer
    for(int i=0; i<LAYER0_SIZE; i++) // columns
        for(int x=0; x<PIXEL_COUNT; x++) // rows
            Wij[x][i] = (float)rand() / (float)((unsigned)RAND_MAX + 1); //random number between 0-1. excluding 1
    //second
    for(int z=0; z<LAYER1_SIZE; z++)
        for(int y=0; y<LAYER0_SIZE; y++)
            Wjk[y][z] = (float)rand() / (float)((unsigned)RAND_MAX + 1);
    //third
    for(int j=0; j<NUM_OUTPUTS; j++)
        for(int k=0; k<LAYER1_SIZE; k++)
            Wkl[k][j] = (float)rand() / (float)((unsigned)RAND_MAX + 1);
}

brain* softmax(brain* b) //TODO extract cost calculation
{
    float temp = 0;
    float temp_max = 0;

    for(int i=0; i<NUM_OUTPUTS; i++) // determine largest output
    {
        if(b->out[i] > b->out[i+1]) //Warning: i+1 checks out of bounds value and discards it
            temp = b->out[i];
        if(temp > temp_max)
            temp_max = temp;
    }
    for(int j=0; j<NUM_OUTPUTS; j++) //set max to 1, all other members to 0
    {
        if(temp_max == b->out[j]) //seek to max
            b->result = j;

        b->softmax[j] = 0; //set output array to zero
    }
    b->softmax[b->result] = 1; // set resultant output to 1

    return b;
}

brain* bp(brain* b) // TODO test result vs true result
{
    //compute gradients
    //compute weights and biases
    float grad_weight_product=0;

    for(int i=0; i<NUM_OUTPUTS; i++) //hidden1->output
    {
        if(i == b->truth)
            b->out_grad[i] = (DESIRED_TRUE - b->out[i]) * (b->out[i] *(1-b->out[i])); //gradient of an output-layer neuron is equal to the target (desired) value minus the computed output value,  function
        else                                                //times the calculus derivative of the output-layer activation
            b->out_grad[i] = (DESIRED_FALSE - b->out[i]) * (b->out[i] *(1-b->out[i]));

        for(int x=0; x<LAYER1_SIZE; x++)
            Wkl[x][i] += ETA * b->out_grad[i] * b->L1->n[x]; //Wkl[LAYER1_SIZE][NUM_OUTPUTS] , delta i-h weight[2][3] = eta * hidden gradient[3] * activation of previous neuron[2]

        bias_out[i] = b->out_grad[i] * ETA; //compute bias, bias equals ETA * gradient of neuron pointed to
    }

    for(int y=0; y<LAYER1_SIZE; y++) //hidden0->hidden1
    {
        grad_weight_product = 0;
        for(int t = 0; t<NUM_OUTPUTS; t++)
            grad_weight_product += (b->out_grad[t] * Wkl[y][t]); // Wkl[LAYER1_SIZE][NUM_OUTPUTS]

        b->L1->gradient[y] = (b->L1->n[y]*(1-b->L1->n[y])) * grad_weight_product; // calculus derivative of the activation function * (FOR EACH: output gradient * hidden->output weight...)
                                                                    // NOTE was simply b->L1->n[y](activation) instead of derivative
        for(int z=0; z<LAYER0_SIZE; z++)
            Wjk[z][y] += ETA * b->L1->gradient[y] * b->L0->n[z]; // Wjk[LAYER0_SIZE][LAYER1_SIZE], calc weights

        bias_L1[y] = b->L1->gradient[y] * ETA;

    }

    for(int g=0; g<LAYER0_SIZE; g++) //input->hidden0
    {
        grad_weight_product = 0;
        for(int w=0; w<LAYER1_SIZE; w++)
            grad_weight_product += (b->L1->gradient[w] * Wjk[g][w]); //Wjk[LAYER0_SIZE][LAYER1_SIZE]

        b->L0->gradient[g] = (b->L0->n[g]*(1-b->L0->n[g])) * grad_weight_product; //The gradient of a hidden-layer neuron is equal to the calculus derivative of the activation function of the hidden layer evaluated at the local output of the neuron
                                                                   //times the sum of the product of the primary outputs times their associated hidden-to-output weights
        for(int u=0; u<PIXEL_COUNT; u++)
            Wij[u][g] += ETA * b->L0->gradient[g] * b->in->mat[u];//Wij[PIXEL_COUNT][LAYER0_SIZE];

        bias_L0[g] = b->L0->gradient[g] * ETA;
    }

    return b;
}


float accuracy(int m)
{
    static float total_matches;
    static float total_tests;
    static float acc;

    total_matches += m;
    total_tests++;
    acc = (total_matches/total_tests) * 100;
    //printf("%s", "total_matches is ");
    //printf("%f \n", total_matches);
    //printf("%s", "total_tests is ");
    //printf("%f \n", total_tests);
    printf("%s", "Accuracy is ");
    printf("%f", acc);
    printf("%s \n", "%");
    return acc;
}

brain* randomize_input(brain* b)
{
    for(int x=0; x<PIXEL_COUNT; x++) //process_input
        b->in->mat[x] = rand()%256; // randomly init input matrix  (0-255)
    return b;
}

brain* test(brain* b, int t)
{
    int match = 0;
    b = randomize_input(b);
    b->in = normalize_inputs(b->in);

    b->truth = t;
    for(int h=0; h<TEST_CYCLES; h++) // TEST_CYCLES is total ff passes for each possible output
        for(uint8_t i = 0; i<TOTAL_LAYERS; i++)
        {
            b = ff(i, b); // feed forward
            b = softmax(b);
            b = bp(b);
        }
    if(b->result == b->truth)
        match=1;

    accuracy(match);
    printf("%d", match);
    printf("%s", " match for output ");
    printf("%d", t);
    printf("%s", ", result was ");
    printf("%d \n", b->result);

    return b;
}

int main(void)
{
        //optionally import model execution or dataset for training
        //initialize weights, biases
        //initialize brain
        //process inputs
        //feed forward
        //calc softmax outputs
        //compare to desired output
        //backprop
        //kill some neurons (biosim), should help overfitting
        //export model when finished

    #ifdef DEBUG
    printf("%s", "Initializing brain...\n");
    #endif // DEBUG
    brain* mind;
    mind = init_brain();

    #ifdef DEBUG
    printf("%s", "Initializing weights...\n");
    #endif // DEBUG
    init_weights();

    #ifdef TEST
    printf("%s", "WARNING: The truth is being simulated.\n\n");
    #endif // TEST


    while(1)
    {
        #ifdef TEST
        for(int x=0; x<NUM_OUTPUTS; x++)
        {
            init_weights();
            mind = test(mind, x);
        }
        #endif // test

        #ifndef TEST
        //TODO fetch dataset
        mind->truth = 0; // placeholder
        mind->in = normalize_inputs(mind->in);

        for(uint8_t i = 0; i<TOTAL_LAYERS; i++)
            mind = ff(i, mind); // feed forward

        mind = softmax(mind);
        #ifdef DEBUG
        printf("%s \n", "____________softmax outputs");
        for(int s=0; s<NUM_OUTPUTS; s++)
            printf("%d \n", mind->softmax[s]);
        #endif

        mind = bp(mind); //WARNING the truth is being simulated
        //TODO output model
        #endif // TEST

    }
}

