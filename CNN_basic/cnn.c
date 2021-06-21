
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <malloc.h>
#include "cnn.h"
#include "config.h"



// WEIGHT, FILTER AND BIAS ARRAYS

// weight arrays initialized randomly and dynamically, used for training then output as a model, need an option to import a model as well
static float Wij[FLAT_SIZE][LAYER0_SIZE]; //first matrix of weights, FLAT_SIZE rows by LAYER0_SIZE columns
static float Wjk[LAYER0_SIZE][LAYER1_SIZE]; //second matrix of weights, LAYER0_SIZE rows by LAYER1_SIZE columns
static float Wkl[LAYER1_SIZE][NUM_OUTPUTS]; //third matrix of weights, LAYER1_SIZE rows by NUM_OUTPUTS columns

// bias arrays for each layer
static float bias_L0[LAYER0_SIZE] = {0.0}; //all biases should be initialized to zero per machinelearningmastery.com/why-initialize-a-neural-network-with-random-weights/
static float bias_L1[LAYER1_SIZE] = {0.0};
static float bias_out[NUM_OUTPUTS] = {0.0};

static float filt_A[FILT_A_NUM][FILT_A_DEPTH][FILT_A_SIZE][FILT_A_SIZE]; //array of 3D filter arrays, depth equals number of kernels(2d arrays) and must match input depth
static float conv_L0_grad[FILT_A_SIZE][FILT_A_SIZE];
static float filt_B[FILT_B_NUM][FILT_B_DEPTH][FILT_B_SIZE][FILT_B_SIZE];
static float conv_L1_grad[FILT_B_SIZE][FILT_B_SIZE];
static float filt_C[FILT_C_NUM][FILT_C_DEPTH][FILT_C_SIZE][FILT_C_SIZE];
static float conv_L2_grad[FILT_C_SIZE][FILT_C_SIZE]; //one gradient per weight (per filter value)

static float bias_filt_A[FILT_A_NUM] = {0.0};//filter biases, one bias for each filter layer
static float bias_filt_B[FILT_B_NUM] = {0.0};
static float bias_filt_C[FILT_C_NUM] = {0.0};


//_______TODO________
// debug conv backprop
// test max pool
// backprop thru maxpool layers
// check padding logic consistency
// confirm ff is working for all layers(conv() and ff())


//FUNCTION DEFS // NOTE: Consider proper layering, interfaces, and configurability
brain* max_pool(brain* b, uint8_t curr_layer)
{
    //take the max of a 4 pixel region and produce and intermediate layer
    switch(curr_layer)
    {
        case e_conv_zero:
            for(int i=0; i<FILT_A_NUM; i++)
                for(int h=0; h<H2A; h+=2)
                    for(int w=0; w<W2A; w+=2)
                        b->c->L0_POOL[i][(h/2)+PAD_A][(w/2)+PAD_A] =  max(max(b->c->L0[i][h][w], b->c->L0[i][h][w+1]), max(b->c->L0[i][h+1][w], b->c->L0[i][h+1][w+1])); //nested max to find max of 4 pixel area
            break;         //NOTE only first two pooling layers have padding

        case e_conv_one:
            for(int i=0; i<FILT_B_NUM; i++)
                for(int h=0; h<H2B; h+=2)
                    for(int w=0; w<W2B; w+=2)
                        b->c->L1_POOL[i][(h/2)+PAD_B][(w/2)+PAD_B] =  max(max(b->c->L1[i][h][w], b->c->L1[i][h][w+1]), max(b->c->L1[i][h+1][w], b->c->L1[i][h+1][w+1]));
            break;

        case e_conv_two:
            for(int i=0; i<FILT_C_NUM; i++)
                for(int h=0; h<H2C; h+=2)
                    for(int w=0; w<W2C; w+=2)
                        b->c->L2_POOL[i][h/2][w/2] =  max(max(b->c->L2[i][h][w], b->c->L2[i][h][w+1]), max(b->c->L2[i][h+1][w], b->c->L2[i][h+1][w+1]));
            break;

        default:
            break;
    }

    return b;
}

brain* conv(brain* b, uint8_t curr_layer)    //result of conv is  W2 = (W1 - F + 2P)/ S+1, and equally H2 = (H1 - F + 2P)/ S+1, also D = K (depth = number of filters)
{
    float accum = 0;
    int x_offset = 0;
    int y_offset = 0;
    switch(curr_layer)
    {                           // convolve 2D input with filt_A, pool and store in b->c->L0
        case e_conv_zero:       // calculate dot product and shift filter in stride
            #ifdef DEBUG
            printf("%s \n", "____________conv layer 0");
            #endif // DEBUG
            for(int z=0; z<FILT_A_NUM; z++) // input image is depth 1, number of filters creates depth of next layer
            {
                for(int d=0; d<FILT_A_DEPTH; d++)
                {
                    for(int h=0; h<H2A; h++) //NOTE: mind the padding in next layer
                    {
                        for(int w=0; w<W2A; w++) //next layer dimensions determine number of strides for filter
                        {
                            for(int y=0; y<FILT_A_SIZE; y++)
                                for(int x=0; x<FILT_A_SIZE; x++)
                                    accum += b->in->mat2D[y+y_offset][x+x_offset] * filt_A[z][d][y][x]; // multiply filter with input image from top left to right

                            x_offset += STRIDE_A;
                            accum += bias_filt_A[z]; // add bias
                            if(accum > 0) // relu
                            {
                                b->c->L0[z][h][w] = 1/(1+exp(-accum));
                                accum = 0;
                            }
                            else
                            {
                                b->c->L0[z][h][w] = 0;
                                accum = 0;
                            }
                            #ifdef DEBUG
                            printf("%f \n", b->c->L0[z][h][w]);
                            #endif // DEBUG
                        }
                        x_offset=0;
                        y_offset+=STRIDE_A;
                    }
                    y_offset=0;
                }
            }
            b=max_pool(b,curr_layer); //max_pool each depth layer of b->c->L0
            break;


        case e_conv_one:        //convolve b->c->L0 with filt_B, pool and store in b->c->L1
            #ifdef DEBUG
            printf("%s \n", "____________conv layer 1");
            #endif // DEBUG
            for(int z=0; z<FILT_B_NUM; z++) //number of filters
            {
                for(int d=0; d<FILT_B_DEPTH; d++) // input layer is depth FILT_A_NUM
                {
                    for(int h=0; h<H2B; h++) //NOTE: mind the padding in next layer
                    {
                        for(int w=0; w<W2B; w++)
                        {
                            for(int y=0; y<FILT_B_SIZE; y++)
                                for(int x=0; x<FILT_B_SIZE; x++)
                                    accum += b->c->L0_POOL[d][y+y_offset][x+x_offset] * filt_B[z][d][y][x]; // multiply filter with input image from top left to right

                            x_offset += STRIDE_B;
                            accum += bias_filt_B[z]; // add bias
                            if(accum > 0) // relu
                            {
                                b->c->L1[z][h][w] = 1/(1+exp(-accum)); //sigmoid
                                accum = 0;
                            }
                            else
                            {
                                b->c->L1[z][h][w] = 0;
                                accum = 0;
                            }
                            #ifdef DEBUG
                            printf("%f \n", b->c->L1[z][h][w]);
                            #endif // DEBUG
                        }
                        x_offset=0;
                        y_offset+=STRIDE_B;
                    }
                    y_offset=0;
                }
            }
            b=max_pool(b,curr_layer); //max_pool each depth layer of b->c->L1
            break;

        case e_conv_two:        //convolve b->c->L1 with filt_C, pool and store in b->c->L2
            #ifdef DEBUG
            printf("%s \n", "____________conv layer 2");
            #endif // DEBUG
            for(int z=0; z<FILT_C_NUM; z++) //number of filters
            {
                for(int d=0; d<FILT_C_DEPTH; d++) // input layer is depth FILT_B_NUM
                {
                    for(int h=0; h<H2C; h++) //NOTE: next layer is dense, so no padding in next layer
                    {
                        for(int w=0; w<W2C; w++)
                        {
                                for(int y=0; y<FILT_C_SIZE; y++)
                                    for(int x=0; x<FILT_C_SIZE; x++)
                                        accum += b->c->L1_POOL[d][y+y_offset][x+x_offset] * filt_C[z][d][y][x]; // multiply filter with input image from top left to right

                            x_offset += STRIDE_C;
                            accum += bias_filt_C[z]; // add bias
                            if(accum > 0) // relu
                            {
                                b->c->L2[z][h][w] = 1/(1+exp(-accum)); //sigmoid
                                accum = 0;
                            }
                            else
                            {
                                b->c->L2[z][h][w] = 0;
                                accum = 0;
                            }
                            #ifdef DEBUG
                            printf("%f \n", b->c->L2[z][h][w]);
                            #endif // DEBUG
                        }
                        x_offset=0;
                        y_offset+=STRIDE_C;
                    }
                    y_offset=0;
                }
            }
            b=max_pool(b,curr_layer); //max_pool each depth layer of b->c->L1
            break;

        default:
            break;
    }

    return b;
}

brain* unroll(brain* b)  // modified for single conv layer
{
    int i=0;
    for(int z=0; z<FILT_A_NUM; z++)
        for(int y=0; y<L0_POOL_Y; y++) //no padding in L2_pool layer
            for(int x=0; x<L0_POOL_X; x++)
            {
                b->c->flat[i] = b->c->L0_POOL[z][y][x];
                i++;
            }

    return b;
}


input* normalize_inputs(input* in, bool two_d)
{
    if(!two_d)
    {
        for(int i = 0; i < PIXEL_COUNT; i++) // false input
        {
            in->mat[i] = in->mat[i]/(INPUT_SCALE+1);
            //printf("%f \n", in->mat[i]); // for test
        }
    }
    else //roll into 2d array
    {
        int x = 0;
        for(int y = PAD_IN; y < IMAGE_HEIGHT+PAD_IN; y++)//compensate for padding
        {
            for(int z = PAD_IN; z < IMAGE_HEIGHT+PAD_IN; z++)
            {
                in->mat2D[y][z] = in->mat[x]/(INPUT_SCALE+1); //TODO: optimize this step
                x++;
                //printf("%f", in->mat2D[y][z]); // for test
            }
            //putchar('\n');
        }
    }
    return in;
}


float ff_calc(uint8_t curr_layer, brain* b, int n_num, float temp) //multiply weights with activations, add bias, apply relu
{
    temp = 0.0;
    switch(curr_layer)
    {
        case e_layer_input:
            for(int i=0; i<FLAT_SIZE; i++)
            {
                if(b->c->flat[i] > ACTIVE_THRESH) // relu, was !=0, only for active neurons
                    temp += b->c->flat[i] * Wij[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_L0[n_num]; //one bias per neuron, NOTE: this has little effect
            temp = 1/(1+exp(-temp));//basic sigmoid
            #ifdef DEBUG
            printf("%f \n", temp);
            #endif // DEBUG
            break;

        case e_layer_zero:
            for(int i=0; i<LAYER0_SIZE; i++)
            {
                if(b->L0->n[i] > ACTIVE_THRESH) // relu, was !=0, only for active neurons
                    temp += b->L0->n[i] * Wjk[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_L1[n_num];  //one bias per neuron
            temp = 1/(1+exp(-temp)); //basic sigmoid
            #ifdef DEBUG
            printf("%f \n", temp);
            #endif // DEBUG
            break;

        case e_layer_one:
            for(int i=0; i<LAYER1_SIZE; i++)
            {
                if(b->L1->n[i] > ACTIVE_THRESH) // was !=0, only for active neurons
                    temp += b->L1->n[i] * Wkl[i][n_num]; //each column holds all feed-in weights for that neuron
            }
            temp += bias_out[n_num]; //outputs, no activation function
            temp = 1/(1+exp(-temp)); //basic sigmoid
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


conv_layer* init_conv(void)
{
    conv_layer* c = malloc(sizeof(conv_layer));
    return c;
}


brain* init_brain(void) //TODO add asserts
{
    #ifdef DEBUG
    printf("%s", "Initializing brain...\n");
    #endif // DEBUG
    brain* brainX = malloc(sizeof(brain));

    brainX->in = init_input();
    brainX->c = init_conv();
    brainX->L0 = init_layer();
    brainX->L1 = init_layer();

    for(int y=0; y<IMAGE_HEIGHT+(2*PAD_IN); y++)
        for(int x=0; x<IMAGE_WIDTH+(2*PAD_IN); x++)
            brainX->in->mat2D[y][x] = 0.0; // zero inputs
    for(int i =0; i<NUM_OUTPUTS; i++)
        brainX->out[i] = 0.0; //zero outputs
    return brainX;
}


void init_weights(void) //init weights to a random value between 0-1
{
    #ifdef DEBUG
    printf("%s", "Initializing weights...\n");
    #endif // DEBUG
    //first layer
    for(int i=0; i<LAYER0_SIZE; i++) // columns
        for(int x=0; x<FLAT_SIZE; x++) // rows
            Wij[x][i] = -1+2*((float)rand())/RAND_MAX;//random num (-1 to 1), was (float)rand()/(float)((unsigned)RAND_MAX + 1); ( 0 to 1, excluding 1 )
    //second
    for(int z=0; z<LAYER1_SIZE; z++)
        for(int y=0; y<LAYER0_SIZE; y++)
            Wjk[y][z] = -1+2*((float)rand())/RAND_MAX;
    //third
    for(int j=0; j<NUM_OUTPUTS; j++)
        for(int k=0; k<LAYER1_SIZE; k++)
            Wkl[k][j] = -1+2*((float)rand())/RAND_MAX;
}

void init_filters(void)
{
    //first filter layer
    for(int z=0; z<FILT_A_NUM; z++)
        for(int d=0; d<FILT_A_DEPTH; d++)
            for(int y=0; y<FILT_A_SIZE; y++) //rows
                for(int x=0; x<FILT_A_SIZE; x++) //columns
                    filt_A[z][d][y][x] = -1+2*((float)rand())/RAND_MAX;
    //second layer filter
    for(int z=0; z<FILT_B_NUM; z++)
        for(int d=0; d<FILT_B_DEPTH; d++)
            for(int y=0; y<FILT_B_SIZE; y++) //rows
                for(int x=0; x<FILT_B_SIZE; x++) //columns
                    filt_B[z][d][y][x] = -1+2*((float)rand())/RAND_MAX;
    //third layer filter
    for(int z=0; z<FILT_C_NUM; z++)
        for(int d=0; d<FILT_C_DEPTH; d++)
            for(int y=0; y<FILT_C_SIZE; y++) //rows
                for(int x=0; x<FILT_C_SIZE; x++) //columns
                    filt_C[z][d][y][x] = -1+2*((float)rand())/RAND_MAX;
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

    #ifdef DEBUG
    printf("%s \n", "____________softmax outputs");
    for(int s=0; s<NUM_OUTPUTS; s++)
        printf("%d \n", b->softmax[s]);

    printf("%s", "Truth: ");
    printf("%d \n", b->truth);
    printf("%s", "Result: ");
    printf("%d \n", b->result);
    #endif

    return b;
}


brain* bp(brain* b) // TODO test result vs true result
{
    //compute gradients
    //compute weights and biases
    //backpropogate
    float grad_weight_product=0;
    float local_grad=0;
    int x_offset=0;
    int y_offset=0;
    int *xy = malloc(sizeof(int[2]));

    //hidden1 <- output
    for(int i=0; i<NUM_OUTPUTS; i++)
    {
        if(i == b->truth)
            b->out_grad[i] = (DESIRED_TRUE - b->out[i]) * (b->out[i] *(1-b->out[i])); //gradient of an output-layer neuron is equal to the target (desired) value minus the computed output value,  function
        else                                                                        //times the calculus derivative of the output-layer activation
            b->out_grad[i] = (DESIRED_FALSE - b->out[i]) * (b->out[i] *(1-b->out[i]));

        for(int x=0; x<LAYER1_SIZE; x++)
            Wkl[x][i] += ETA * b->out_grad[i] * b->L1->n[x]; //Wkl[LAYER1_SIZE][NUM_OUTPUTS] , delta i-h weight[2][3] = eta * hidden gradient[3] * activation of previous neuron[2]

        bias_out[i] = b->out_grad[i] * ETA; //compute bias, bias equals ETA * gradient of neuron pointed to
    }

    //hidden0 <- hidden1
    for(int y=0; y<LAYER1_SIZE; y++)
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

    //conv L2 <- hidden0
    for(int g=0; g<LAYER0_SIZE; g++)
    {
        grad_weight_product = 0;
        for(int w=0; w<LAYER1_SIZE; w++)
            grad_weight_product += (b->L1->gradient[w] * Wjk[g][w]); //Wjk[LAYER0_SIZE][LAYER1_SIZE]

        b->L0->gradient[g] = (b->L0->n[g]*(1-b->L0->n[g])) * grad_weight_product; //The gradient of a hidden-layer neuron is equal to the calculus derivative of the activation function of the hidden layer evaluated at the local output of the neuron
                                                                   //times the sum of the product of the primary outputs times their associated hidden-to-output weights
        for(int u=0; u<FLAT_SIZE; u++)
            Wij[u][g] += ETA * b->L0->gradient[g] * b->c->flat[u];//Wij[FLAT_SIZE][LAYER0_SIZE];

        bias_L0[g] = b->L0->gradient[g] * ETA;
    }
/*
    //_____________________________________________ERROR: conv layer backprop below broken_________________________________________
    //                                 NOTE: determine how to effectively backprop thru max pool layers
    //                                 NOTE: winning unit from pooling is tracked and gradient(error) is applied to winning unit only
    //                                 NOTE: breaking logic assumes same num of filters in each layer

    //conv L1 <- conv L2(flat buffer input)
    for(int i=0; i<FILT_C_NUM; i++)
    {
        for(int u=(i*FEAT_MAP_SIZE_C); u<FEAT_MAP_SIZE_C+(i*FEAT_MAP_SIZE_C); u++) //FEAT_MAP 1 maps to DENSE_RANGE 1 and so on
        {
            grad_weight_product=0;

            for(int w=0; w<LAYER0_SIZE; w++) // each feature map has limited connectivity to first dense layer
                grad_weight_product += (b->L0->gradient[w] * Wij[u][w]); // static float Wij[FLAT_SIZE][LAYER0_SIZE];

            xy = flat_to_2D(L2_POOL_X, u, xy); // test this
            local_grad = (b->c->L2_POOL[i][*xy][*(xy+1)]) * (1-b->c->L2_POOL[i][*xy][*(xy+1)]);

            for(int y=0; y<FILT_C_SIZE; y++)
                for(int f=0; f<FILT_C_SIZE; f++)
                    conv_L2_grad[y][f] = local_grad * grad_weight_product; // is this correct derivative?

            //printf("%s \n", "Filt C______________________"); //NOTE: a gradient is applied to each weight NOT each neuron.
            for(int n=0; n<FILT_B_NUM; n++)
                for(int l=0; l<FILT_C_SIZE; l++)
                    for(int m=0; m<FILT_C_SIZE; m++)
                    {
                        filt_C[i][l][m] += ETA * conv_L2_grad[l][m] * b->c->L1_POOL[n][l+y_offset][m+x_offset]; //NOTE: new filter kernel = (input feature map) * (gradients) * (learning rate)
                        //printf("%f \n", filt_C[i][l][m]);
                        //bias_filt_C[i] = conv_L2_grad[l][m] * ETA; //wrong, accumulating here drops accuracy by 20%
                    }

            x_offset+=STRIDE_C; // check logic here
            if(x_offset>L2_POOL_X-1)
            {
                x_offset=0;
                y_offset+=STRIDE_C; // if y_offset if running over L2_POOL_Y then feature map size is wrong
            }
        }
        y_offset=0;
    }

    //conv L0 <- conv L1
    for(int i=0; i<FILT_B_NUM; i++)
    {
        for(int x=(i*FEAT_MAP_SIZE_B); x<FEAT_MAP_SIZE_B+(i*FEAT_MAP_SIZE_B); x++)
        {
            grad_weight_product = 0;
            for(int l=0; l<FILT_C_SIZE; l++)
                for(int m=0; m<FILT_C_SIZE; m++)
                    grad_weight_product += (conv_L2_grad[l][m] * filt_C[i][l][m]); //breaking logic here, filt_C is reference by filt_B num i

            xy = flat_to_2D(L1_POOL_X, x, xy); // test this
            local_grad = (b->c->L1_POOL[i][*xy][*(xy+1)]*(1-b->c->L1_POOL[i][*xy][*(xy+1)]));

            for(int y=0; y<FILT_B_SIZE; y++)
                for(int x=0; x<FILT_B_SIZE; x++)
                    conv_L1_grad[y][x] = local_grad * grad_weight_product; // is this correct derivative?

            //printf("%s \n", "Filt B______________________");
            for(int n=0; n<FILT_A_NUM; n++)
                for(int l=0; l<FILT_B_SIZE; l++)
                    for(int m=0; m<FILT_B_SIZE; m++)
                    {
                        filt_B[i][l][m] += ETA * conv_L1_grad[l][m] * b->c->L0_POOL[n][l+y_offset][m+x_offset]; //NOTE: new filter kernel = (input feature map) * (gradients) * (learning rate)
                        //printf("%f \n", filt_B[i][l][m]);
                        //bias_filt_B[i] = conv_L1_grad[l][m] * ETA; //wrong
                    }

            x_offset+=STRIDE_B; // check logic here
            if(x_offset>L1_POOL_X-1)
            {
                x_offset=0;
                y_offset+=STRIDE_B;
            }
        }
        y_offset=0;
    }
*/
    //image input <- conv L0
    for(int i=0; i<FILT_A_NUM; i++)
    {
        for(int d=0; d<FILT_A_DEPTH; d++)
        {
            grad_weight_product = 0;

            for(int x=0; x<FLAT_SIZE; x++)
            {
                for(int w=0; w<LAYER0_SIZE; w++)
                    grad_weight_product += (b->L0->gradient[w] * Wij[x][w]); // static float Wij[FLAT_SIZE][LAYER0_SIZE];

                xy = flat_to_2D(L0_POOL_X, x, xy); // test this
                local_grad = (b->c->L0_POOL[i][*xy][*(xy+1)]*(1-b->c->L0_POOL[i][*xy][*(xy+1)])); // sigmoid derivative
            }


            for(int y=0; y<FILT_A_SIZE; y++)
                for(int x=0; x<FILT_A_SIZE; x++)
                    conv_L0_grad[y][x] = local_grad * grad_weight_product; // is this correct derivative?

           // printf("%s \n", "Filt A____________________ ");
            for(int l=0; l<FILT_A_SIZE; l++)
                for(int m=0; m<FILT_A_SIZE; m++)
                {
                    filt_A[i][d][l][m] += ETA * conv_L0_grad[l][m] * b->in->mat2D[l+y_offset][m+x_offset]; //NOTE: new filter kernel = (input feature map) * (gradients) * (learning rate)
                    //printf("%f \n", filt_A[i][l][m]);
                    //bias_filt_A[i] = conv_L0_grad[l][m] * ETA; //wrong
                }

            x_offset+=STRIDE_A; // check logic here
            if(x_offset>L0_POOL_X-1)
            {
                x_offset=0;
                y_offset+=STRIDE_A;
            }
        }
        y_offset=0;
    }

    return b;
}

int* flat_to_2D(uint8_t side, int inc, int* coord) // coord(row,column)
{
    *coord = (inc/side); //row
    *(coord+1) = (inc%side); //column
    if(*coord > side-1)
        *coord = *coord-side;

    return coord;
}

brain* randomize_input(brain* b)
{
    for(int x=0; x<PIXEL_COUNT; x++) //process_input
        b->in->mat[x] = rand()%256; // randomly init input matrix  (0-255)
    return b;
}



void accuracy(int m)
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
}


void test_accuracy(brain *b)
{
    int match = 0;
    if(b->result == b->truth)
        match = 1;

    accuracy(match);
    printf("%d", match);
    printf("%s", " match for output ");
    printf("%d", b->truth);
    printf("%s", ", result was ");
    printf("%d \n", b->result);
}


brain* test(brain* b, int t) //Note, not currently used
{
    b = randomize_input(b);
    b->in = normalize_inputs(b->in, false);

    b->truth = t;
    for(int h=0; h<TEST_CYCLES; h++) // TEST_CYCLES is total ff passes for each possible output
        for(uint8_t i = 0; i<TOTAL_LAYERS; i++)
        {
            b = ff(i, b); // feed forward
            b = softmax(b);
            b = bp(b); //back prop
        }

    test_accuracy(b);

    return b;
}


