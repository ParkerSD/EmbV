

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include "mnist.h"
#include "cnn.h"
#include "config.h"

//#define TEST

brain* mnist_train(brain* b)
{
    printf("%s \n", "Training...");

    load_mnist(); // fetch dataset

    for(int i=0; i<NUM_TRAIN; i++)
    {
        for(int s=0; s<SIZE; s++)
            b->in->mat[s] = (float)train_image_char[i][s];

        b->truth = train_label[i];// set truth as current MNIST label
        b->in = normalize_inputs(b->in, e_two_d);

        for(uint8_t c = 0; c<TOTAL_CONV_LAYERS; c++) //currently only using first conv layer for testing
            b = conv(b, c);//convolutions

        b = unroll(b);

        for(uint8_t i = 0; i<TOTAL_LAYERS; i++)
            b = ff(i, b); // feed forward

        b = softmax(b);
        b = bp(b);
    }
    printf("%s \n", "Completed");
    return b;
}

int main(void)
{
    //MASTER TODO
        //Add convolution and pooling layers
        //fixed point (int) implementation
        //implement death and regeneration of neurons. This should help with overfitting on small datasets

    //CNN Processes
        //optionally import model execution or dataset for training
        //initialize weights, biases
        //initialize brain
        //process inputs
        //feed forward
        //calc softmax outputs
        //compare to desired output
        //backprop
        //kill and regen some neurons
        //export model when finished

    brain* mind;
    mind = init_brain();

    init_filters();
    init_weights();

    for(int e=0; e<EPOCHS; e++) // number of passes thru training data
        mind = mnist_train(mind);

    while(1)
    {
        #ifdef TEST
        printf("%s", "WARNING: The truth is being simulated.\n\n");
        int test_output = 0;
        mind = test(mind, test_output);
        #endif // test

        #ifndef TEST //begin inferencing
        for(int i=NUM_TEST-1; i>-1; i--) // reverse MNIST numbers for test
        {
            for(int s=0; s<SIZE; s++)
                mind->in->mat[s] = (float)test_image_char[i][s]; // transfer test data to input layer

            mind->truth = test_label[i]; // set truth
            mind->in = normalize_inputs(mind->in, e_two_d);

            for(uint8_t c = 0; c<TOTAL_CONV_LAYERS; c++)
                mind = conv(mind, c); //convolutions

            mind = unroll(mind); //unroll last conv layer to flat buffer

            for(uint8_t x = 0; x<TOTAL_LAYERS; x++)
                mind = ff(x, mind); // feed forward

            mind = softmax(mind);
            test_accuracy(mind);
        }

        //TODO output model
        #endif // TEST
    }
}

