#ifndef CONFIG_H_INCLUDED
#define CONFIG_H_INCLUDED

        //CNN CONFIG
#define TOTAL_LAYERS 3
#define IMAGE_HEIGHT 28
#define IMAGE_WIDTH 28
#define PIXEL_COUNT (IMAGE_HEIGHT * IMAGE_WIDTH)
#define NUM_OUTPUTS 10
#define INPUT_SCALE 255
#define LAYER_SIZE_MAX 32
#define LAYER0_SIZE 32 // must be power of 2
#define LAYER1_SIZE 32
#define ETA 0.4// learning speed (0.01 to 0.9)
#define DESIRED_TRUE 1
#define DESIRED_FALSE 0
#define TEST_CYCLES 1000
#define ACTIVE_THRESH 0 //threshold at which neuron is active and feeds data forward
#define EPOCHS 1 // 6 is current optimum

#define TOTAL_CONV_LAYERS 3
#define FILT_A_NUM 2 // number of filters (power of 2)
#define FILT_A_SIZE 3 // filter 0 size
#define FILT_B_NUM 2 // number of filters
#define FILT_B_SIZE 3 // filter 1 size
#define FILT_C_NUM 2 // number of filters, must be power of 2
#define FILT_C_SIZE 2 // filter 2 size
#define PAD_IN 3 // padding for input
#define PAD_A 2 // padding for filter A
#define PAD_B 1 // padding for filter B
#define STRIDE_A 1
#define STRIDE_B 1
#define STRIDE_C 1
#define W2A (((IMAGE_WIDTH - FILT_A_SIZE + (2*PAD_IN))/STRIDE_A)+1)  //W2 = (W1 - F + 2P)/ S+1, width equation for result of convolution
#define H2A (((IMAGE_HEIGHT - FILT_A_SIZE + (2*PAD_IN))/STRIDE_A)+1)  //H2 = (H1 - F + 2P)/ S+1
#define W2B (((((W2A)/(L0_POOL_FACTOR)) - FILT_B_SIZE + (2*PAD_A))/STRIDE_B)+1)
#define H2B (((((H2A)/(L0_POOL_FACTOR)) - FILT_B_SIZE + (2*PAD_A))/STRIDE_B)+1)
#define W2C (((((H2B)/(L1_POOL_FACTOR)) - FILT_C_SIZE + (2*PAD_B))/STRIDE_C)+1)
#define H2C (((((H2B)/(L1_POOL_FACTOR)) - FILT_C_SIZE + (2*PAD_B))/STRIDE_C)+1)
#define A_SIZE ((FILT_A_NUM)*(L0_POOL_Y)*(L0_POOL_X))
#define B_SIZE ((FILT_B_NUM)*(L1_POOL_Y)*(L1_POOL_X))
#define FLAT_SIZE ((FILT_C_NUM)*(L2_POOL_Y)*(L2_POOL_X)) // was FILT_C_NUM * H2C * W2C
#define DENSE_RANGE ((LAYER0_SIZE)/(FILT_C_NUM)) //=8, each feature map connects to 8 neurons in first dense layer
#define FEAT_MAP_SIZE_C ((FLAT_SIZE)/(FILT_C_NUM))
#define FEAT_MAP_SIZE_B ((B_SIZE)/(FILT_B_NUM))
#define FEAT_MAP_SIZE_A ((A_SIZE)/(FILT_A_NUM))

#define L0_POOL_FACTOR 2 // NOTE 2 = 2x2 pooling, set pool factors to zero to skip pooling step
#define L1_POOL_FACTOR 2
#define L2_POOL_FACTOR 2
#define L0_POOL_SIZE ((A_SIZE)/(L0_POOL_FACTOR))
#define L0_POOL_Y ((H2A)/(L0_POOL_FACTOR))
#define L0_POOL_X ((W2A)/(L0_POOL_FACTOR))
#define L1_POOL_SIZE ((B_SIZE)/(L1_POOL_FACTOR))
#define L1_POOL_Y ((H2B)/(L1_POOL_FACTOR))
#define L1_POOL_X ((W2B)/(L1_POOL_FACTOR))
#define L2_POOL_SIZE ((FLAT_SIZE)/(L2_POOL_FACTOR))
#define L2_POOL_Y (H2C)/(L2_POOL_FACTOR)
#define L2_POOL_X (W2C)/(L2_POOL_FACTOR)

#endif // CONFIG_H_INCLUDED
