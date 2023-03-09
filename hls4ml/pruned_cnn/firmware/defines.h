#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_int.h"
#include "ap_fixed.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 200
#define N_INPUT_2_1 255
#define N_INPUT_3_1 2
#define OUT_HEIGHT_40 202
#define OUT_WIDTH_40 257
#define N_CHAN_40 2
#define OUT_HEIGHT_2 200
#define OUT_WIDTH_2 255
#define N_FILT_2 16
#define OUT_HEIGHT_41 202
#define OUT_WIDTH_41 257
#define N_CHAN_41 16
#define OUT_HEIGHT_6 200
#define OUT_WIDTH_6 255
#define N_FILT_6 16
#define OUT_HEIGHT_42 202
#define OUT_WIDTH_42 257
#define N_CHAN_42 16
#define OUT_HEIGHT_10 200
#define OUT_WIDTH_10 255
#define N_FILT_10 16
#define OUT_HEIGHT_43 201
#define OUT_WIDTH_43 257
#define N_CHAN_43 16
#define OUT_HEIGHT_15 100
#define OUT_WIDTH_15 128
#define N_FILT_15 32
#define OUT_HEIGHT_44 102
#define OUT_WIDTH_44 130
#define N_CHAN_44 32
#define OUT_HEIGHT_19 100
#define OUT_WIDTH_19 128
#define N_FILT_19 32
#define OUT_HEIGHT_47 100
#define OUT_WIDTH_47 128
#define N_FILT_47 32
#define OUT_HEIGHT_21 100
#define OUT_WIDTH_21 128
#define N_FILT_21 32
#define OUT_HEIGHT_45 101
#define OUT_WIDTH_45 129
#define N_CHAN_45 32
#define OUT_HEIGHT_26 50
#define OUT_WIDTH_26 64
#define N_FILT_26 64
#define OUT_HEIGHT_46 52
#define OUT_WIDTH_46 66
#define N_CHAN_46 64
#define OUT_HEIGHT_30 50
#define OUT_WIDTH_30 64
#define N_FILT_30 64
#define OUT_HEIGHT_48 50
#define OUT_WIDTH_48 64
#define N_FILT_48 64
#define OUT_HEIGHT_32 50
#define OUT_WIDTH_32 64
#define N_FILT_32 64
#define N_SIZE_1_37 204800
#define N_LAYER_38 2

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer40_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer2_t;
typedef ap_fixed<16,6> conv2d_48_weight_t;
typedef ap_fixed<16,6> conv2d_48_bias_t;
typedef ap_fixed<16,6> conv2d_48_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer3_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer4_t;
typedef ap_fixed<16,6> batch_normalization_208_scale_t;
typedef ap_fixed<16,6> batch_normalization_208_bias_t;
typedef ap_fixed<16,6> activation_14_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer5_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer49_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer41_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer6_t;
typedef ap_fixed<16,6> conv2d_49_weight_t;
typedef ap_fixed<16,6> conv2d_49_bias_t;
typedef ap_fixed<16,6> conv2d_49_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer7_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer8_t;
typedef ap_fixed<16,6> batch_normalization_209_scale_t;
typedef ap_fixed<16,6> batch_normalization_209_bias_t;
typedef ap_fixed<16,6> activation_15_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer9_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer42_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer10_t;
typedef ap_fixed<16,6> conv2d_50_weight_t;
typedef ap_fixed<16,6> conv2d_50_bias_t;
typedef ap_fixed<16,6> conv2d_50_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer11_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer12_t;
typedef ap_fixed<16,6> batch_normalization_210_scale_t;
typedef ap_fixed<16,6> batch_normalization_210_bias_t;
typedef ap_fixed<16,6> add_78_default_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer13_t;
typedef ap_fixed<16,6> activation_16_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer14_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer50_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer43_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer15_t;
typedef ap_fixed<16,6> conv2d_51_weight_t;
typedef ap_fixed<16,6> conv2d_51_bias_t;
typedef ap_fixed<16,6> conv2d_51_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer16_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer17_t;
typedef ap_fixed<16,6> batch_normalization_211_scale_t;
typedef ap_fixed<16,6> batch_normalization_211_bias_t;
typedef ap_fixed<16,6> activation_17_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer18_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer44_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer19_t;
typedef ap_fixed<16,6> conv2d_52_weight_t;
typedef ap_fixed<16,6> conv2d_52_bias_t;
typedef ap_fixed<16,6> conv2d_52_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer20_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer47_t;
typedef ap_fixed<16,6> conv2d_53_weight_t;
typedef ap_fixed<16,6> conv2d_53_bias_t;
typedef ap_fixed<16,6> conv2d_53_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer22_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer23_t;
typedef ap_fixed<16,6> batch_normalization_212_scale_t;
typedef ap_fixed<16,6> batch_normalization_212_bias_t;
typedef ap_fixed<16,6> add_79_default_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer24_t;
typedef ap_fixed<16,6> activation_18_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer25_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer51_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer45_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer26_t;
typedef ap_fixed<16,6> conv2d_54_weight_t;
typedef ap_fixed<16,6> conv2d_54_bias_t;
typedef ap_fixed<16,6> conv2d_54_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer27_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer28_t;
typedef ap_fixed<16,6> batch_normalization_213_scale_t;
typedef ap_fixed<16,6> batch_normalization_213_bias_t;
typedef ap_fixed<16,6> activation_19_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer29_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer46_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer30_t;
typedef ap_fixed<16,6> conv2d_55_weight_t;
typedef ap_fixed<16,6> conv2d_55_bias_t;
typedef ap_fixed<16,6> conv2d_55_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer31_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer48_t;
typedef ap_fixed<16,6> conv2d_56_weight_t;
typedef ap_fixed<16,6> conv2d_56_bias_t;
typedef ap_fixed<16,6> conv2d_56_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer33_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer34_t;
typedef ap_fixed<16,6> batch_normalization_214_scale_t;
typedef ap_fixed<16,6> batch_normalization_214_bias_t;
typedef ap_fixed<16,6> add_80_default_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer35_t;
typedef ap_fixed<16,6> activation_20_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer36_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer38_t;
typedef ap_fixed<16,6> dense_11_weight_t;
typedef ap_fixed<16,6> dense_11_bias_t;
typedef ap_fixed<16,6> dense_11_linear_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 2*1> result_t;

#endif
