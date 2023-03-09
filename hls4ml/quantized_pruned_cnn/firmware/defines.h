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
#define OUT_HEIGHT_39 202
#define OUT_WIDTH_39 257
#define N_CHAN_39 2
#define OUT_HEIGHT_2 200
#define OUT_WIDTH_2 255
#define N_FILT_2 16
#define OUT_HEIGHT_40 202
#define OUT_WIDTH_40 257
#define N_CHAN_40 16
#define OUT_HEIGHT_5 200
#define OUT_WIDTH_5 255
#define N_FILT_5 16
#define OUT_HEIGHT_41 202
#define OUT_WIDTH_41 257
#define N_CHAN_41 16
#define OUT_HEIGHT_8 200
#define OUT_WIDTH_8 255
#define N_FILT_8 16
#define OUT_HEIGHT_42 201
#define OUT_WIDTH_42 257
#define N_CHAN_42 16
#define OUT_HEIGHT_13 100
#define OUT_WIDTH_13 128
#define N_FILT_13 32
#define OUT_HEIGHT_43 102
#define OUT_WIDTH_43 130
#define N_CHAN_43 32
#define OUT_HEIGHT_46 100
#define OUT_WIDTH_46 128
#define N_FILT_46 32
#define OUT_HEIGHT_16 100
#define OUT_WIDTH_16 128
#define N_FILT_16 32
#define OUT_HEIGHT_18 100
#define OUT_WIDTH_18 128
#define N_FILT_18 32
#define OUT_HEIGHT_44 101
#define OUT_WIDTH_44 129
#define N_CHAN_44 32
#define OUT_HEIGHT_24 50
#define OUT_WIDTH_24 64
#define N_FILT_24 64
#define OUT_HEIGHT_45 52
#define OUT_WIDTH_45 66
#define N_CHAN_45 64
#define OUT_HEIGHT_47 50
#define OUT_WIDTH_47 64
#define N_FILT_47 64
#define OUT_HEIGHT_27 50
#define OUT_WIDTH_27 64
#define N_FILT_27 64
#define OUT_HEIGHT_29 50
#define OUT_WIDTH_29 64
#define N_FILT_29 64
#define N_SIZE_1_35 204800
#define N_LAYER_36 2

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> model_default_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer39_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer2_t;
typedef ap_fixed<10,5> weight2_t;
typedef ap_fixed<10,5> bias2_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer3_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer4_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer48_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer40_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer5_t;
typedef ap_fixed<10,5> weight5_t;
typedef ap_fixed<10,5> bias5_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer6_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer7_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer41_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer8_t;
typedef ap_fixed<10,5> weight8_t;
typedef ap_fixed<10,5> bias8_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 16*1> layer9_t;
typedef nnet::array<ap_fixed<10,5,AP_RND,AP_SAT>, 16*1> layer10_t;
typedef ap_fixed<16,6> add_3_default_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer11_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer12_t;
typedef nnet::array<ap_fixed<16,6>, 16*1> layer49_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 16*1> layer42_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer13_t;
typedef ap_fixed<10,5> weight13_t;
typedef ap_fixed<10,5> bias13_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer14_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 32*1> layer15_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 32*1> layer43_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer46_t;
typedef ap_fixed<10,5> weight46_t;
typedef ap_fixed<10,5> bias46_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer17_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer18_t;
typedef ap_fixed<10,5> weight18_t;
typedef ap_fixed<10,5> bias18_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 32*1> layer19_t;
typedef nnet::array<ap_fixed<10,5,AP_RND,AP_SAT>, 32*1> layer20_t;
typedef nnet::array<ap_fixed<10,5,AP_RND,AP_SAT>, 32*1> layer21_t;
typedef ap_fixed<16,6> add_4_default_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer22_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 32*1> layer23_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer50_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 32*1> layer44_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer24_t;
typedef ap_fixed<10,5> weight24_t;
typedef ap_fixed<10,5> bias24_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer25_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 64*1> layer26_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 64*1> layer45_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer47_t;
typedef ap_fixed<10,5> weight47_t;
typedef ap_fixed<10,5> bias47_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer28_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer29_t;
typedef ap_fixed<10,5> weight29_t;
typedef ap_fixed<10,5> bias29_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 64*1> layer30_t;
typedef nnet::array<ap_fixed<10,5,AP_RND,AP_SAT>, 64*1> layer31_t;
typedef nnet::array<ap_fixed<10,5,AP_RND,AP_SAT>, 64*1> layer32_t;
typedef ap_fixed<16,6> add_5_default_t;
typedef nnet::array<ap_fixed<16,6>, 64*1> layer33_t;
typedef nnet::array<ap_ufixed<16,12,AP_RND,AP_SAT>, 64*1> layer34_t;
typedef nnet::array<ap_fixed<16,6>, 2*1> layer36_t;
typedef ap_fixed<10,5> weight36_t;
typedef ap_fixed<10,5> bias36_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 2*1> layer37_t;
typedef ap_fixed<16,6> sigmoid_default_t;
typedef nnet::array<ap_fixed<16,6,AP_RND,AP_SAT>, 2*1> result_t;

#endif
