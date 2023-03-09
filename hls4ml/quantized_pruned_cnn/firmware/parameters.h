#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_conv2d.h"
#include "nnet_utils/nnet_conv2d_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"
#include "nnet_utils/nnet_merge.h"
#include "nnet_utils/nnet_merge_stream.h"
#include "nnet_utils/nnet_padding.h"
#include "nnet_utils/nnet_padding_stream.h"
#include "nnet_utils/nnet_sepconv2d_stream.h"
#include "nnet_utils/nnet_stream.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w13.h"
#include "weights/b13.h"
#include "weights/w46.h"
#include "weights/b46.h"
#include "weights/w18.h"
#include "weights/b18.h"
#include "weights/w24.h"
#include "weights/b24.h"
#include "weights/w47.h"
#include "weights/b47.h"
#include "weights/w29.h"
#include "weights/b29.h"
#include "weights/w36.h"
#include "weights/b36.h"

//hls-fpga-machine-learning insert layer-config
// zp2d_q_conv2d_batchnorm
struct config39 : nnet::padding2d_config {
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1;
    static const unsigned out_height = OUT_HEIGHT_39;
    static const unsigned out_width = OUT_WIDTH_39;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 18;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config2 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_39;
    static const unsigned in_width = OUT_WIDTH_39;
    static const unsigned n_chan = N_CHAN_39;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 13;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef config2_mult mult_config;
};
const ap_uint<config2::filt_height * config2::filt_width> config2::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_linear
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_1
struct config40 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned out_height = OUT_HEIGHT_40;
    static const unsigned out_width = OUT_WIDTH_40;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_1
struct config5_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config5 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_40;
    static const unsigned in_width = OUT_WIDTH_40;
    static const unsigned n_chan = N_CHAN_40;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_5;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_5;
    static const unsigned out_width = OUT_WIDTH_5;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 250;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef config5_mult mult_config;
};
const ap_uint<config5::filt_height * config5::filt_width> config5::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_1_linear
struct linear_config6 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_1
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_5*OUT_WIDTH_5*N_FILT_5;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_2
struct config41 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_5;
    static const unsigned in_width = OUT_WIDTH_5;
    static const unsigned n_chan = N_FILT_5;
    static const unsigned out_height = OUT_HEIGHT_41;
    static const unsigned out_width = OUT_WIDTH_41;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_2
struct config8_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config8 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_41;
    static const unsigned in_width = OUT_WIDTH_41;
    static const unsigned n_chan = N_CHAN_41;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_8;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_8;
    static const unsigned out_width = OUT_WIDTH_8;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 263;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef config8_mult mult_config;
};
const ap_uint<config8::filt_height * config8::filt_width> config8::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_2_linear
struct linear_config9 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_2
struct linear_config10 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_8*OUT_WIDTH_8*N_FILT_8;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// add_3
struct config11 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
};

// q_activation_3
struct relu_config12 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_3
struct config42 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned out_height = OUT_HEIGHT_42;
    static const unsigned out_width = OUT_WIDTH_42;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_3
struct config13_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias13_t bias_t;
    typedef weight13_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config13 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_42;
    static const unsigned in_width = OUT_WIDTH_42;
    static const unsigned n_chan = N_CHAN_42;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_13;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_13;
    static const unsigned out_width = OUT_WIDTH_13;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 723;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias13_t bias_t;
    typedef weight13_t weight_t;
    typedef config13_mult mult_config;
};
const ap_uint<config13::filt_height * config13::filt_width> config13::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// q_conv2d_batchnorm_3_linear
struct linear_config14 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_13*OUT_WIDTH_13*N_FILT_13;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_4
struct relu_config15 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_13*OUT_WIDTH_13*N_FILT_13;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_4
struct config43 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_13;
    static const unsigned in_width = OUT_WIDTH_13;
    static const unsigned n_chan = N_FILT_13;
    static const unsigned out_height = OUT_HEIGHT_43;
    static const unsigned out_width = OUT_WIDTH_43;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d
struct config46_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias46_t bias_t;
    typedef weight46_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config46 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_46;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_46;
    static const unsigned out_width = OUT_WIDTH_46;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 26;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 3;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias46_t bias_t;
    typedef weight46_t weight_t;
    typedef config46_mult mult_config;
};
const ap_uint<config46::filt_height * config46::filt_width> config46::pixels[] = {1,0,1,0,0,0,1,0,1,0,0,0};

// q_conv2d_linear
struct linear_config17 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_46*OUT_WIDTH_46*N_FILT_46;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_conv2d_batchnorm_4
struct config18_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config18 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_43;
    static const unsigned in_width = OUT_WIDTH_43;
    static const unsigned n_chan = N_CHAN_43;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_18;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_18;
    static const unsigned out_width = OUT_WIDTH_18;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 2216;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias18_t bias_t;
    typedef weight18_t weight_t;
    typedef config18_mult mult_config;
};
const ap_uint<config18::filt_height * config18::filt_width> config18::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_4_linear
struct linear_config19 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_18*OUT_WIDTH_18*N_FILT_18;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_5
struct linear_config20 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_16*OUT_WIDTH_16*N_FILT_16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_6
struct linear_config21 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_18*OUT_WIDTH_18*N_FILT_18;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// add_4
struct config22 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_16*OUT_WIDTH_16*N_FILT_16;
};

// q_activation_7
struct relu_config23 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_16*OUT_WIDTH_16*N_FILT_16;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_5
struct config44 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_16;
    static const unsigned in_width = OUT_WIDTH_16;
    static const unsigned n_chan = N_FILT_16;
    static const unsigned out_height = OUT_HEIGHT_44;
    static const unsigned out_width = OUT_WIDTH_44;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 1;
};

// q_conv2d_batchnorm_5
struct config24_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias24_t bias_t;
    typedef weight24_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config24 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_44;
    static const unsigned in_width = OUT_WIDTH_44;
    static const unsigned n_chan = N_CHAN_44;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_24;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_24;
    static const unsigned out_width = OUT_WIDTH_24;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 9603;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias24_t bias_t;
    typedef weight24_t weight_t;
    typedef config24_mult mult_config;
};
const ap_uint<config24::filt_height * config24::filt_width> config24::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// q_conv2d_batchnorm_5_linear
struct linear_config25 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_24*OUT_WIDTH_24*N_FILT_24;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_8
struct relu_config26 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_24*OUT_WIDTH_24*N_FILT_24;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_q_conv2d_batchnorm_6
struct config45 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_24;
    static const unsigned in_width = OUT_WIDTH_24;
    static const unsigned n_chan = N_FILT_24;
    static const unsigned out_height = OUT_HEIGHT_45;
    static const unsigned out_width = OUT_WIDTH_45;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// q_conv2d_1
struct config47_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias47_t bias_t;
    typedef weight47_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config47 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_16;
    static const unsigned in_width = OUT_WIDTH_16;
    static const unsigned n_chan = N_FILT_16;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_47;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_47;
    static const unsigned out_width = OUT_WIDTH_47;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 245;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias47_t bias_t;
    typedef weight47_t weight_t;
    typedef config47_mult mult_config;
};
const ap_uint<config47::filt_height * config47::filt_width> config47::pixels[] = {1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0};

// q_conv2d_1_linear
struct linear_config28 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_47*OUT_WIDTH_47*N_FILT_47;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_conv2d_batchnorm_6
struct config29_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config29 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_45;
    static const unsigned in_width = OUT_WIDTH_45;
    static const unsigned n_chan = N_CHAN_45;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_29;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_29;
    static const unsigned out_width = OUT_WIDTH_29;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 20988;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef bias29_t bias_t;
    typedef weight29_t weight_t;
    typedef config29_mult mult_config;
};
const ap_uint<config29::filt_height * config29::filt_width> config29::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// q_conv2d_batchnorm_6_linear
struct linear_config30 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_29*OUT_WIDTH_29*N_FILT_29;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_9
struct linear_config31 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_27*OUT_WIDTH_27*N_FILT_27;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_activation_10
struct linear_config32 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_29*OUT_WIDTH_29*N_FILT_29;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// add_5
struct config33 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_27*OUT_WIDTH_27*N_FILT_27;
};

// q_activation_11
struct relu_config34 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_27*OUT_WIDTH_27*N_FILT_27;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// q_dense
struct config36 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_35;
    static const unsigned n_out = N_LAYER_36;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 20431;
    static const unsigned n_nonzeros = 389169;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef bias36_t bias_t;
    typedef weight36_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// q_dense_linear
struct linear_config37 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_36;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// sigmoid
struct sigmoid_config38 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_36;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
