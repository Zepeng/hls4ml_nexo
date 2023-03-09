#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_batchnorm.h"
#include "nnet_utils/nnet_batchnorm_stream.h"
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
#include "weights/s4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/s8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"
#include "weights/s12.h"
#include "weights/b12.h"
#include "weights/w15.h"
#include "weights/b15.h"
#include "weights/s17.h"
#include "weights/b17.h"
#include "weights/w19.h"
#include "weights/b19.h"
#include "weights/w47.h"
#include "weights/b47.h"
#include "weights/s23.h"
#include "weights/b23.h"
#include "weights/w26.h"
#include "weights/b26.h"
#include "weights/s28.h"
#include "weights/b28.h"
#include "weights/w30.h"
#include "weights/b30.h"
#include "weights/w48.h"
#include "weights/b48.h"
#include "weights/s34.h"
#include "weights/b34.h"
#include "weights/w38.h"
#include "weights/b38.h"

//hls-fpga-machine-learning insert layer-config
// zp2d_conv2d_48
struct config40 : nnet::padding2d_config {
    static const unsigned in_height = N_INPUT_1_1;
    static const unsigned in_width = N_INPUT_2_1;
    static const unsigned n_chan = N_INPUT_3_1;
    static const unsigned out_height = OUT_HEIGHT_40;
    static const unsigned out_width = OUT_WIDTH_40;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_48
struct config2_mult : nnet::dense_config {
    static const unsigned n_in = 18;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_48_bias_t bias_t;
    typedef conv2d_48_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config2 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_2;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_2;
    static const unsigned out_width = OUT_WIDTH_2;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_48_bias_t bias_t;
    typedef conv2d_48_weight_t weight_t;
    typedef config2_mult mult_config;
};
const ap_uint<config2::filt_height * config2::filt_width> config2::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// conv2d_48_linear
struct linear_config3 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_208
struct config4 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned n_filt = 16;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_208_bias_t bias_t;
    typedef batch_normalization_208_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_14
struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_49
struct config41 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned out_height = OUT_HEIGHT_41;
    static const unsigned out_width = OUT_WIDTH_41;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_49
struct config6_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_49_bias_t bias_t;
    typedef conv2d_49_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config6 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_6;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_6;
    static const unsigned out_width = OUT_WIDTH_6;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_49_bias_t bias_t;
    typedef conv2d_49_weight_t weight_t;
    typedef config6_mult mult_config;
};
const ap_uint<config6::filt_height * config6::filt_width> config6::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// conv2d_49_linear
struct linear_config7 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_209
struct config8 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6;
    static const unsigned n_filt = 16;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_209_bias_t bias_t;
    typedef batch_normalization_209_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_15
struct relu_config9 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_50
struct config42 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_6;
    static const unsigned in_width = OUT_WIDTH_6;
    static const unsigned n_chan = N_FILT_6;
    static const unsigned out_height = OUT_HEIGHT_42;
    static const unsigned out_width = OUT_WIDTH_42;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_50
struct config10_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 16;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_50_bias_t bias_t;
    typedef conv2d_50_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config10 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_10;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_10;
    static const unsigned out_width = OUT_WIDTH_10;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_50_bias_t bias_t;
    typedef conv2d_50_weight_t weight_t;
    typedef config10_mult mult_config;
};
const ap_uint<config10::filt_height * config10::filt_width> config10::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// conv2d_50_linear
struct linear_config11 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_10*OUT_WIDTH_10*N_FILT_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_210
struct config12 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_10*OUT_WIDTH_10*N_FILT_10;
    static const unsigned n_filt = 16;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_210_bias_t bias_t;
    typedef batch_normalization_210_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// add_78
struct config13 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
};

// activation_16
struct relu_config14 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_2*OUT_WIDTH_2*N_FILT_2;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_51
struct config43 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_2;
    static const unsigned in_width = OUT_WIDTH_2;
    static const unsigned n_chan = N_FILT_2;
    static const unsigned out_height = OUT_HEIGHT_43;
    static const unsigned out_width = OUT_WIDTH_43;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_51
struct config15_mult : nnet::dense_config {
    static const unsigned n_in = 144;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_51_bias_t bias_t;
    typedef conv2d_51_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config15 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_15;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_15;
    static const unsigned out_width = OUT_WIDTH_15;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_51_bias_t bias_t;
    typedef conv2d_51_weight_t weight_t;
    typedef config15_mult mult_config;
};
const ap_uint<config15::filt_height * config15::filt_width> config15::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// conv2d_51_linear
struct linear_config16 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_15*OUT_WIDTH_15*N_FILT_15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_211
struct config17 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_15*OUT_WIDTH_15*N_FILT_15;
    static const unsigned n_filt = 32;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_211_bias_t bias_t;
    typedef batch_normalization_211_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_17
struct relu_config18 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_15*OUT_WIDTH_15*N_FILT_15;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_52
struct config44 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_15;
    static const unsigned in_width = OUT_WIDTH_15;
    static const unsigned n_chan = N_FILT_15;
    static const unsigned out_height = OUT_HEIGHT_44;
    static const unsigned out_width = OUT_WIDTH_44;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_52
struct config19_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_52_bias_t bias_t;
    typedef conv2d_52_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config19 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_19;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_19;
    static const unsigned out_width = OUT_WIDTH_19;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_52_bias_t bias_t;
    typedef conv2d_52_weight_t weight_t;
    typedef config19_mult mult_config;
};
const ap_uint<config19::filt_height * config19::filt_width> config19::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// conv2d_52_linear
struct linear_config20 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_19*OUT_WIDTH_19*N_FILT_19;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// conv2d_53
struct config47_mult : nnet::dense_config {
    static const unsigned n_in = 16;
    static const unsigned n_out = 32;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_53_bias_t bias_t;
    typedef conv2d_53_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config47 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_47;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_47;
    static const unsigned out_width = OUT_WIDTH_47;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 3;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_53_bias_t bias_t;
    typedef conv2d_53_weight_t weight_t;
    typedef config47_mult mult_config;
};
const ap_uint<config47::filt_height * config47::filt_width> config47::pixels[] = {1,0,1,0,0,0,1,0,1,0,0,0};

// conv2d_53_linear
struct linear_config22 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_47*OUT_WIDTH_47*N_FILT_47;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_212
struct config23 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_19*OUT_WIDTH_19*N_FILT_19;
    static const unsigned n_filt = 32;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_212_bias_t bias_t;
    typedef batch_normalization_212_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// add_79
struct config24 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_21*OUT_WIDTH_21*N_FILT_21;
};

// activation_18
struct relu_config25 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_21*OUT_WIDTH_21*N_FILT_21;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_54
struct config45 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_21;
    static const unsigned in_width = OUT_WIDTH_21;
    static const unsigned n_chan = N_FILT_21;
    static const unsigned out_height = OUT_HEIGHT_45;
    static const unsigned out_width = OUT_WIDTH_45;
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 1;
};

// conv2d_54
struct config26_mult : nnet::dense_config {
    static const unsigned n_in = 288;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_54_bias_t bias_t;
    typedef conv2d_54_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config26 : nnet::conv2d_config {
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
    static const unsigned n_filt = N_FILT_26;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_26;
    static const unsigned out_width = OUT_WIDTH_26;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_54_bias_t bias_t;
    typedef conv2d_54_weight_t weight_t;
    typedef config26_mult mult_config;
};
const ap_uint<config26::filt_height * config26::filt_width> config26::pixels[] = {1,2,5,2,4,8,16,40,16,32,65,130,325,130,260,8,16,40,16,32,64,128,320,128,256};

// conv2d_54_linear
struct linear_config27 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_26*OUT_WIDTH_26*N_FILT_26;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_213
struct config28 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_26*OUT_WIDTH_26*N_FILT_26;
    static const unsigned n_filt = 64;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_213_bias_t bias_t;
    typedef batch_normalization_213_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// activation_19
struct relu_config29 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_26*OUT_WIDTH_26*N_FILT_26;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// zp2d_conv2d_55
struct config46 : nnet::padding2d_config {
    static const unsigned in_height = OUT_HEIGHT_26;
    static const unsigned in_width = OUT_WIDTH_26;
    static const unsigned n_chan = N_FILT_26;
    static const unsigned out_height = OUT_HEIGHT_46;
    static const unsigned out_width = OUT_WIDTH_46;
    static const unsigned pad_top = 1;
    static const unsigned pad_bottom = 1;
    static const unsigned pad_left = 1;
    static const unsigned pad_right = 1;
};

// conv2d_55
struct config30_mult : nnet::dense_config {
    static const unsigned n_in = 576;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_55_bias_t bias_t;
    typedef conv2d_55_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config30 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_46;
    static const unsigned in_width = OUT_WIDTH_46;
    static const unsigned n_chan = N_CHAN_46;
    static const unsigned filt_height = 3;
    static const unsigned filt_width = 3;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_30;
    static const unsigned stride_height = 1;
    static const unsigned stride_width = 1;
    static const unsigned out_height = OUT_HEIGHT_30;
    static const unsigned out_width = OUT_WIDTH_30;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 5;
    static const unsigned min_width = 5;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_55_bias_t bias_t;
    typedef conv2d_55_weight_t weight_t;
    typedef config30_mult mult_config;
};
const ap_uint<config30::filt_height * config30::filt_width> config30::pixels[] = {1,3,7,6,4,9,27,63,54,36,73,219,511,438,292,72,216,504,432,288,64,192,448,384,256};

// conv2d_55_linear
struct linear_config31 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_30*OUT_WIDTH_30*N_FILT_30;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// conv2d_56
struct config48_mult : nnet::dense_config {
    static const unsigned n_in = 32;
    static const unsigned n_out = 64;
    static const unsigned reuse_factor = 1;
    static const unsigned strategy = nnet::latency;
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_56_bias_t bias_t;
    typedef conv2d_56_weight_t weight_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

struct config48 : nnet::conv2d_config {
    static const unsigned pad_top = 0;
    static const unsigned pad_bottom = 0;
    static const unsigned pad_left = 0;
    static const unsigned pad_right = 0;
    static const unsigned in_height = OUT_HEIGHT_21;
    static const unsigned in_width = OUT_WIDTH_21;
    static const unsigned n_chan = N_FILT_21;
    static const unsigned filt_height = 1;
    static const unsigned filt_width = 1;
    static const unsigned kernel_size = filt_height * filt_width;
    static const unsigned n_filt = N_FILT_48;
    static const unsigned stride_height = 2;
    static const unsigned stride_width = 2;
    static const unsigned out_height = OUT_HEIGHT_48;
    static const unsigned out_width = OUT_WIDTH_48;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const bool store_weights_in_bram = false;
    static const unsigned strategy = nnet::latency;
    static const nnet::conv_implementation implementation = nnet::conv_implementation::linebuffer;
    static const unsigned min_height = 4;
    static const unsigned min_width = 4;
    static const ap_uint<filt_height * filt_width> pixels[min_height * min_width];
    typedef ap_fixed<16,6> accum_t;
    typedef conv2d_56_bias_t bias_t;
    typedef conv2d_56_weight_t weight_t;
    typedef config48_mult mult_config;
};
const ap_uint<config48::filt_height * config48::filt_width> config48::pixels[] = {1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0};

// conv2d_56_linear
struct linear_config33 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_48*OUT_WIDTH_48*N_FILT_48;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// batch_normalization_214
struct config34 : nnet::batchnorm_config {
    static const unsigned n_in = OUT_HEIGHT_30*OUT_WIDTH_30*N_FILT_30;
    static const unsigned n_filt = 64;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    static const bool store_weights_in_bram = false;
    typedef batch_normalization_214_bias_t bias_t;
    typedef batch_normalization_214_scale_t scale_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// add_80
struct config35 : nnet::merge_config {
    static const unsigned n_elem = OUT_HEIGHT_32*OUT_WIDTH_32*N_FILT_32;
};

// activation_20
struct relu_config36 : nnet::activ_config {
    static const unsigned n_in = OUT_HEIGHT_32*OUT_WIDTH_32*N_FILT_32;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

// dense_11
struct config38 : nnet::dense_config {
    static const unsigned n_in = N_SIZE_1_37;
    static const unsigned n_out = N_LAYER_38;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 409600;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<16,6> accum_t;
    typedef dense_11_bias_t bias_t;
    typedef dense_11_weight_t weight_t;
    typedef ap_uint<1> index_t;
    template<class x_T, class y_T, class res_T>
    using product = nnet::product::mult<x_T, y_T, res_T>;
};

// dense_11_linear
struct linear_config39 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_38;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_stream;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
