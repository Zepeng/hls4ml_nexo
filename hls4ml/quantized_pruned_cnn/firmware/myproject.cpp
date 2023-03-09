//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    hls::stream<input_t> &input_2,
    hls::stream<result_t> &layer38_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_2,layer38_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_36;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight2_t, 288>(w2, "w2.txt");
        nnet::load_weights_from_txt<bias2_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<weight5_t, 2304>(w5, "w5.txt");
        nnet::load_weights_from_txt<bias5_t, 16>(b5, "b5.txt");
        nnet::load_weights_from_txt<weight8_t, 2304>(w8, "w8.txt");
        nnet::load_weights_from_txt<bias8_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<weight13_t, 4608>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 32>(b13, "b13.txt");
        nnet::load_weights_from_txt<weight46_t, 512>(w46, "w46.txt");
        nnet::load_weights_from_txt<bias46_t, 32>(b46, "b46.txt");
        nnet::load_weights_from_txt<weight18_t, 9216>(w18, "w18.txt");
        nnet::load_weights_from_txt<bias18_t, 32>(b18, "b18.txt");
        nnet::load_weights_from_txt<weight24_t, 18432>(w24, "w24.txt");
        nnet::load_weights_from_txt<bias24_t, 64>(b24, "b24.txt");
        nnet::load_weights_from_txt<weight47_t, 2048>(w47, "w47.txt");
        nnet::load_weights_from_txt<bias47_t, 64>(b47, "b47.txt");
        nnet::load_weights_from_txt<weight29_t, 36864>(w29, "w29.txt");
        nnet::load_weights_from_txt<bias29_t, 64>(b29, "b29.txt");
        nnet::load_weights_from_txt<weight36_t, 409600>(w36, "w36.txt");
        nnet::load_weights_from_txt<bias36_t, 2>(b36, "b36.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer39_t> layer39_out("layer39_out");
    #pragma HLS STREAM variable=layer39_out depth=51914
    nnet::zeropad2d_cl<input_t, layer39_t, config39>(input_2, layer39_out); // zp2d_q_conv2d_batchnorm

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=51000
    nnet::conv_2d_cl<layer39_t, layer2_t, config2>(layer39_out, layer2_out, w2, b2); // q_conv2d_batchnorm

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=51000
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // q_conv2d_batchnorm_linear

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=51000
    nnet::relu<layer3_t, layer4_t, relu_config4>(layer3_out, layer4_out); // q_activation

    hls::stream<layer48_t> layer48_cpy1("layer48_cpy1");
    #pragma HLS STREAM variable=layer48_cpy1 depth=51000
    hls::stream<layer48_t> layer48_cpy2("layer48_cpy2");
    #pragma HLS STREAM variable=layer48_cpy2 depth=51000
    nnet::clone_stream<layer4_t, layer48_t, 816000>(layer4_out, layer48_cpy1, layer48_cpy2); // clone_q_activation

    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=51914
    nnet::zeropad2d_cl<layer48_t, layer40_t, config40>(layer48_cpy1, layer40_out); // zp2d_q_conv2d_batchnorm_1

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=51000
    nnet::conv_2d_cl<layer40_t, layer5_t, config5>(layer40_out, layer5_out, w5, b5); // q_conv2d_batchnorm_1

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=51000
    nnet::linear<layer5_t, layer6_t, linear_config6>(layer5_out, layer6_out); // q_conv2d_batchnorm_1_linear

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=51000
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out); // q_activation_1

    hls::stream<layer41_t> layer41_out("layer41_out");
    #pragma HLS STREAM variable=layer41_out depth=51914
    nnet::zeropad2d_cl<layer7_t, layer41_t, config41>(layer7_out, layer41_out); // zp2d_q_conv2d_batchnorm_2

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=51000
    nnet::conv_2d_cl<layer41_t, layer8_t, config8>(layer41_out, layer8_out, w8, b8); // q_conv2d_batchnorm_2

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=51000
    nnet::linear<layer8_t, layer9_t, linear_config9>(layer8_out, layer9_out); // q_conv2d_batchnorm_2_linear

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=51000
    nnet::linear<layer9_t, layer10_t, linear_config10>(layer9_out, layer10_out); // q_activation_2

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=51000
    nnet::add<layer48_t, layer10_t, layer11_t, config11>(layer48_cpy2, layer10_out, layer11_out); // add_3

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=51000
    nnet::relu<layer11_t, layer12_t, relu_config12>(layer11_out, layer12_out); // q_activation_3

    hls::stream<layer49_t> layer49_cpy1("layer49_cpy1");
    #pragma HLS STREAM variable=layer49_cpy1 depth=51000
    hls::stream<layer49_t> layer49_cpy2("layer49_cpy2");
    #pragma HLS STREAM variable=layer49_cpy2 depth=51000
    nnet::clone_stream<layer12_t, layer49_t, 816000>(layer12_out, layer49_cpy1, layer49_cpy2); // clone_q_activation_3

    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS STREAM variable=layer42_out depth=51657
    nnet::zeropad2d_cl<layer49_t, layer42_t, config42>(layer49_cpy1, layer42_out); // zp2d_q_conv2d_batchnorm_3

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=12800
    nnet::conv_2d_cl<layer42_t, layer13_t, config13>(layer42_out, layer13_out, w13, b13); // q_conv2d_batchnorm_3

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=12800
    nnet::linear<layer13_t, layer14_t, linear_config14>(layer13_out, layer14_out); // q_conv2d_batchnorm_3_linear

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=12800
    nnet::relu<layer14_t, layer15_t, relu_config15>(layer14_out, layer15_out); // q_activation_4

    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=13260
    nnet::zeropad2d_cl<layer15_t, layer43_t, config43>(layer15_out, layer43_out); // zp2d_q_conv2d_batchnorm_4

    hls::stream<layer46_t> layer46_out("layer46_out");
    #pragma HLS STREAM variable=layer46_out depth=12800
    nnet::pointwise_conv_2d_cl<layer49_t, layer46_t, config46>(layer49_cpy2, layer46_out, w46, b46); // q_conv2d

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=12800
    nnet::linear<layer46_t, layer17_t, linear_config17>(layer46_out, layer17_out); // q_conv2d_linear

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=12800
    nnet::conv_2d_cl<layer43_t, layer18_t, config18>(layer43_out, layer18_out, w18, b18); // q_conv2d_batchnorm_4

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=12800
    nnet::linear<layer18_t, layer19_t, linear_config19>(layer18_out, layer19_out); // q_conv2d_batchnorm_4_linear

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=12800
    nnet::linear<layer17_t, layer20_t, linear_config20>(layer17_out, layer20_out); // q_activation_5

    hls::stream<layer21_t> layer21_out("layer21_out");
    #pragma HLS STREAM variable=layer21_out depth=12800
    nnet::linear<layer19_t, layer21_t, linear_config21>(layer19_out, layer21_out); // q_activation_6

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=12800
    nnet::add<layer20_t, layer21_t, layer22_t, config22>(layer20_out, layer21_out, layer22_out); // add_4

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=12800
    nnet::relu<layer22_t, layer23_t, relu_config23>(layer22_out, layer23_out); // q_activation_7

    hls::stream<layer50_t> layer50_cpy1("layer50_cpy1");
    #pragma HLS STREAM variable=layer50_cpy1 depth=12800
    hls::stream<layer50_t> layer50_cpy2("layer50_cpy2");
    #pragma HLS STREAM variable=layer50_cpy2 depth=12800
    nnet::clone_stream<layer23_t, layer50_t, 409600>(layer23_out, layer50_cpy1, layer50_cpy2); // clone_q_activation_7

    hls::stream<layer44_t> layer44_out("layer44_out");
    #pragma HLS STREAM variable=layer44_out depth=13029
    nnet::zeropad2d_cl<layer50_t, layer44_t, config44>(layer50_cpy1, layer44_out); // zp2d_q_conv2d_batchnorm_5

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=3200
    nnet::conv_2d_cl<layer44_t, layer24_t, config24>(layer44_out, layer24_out, w24, b24); // q_conv2d_batchnorm_5

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=3200
    nnet::linear<layer24_t, layer25_t, linear_config25>(layer24_out, layer25_out); // q_conv2d_batchnorm_5_linear

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=3200
    nnet::relu<layer25_t, layer26_t, relu_config26>(layer25_out, layer26_out); // q_activation_8

    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS STREAM variable=layer45_out depth=3432
    nnet::zeropad2d_cl<layer26_t, layer45_t, config45>(layer26_out, layer45_out); // zp2d_q_conv2d_batchnorm_6

    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS STREAM variable=layer47_out depth=3200
    nnet::pointwise_conv_2d_cl<layer50_t, layer47_t, config47>(layer50_cpy2, layer47_out, w47, b47); // q_conv2d_1

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=3200
    nnet::linear<layer47_t, layer28_t, linear_config28>(layer47_out, layer28_out); // q_conv2d_1_linear

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=3200
    nnet::conv_2d_cl<layer45_t, layer29_t, config29>(layer45_out, layer29_out, w29, b29); // q_conv2d_batchnorm_6

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=3200
    nnet::linear<layer29_t, layer30_t, linear_config30>(layer29_out, layer30_out); // q_conv2d_batchnorm_6_linear

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=3200
    nnet::linear<layer28_t, layer31_t, linear_config31>(layer28_out, layer31_out); // q_activation_9

    hls::stream<layer32_t> layer32_out("layer32_out");
    #pragma HLS STREAM variable=layer32_out depth=3200
    nnet::linear<layer30_t, layer32_t, linear_config32>(layer30_out, layer32_out); // q_activation_10

    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS STREAM variable=layer33_out depth=3200
    nnet::add<layer31_t, layer32_t, layer33_t, config33>(layer31_out, layer32_out, layer33_out); // add_5

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=3200
    nnet::relu<layer33_t, layer34_t, relu_config34>(layer33_out, layer34_out); // q_activation_11

    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=1
    nnet::dense<layer34_t, layer36_t, config36>(layer34_out, layer36_out, w36, b36); // q_dense

    hls::stream<layer37_t> layer37_out("layer37_out");
    #pragma HLS STREAM variable=layer37_out depth=1
    nnet::linear<layer36_t, layer37_t, linear_config37>(layer36_out, layer37_out); // q_dense_linear

    nnet::sigmoid<layer37_t, result_t, sigmoid_config38>(layer37_out, layer38_out); // sigmoid

}
