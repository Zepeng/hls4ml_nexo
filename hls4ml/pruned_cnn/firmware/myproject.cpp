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
    hls::stream<input_t> &input_14,
    hls::stream<result_t> &layer39_out,
    unsigned short &const_size_in_1,
    unsigned short &const_size_out_1
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=input_14,layer39_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1*N_INPUT_3_1;
    const_size_out_1 = N_LAYER_38;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<conv2d_48_weight_t, 288>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv2d_48_bias_t, 16>(b2, "b2.txt");
        nnet::load_weights_from_txt<batch_normalization_208_scale_t, 16>(s4, "s4.txt");
        nnet::load_weights_from_txt<batch_normalization_208_bias_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<conv2d_49_weight_t, 2304>(w6, "w6.txt");
        nnet::load_weights_from_txt<conv2d_49_bias_t, 16>(b6, "b6.txt");
        nnet::load_weights_from_txt<batch_normalization_209_scale_t, 16>(s8, "s8.txt");
        nnet::load_weights_from_txt<batch_normalization_209_bias_t, 16>(b8, "b8.txt");
        nnet::load_weights_from_txt<conv2d_50_weight_t, 2304>(w10, "w10.txt");
        nnet::load_weights_from_txt<conv2d_50_bias_t, 16>(b10, "b10.txt");
        nnet::load_weights_from_txt<batch_normalization_210_scale_t, 16>(s12, "s12.txt");
        nnet::load_weights_from_txt<batch_normalization_210_bias_t, 16>(b12, "b12.txt");
        nnet::load_weights_from_txt<conv2d_51_weight_t, 4608>(w15, "w15.txt");
        nnet::load_weights_from_txt<conv2d_51_bias_t, 32>(b15, "b15.txt");
        nnet::load_weights_from_txt<batch_normalization_211_scale_t, 32>(s17, "s17.txt");
        nnet::load_weights_from_txt<batch_normalization_211_bias_t, 32>(b17, "b17.txt");
        nnet::load_weights_from_txt<conv2d_52_weight_t, 9216>(w19, "w19.txt");
        nnet::load_weights_from_txt<conv2d_52_bias_t, 32>(b19, "b19.txt");
        nnet::load_weights_from_txt<conv2d_53_weight_t, 512>(w47, "w47.txt");
        nnet::load_weights_from_txt<conv2d_53_bias_t, 32>(b47, "b47.txt");
        nnet::load_weights_from_txt<batch_normalization_212_scale_t, 32>(s23, "s23.txt");
        nnet::load_weights_from_txt<batch_normalization_212_bias_t, 32>(b23, "b23.txt");
        nnet::load_weights_from_txt<conv2d_54_weight_t, 18432>(w26, "w26.txt");
        nnet::load_weights_from_txt<conv2d_54_bias_t, 64>(b26, "b26.txt");
        nnet::load_weights_from_txt<batch_normalization_213_scale_t, 64>(s28, "s28.txt");
        nnet::load_weights_from_txt<batch_normalization_213_bias_t, 64>(b28, "b28.txt");
        nnet::load_weights_from_txt<conv2d_55_weight_t, 36864>(w30, "w30.txt");
        nnet::load_weights_from_txt<conv2d_55_bias_t, 64>(b30, "b30.txt");
        nnet::load_weights_from_txt<conv2d_56_weight_t, 2048>(w48, "w48.txt");
        nnet::load_weights_from_txt<conv2d_56_bias_t, 64>(b48, "b48.txt");
        nnet::load_weights_from_txt<batch_normalization_214_scale_t, 64>(s34, "s34.txt");
        nnet::load_weights_from_txt<batch_normalization_214_bias_t, 64>(b34, "b34.txt");
        nnet::load_weights_from_txt<dense_11_weight_t, 409600>(w38, "w38.txt");
        nnet::load_weights_from_txt<dense_11_bias_t, 2>(b38, "b38.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    hls::stream<layer40_t> layer40_out("layer40_out");
    #pragma HLS STREAM variable=layer40_out depth=51914
    nnet::zeropad2d_cl<input_t, layer40_t, config40>(input_14, layer40_out); // zp2d_conv2d_48

    hls::stream<layer2_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=51000
    nnet::conv_2d_cl<layer40_t, layer2_t, config2>(layer40_out, layer2_out, w2, b2); // conv2d_48

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=51000
    nnet::linear<layer2_t, layer3_t, linear_config3>(layer2_out, layer3_out); // conv2d_48_linear

    hls::stream<layer4_t> layer4_out("layer4_out");
    #pragma HLS STREAM variable=layer4_out depth=51000
    nnet::normalize<layer3_t, layer4_t, config4>(layer3_out, layer4_out, s4, b4); // batch_normalization_208

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=51000
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out); // activation_14

    hls::stream<layer49_t> layer49_cpy1("layer49_cpy1");
    #pragma HLS STREAM variable=layer49_cpy1 depth=51000
    hls::stream<layer49_t> layer49_cpy2("layer49_cpy2");
    #pragma HLS STREAM variable=layer49_cpy2 depth=51000
    nnet::clone_stream<layer5_t, layer49_t, 816000>(layer5_out, layer49_cpy1, layer49_cpy2); // clone_activation_14

    hls::stream<layer41_t> layer41_out("layer41_out");
    #pragma HLS STREAM variable=layer41_out depth=51914
    nnet::zeropad2d_cl<layer49_t, layer41_t, config41>(layer49_cpy1, layer41_out); // zp2d_conv2d_49

    hls::stream<layer6_t> layer6_out("layer6_out");
    #pragma HLS STREAM variable=layer6_out depth=51000
    nnet::conv_2d_cl<layer41_t, layer6_t, config6>(layer41_out, layer6_out, w6, b6); // conv2d_49

    hls::stream<layer7_t> layer7_out("layer7_out");
    #pragma HLS STREAM variable=layer7_out depth=51000
    nnet::linear<layer6_t, layer7_t, linear_config7>(layer6_out, layer7_out); // conv2d_49_linear

    hls::stream<layer8_t> layer8_out("layer8_out");
    #pragma HLS STREAM variable=layer8_out depth=51000
    nnet::normalize<layer7_t, layer8_t, config8>(layer7_out, layer8_out, s8, b8); // batch_normalization_209

    hls::stream<layer9_t> layer9_out("layer9_out");
    #pragma HLS STREAM variable=layer9_out depth=51000
    nnet::relu<layer8_t, layer9_t, relu_config9>(layer8_out, layer9_out); // activation_15

    hls::stream<layer42_t> layer42_out("layer42_out");
    #pragma HLS STREAM variable=layer42_out depth=51914
    nnet::zeropad2d_cl<layer9_t, layer42_t, config42>(layer9_out, layer42_out); // zp2d_conv2d_50

    hls::stream<layer10_t> layer10_out("layer10_out");
    #pragma HLS STREAM variable=layer10_out depth=51000
    nnet::conv_2d_cl<layer42_t, layer10_t, config10>(layer42_out, layer10_out, w10, b10); // conv2d_50

    hls::stream<layer11_t> layer11_out("layer11_out");
    #pragma HLS STREAM variable=layer11_out depth=51000
    nnet::linear<layer10_t, layer11_t, linear_config11>(layer10_out, layer11_out); // conv2d_50_linear

    hls::stream<layer12_t> layer12_out("layer12_out");
    #pragma HLS STREAM variable=layer12_out depth=51000
    nnet::normalize<layer11_t, layer12_t, config12>(layer11_out, layer12_out, s12, b12); // batch_normalization_210

    hls::stream<layer13_t> layer13_out("layer13_out");
    #pragma HLS STREAM variable=layer13_out depth=51000
    nnet::add<layer49_t, layer12_t, layer13_t, config13>(layer49_cpy2, layer12_out, layer13_out); // add_78

    hls::stream<layer14_t> layer14_out("layer14_out");
    #pragma HLS STREAM variable=layer14_out depth=51000
    nnet::relu<layer13_t, layer14_t, relu_config14>(layer13_out, layer14_out); // activation_16

    hls::stream<layer50_t> layer50_cpy1("layer50_cpy1");
    #pragma HLS STREAM variable=layer50_cpy1 depth=51000
    hls::stream<layer50_t> layer50_cpy2("layer50_cpy2");
    #pragma HLS STREAM variable=layer50_cpy2 depth=51000
    nnet::clone_stream<layer14_t, layer50_t, 816000>(layer14_out, layer50_cpy1, layer50_cpy2); // clone_activation_16

    hls::stream<layer43_t> layer43_out("layer43_out");
    #pragma HLS STREAM variable=layer43_out depth=51657
    nnet::zeropad2d_cl<layer50_t, layer43_t, config43>(layer50_cpy1, layer43_out); // zp2d_conv2d_51

    hls::stream<layer15_t> layer15_out("layer15_out");
    #pragma HLS STREAM variable=layer15_out depth=12800
    nnet::conv_2d_cl<layer43_t, layer15_t, config15>(layer43_out, layer15_out, w15, b15); // conv2d_51

    hls::stream<layer16_t> layer16_out("layer16_out");
    #pragma HLS STREAM variable=layer16_out depth=12800
    nnet::linear<layer15_t, layer16_t, linear_config16>(layer15_out, layer16_out); // conv2d_51_linear

    hls::stream<layer17_t> layer17_out("layer17_out");
    #pragma HLS STREAM variable=layer17_out depth=12800
    nnet::normalize<layer16_t, layer17_t, config17>(layer16_out, layer17_out, s17, b17); // batch_normalization_211

    hls::stream<layer18_t> layer18_out("layer18_out");
    #pragma HLS STREAM variable=layer18_out depth=12800
    nnet::relu<layer17_t, layer18_t, relu_config18>(layer17_out, layer18_out); // activation_17

    hls::stream<layer44_t> layer44_out("layer44_out");
    #pragma HLS STREAM variable=layer44_out depth=13260
    nnet::zeropad2d_cl<layer18_t, layer44_t, config44>(layer18_out, layer44_out); // zp2d_conv2d_52

    hls::stream<layer19_t> layer19_out("layer19_out");
    #pragma HLS STREAM variable=layer19_out depth=12800
    nnet::conv_2d_cl<layer44_t, layer19_t, config19>(layer44_out, layer19_out, w19, b19); // conv2d_52

    hls::stream<layer20_t> layer20_out("layer20_out");
    #pragma HLS STREAM variable=layer20_out depth=12800
    nnet::linear<layer19_t, layer20_t, linear_config20>(layer19_out, layer20_out); // conv2d_52_linear

    hls::stream<layer47_t> layer47_out("layer47_out");
    #pragma HLS STREAM variable=layer47_out depth=12800
    nnet::pointwise_conv_2d_cl<layer50_t, layer47_t, config47>(layer50_cpy2, layer47_out, w47, b47); // conv2d_53

    hls::stream<layer22_t> layer22_out("layer22_out");
    #pragma HLS STREAM variable=layer22_out depth=12800
    nnet::linear<layer47_t, layer22_t, linear_config22>(layer47_out, layer22_out); // conv2d_53_linear

    hls::stream<layer23_t> layer23_out("layer23_out");
    #pragma HLS STREAM variable=layer23_out depth=12800
    nnet::normalize<layer20_t, layer23_t, config23>(layer20_out, layer23_out, s23, b23); // batch_normalization_212

    hls::stream<layer24_t> layer24_out("layer24_out");
    #pragma HLS STREAM variable=layer24_out depth=12800
    nnet::add<layer22_t, layer23_t, layer24_t, config24>(layer22_out, layer23_out, layer24_out); // add_79

    hls::stream<layer25_t> layer25_out("layer25_out");
    #pragma HLS STREAM variable=layer25_out depth=12800
    nnet::relu<layer24_t, layer25_t, relu_config25>(layer24_out, layer25_out); // activation_18

    hls::stream<layer51_t> layer51_cpy1("layer51_cpy1");
    #pragma HLS STREAM variable=layer51_cpy1 depth=12800
    hls::stream<layer51_t> layer51_cpy2("layer51_cpy2");
    #pragma HLS STREAM variable=layer51_cpy2 depth=12800
    nnet::clone_stream<layer25_t, layer51_t, 409600>(layer25_out, layer51_cpy1, layer51_cpy2); // clone_activation_18

    hls::stream<layer45_t> layer45_out("layer45_out");
    #pragma HLS STREAM variable=layer45_out depth=13029
    nnet::zeropad2d_cl<layer51_t, layer45_t, config45>(layer51_cpy1, layer45_out); // zp2d_conv2d_54

    hls::stream<layer26_t> layer26_out("layer26_out");
    #pragma HLS STREAM variable=layer26_out depth=3200
    nnet::conv_2d_cl<layer45_t, layer26_t, config26>(layer45_out, layer26_out, w26, b26); // conv2d_54

    hls::stream<layer27_t> layer27_out("layer27_out");
    #pragma HLS STREAM variable=layer27_out depth=3200
    nnet::linear<layer26_t, layer27_t, linear_config27>(layer26_out, layer27_out); // conv2d_54_linear

    hls::stream<layer28_t> layer28_out("layer28_out");
    #pragma HLS STREAM variable=layer28_out depth=3200
    nnet::normalize<layer27_t, layer28_t, config28>(layer27_out, layer28_out, s28, b28); // batch_normalization_213

    hls::stream<layer29_t> layer29_out("layer29_out");
    #pragma HLS STREAM variable=layer29_out depth=3200
    nnet::relu<layer28_t, layer29_t, relu_config29>(layer28_out, layer29_out); // activation_19

    hls::stream<layer46_t> layer46_out("layer46_out");
    #pragma HLS STREAM variable=layer46_out depth=3432
    nnet::zeropad2d_cl<layer29_t, layer46_t, config46>(layer29_out, layer46_out); // zp2d_conv2d_55

    hls::stream<layer30_t> layer30_out("layer30_out");
    #pragma HLS STREAM variable=layer30_out depth=3200
    nnet::conv_2d_cl<layer46_t, layer30_t, config30>(layer46_out, layer30_out, w30, b30); // conv2d_55

    hls::stream<layer31_t> layer31_out("layer31_out");
    #pragma HLS STREAM variable=layer31_out depth=3200
    nnet::linear<layer30_t, layer31_t, linear_config31>(layer30_out, layer31_out); // conv2d_55_linear

    hls::stream<layer48_t> layer48_out("layer48_out");
    #pragma HLS STREAM variable=layer48_out depth=3200
    nnet::pointwise_conv_2d_cl<layer51_t, layer48_t, config48>(layer51_cpy2, layer48_out, w48, b48); // conv2d_56

    hls::stream<layer33_t> layer33_out("layer33_out");
    #pragma HLS STREAM variable=layer33_out depth=3200
    nnet::linear<layer48_t, layer33_t, linear_config33>(layer48_out, layer33_out); // conv2d_56_linear

    hls::stream<layer34_t> layer34_out("layer34_out");
    #pragma HLS STREAM variable=layer34_out depth=3200
    nnet::normalize<layer31_t, layer34_t, config34>(layer31_out, layer34_out, s34, b34); // batch_normalization_214

    hls::stream<layer35_t> layer35_out("layer35_out");
    #pragma HLS STREAM variable=layer35_out depth=3200
    nnet::add<layer33_t, layer34_t, layer35_t, config35>(layer33_out, layer34_out, layer35_out); // add_80

    hls::stream<layer36_t> layer36_out("layer36_out");
    #pragma HLS STREAM variable=layer36_out depth=3200
    nnet::relu<layer35_t, layer36_t, relu_config36>(layer35_out, layer36_out); // activation_20

    hls::stream<layer38_t> layer38_out("layer38_out");
    #pragma HLS STREAM variable=layer38_out depth=1
    nnet::dense<layer36_t, layer38_t, config38>(layer36_out, layer38_out, w38, b38); // dense_11

    nnet::linear<layer38_t, result_t, linear_config39>(layer38_out, layer39_out); // dense_11_linear

}
