#ifndef TEMPORAL_LAYER
#define TEMPORAL_LAYER

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>
#include <cstring>
#include "host.h"




struct EncoderBlockParams {
    std::vector<float> ln0_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> ln0_bias = std::vector<float>(EMBED_DIM);
    std::vector<float> ln1_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> ln1_bias = std::vector<float>(EMBED_DIM);
    std::vector<int8_t> wq = std::vector<int8_t>(EMBED_DIM * EMBED_DIM);
    std::vector<int8_t> wk = std::vector<int8_t>(EMBED_DIM * EMBED_DIM);
    std::vector<int8_t> wv = std::vector<int8_t>(EMBED_DIM * EMBED_DIM);
    std::vector<int8_t> wo = std::vector<int8_t>(EMBED_DIM * EMBED_DIM);
    std::vector<float> q_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> k_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> v_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> o_scale = std::vector<float>(EMBED_DIM);
    std::vector<float> q_bias = std::vector<float>(EMBED_DIM);
    std::vector<float> k_bias = std::vector<float>(EMBED_DIM);
    std::vector<float> v_bias = std::vector<float>(EMBED_DIM);
    std::vector<float> o_bias = std::vector<float>(EMBED_DIM);
    std::vector<int8_t> mlp_w1 = std::vector<int8_t>(EMBED_DIM * MLP_DIM);
    std::vector<int8_t> mlp_w2 = std::vector<int8_t>(MLP_DIM * EMBED_DIM);
    std::vector<float> mlp_b1 = std::vector<float>(MLP_DIM);
    std::vector<float> mlp_b2 = std::vector<float>(EMBED_DIM);
    std::vector<float> mlp_s1 = std::vector<float>(MLP_DIM);
    std::vector<float> mlp_s2 = std::vector<float>(EMBED_DIM);
};



namespace vivit_temp {
    bool load_Temporal_encoder_block(const std::string &root, int block, EncoderBlockParams &p);


    float gelu_int_sw(float x);
    void quantize_activation_sw(const float in[TEMP_LEN][EMBED_DIM],
                            int8_t out[TEMP_LEN][EMBED_DIM],
                            float scale[TEMP_LEN]);
    void layernorm_sw(const float in[TEMP_LEN][EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM],
                  const float scale[EMBED_DIM],
                  const float bias[EMBED_DIM]);
    void linear_quant_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
                     const float in_scale[TEMP_LEN],
                     const int8_t weight[EMBED_DIM][EMBED_DIM],
                     const float weight_scale[EMBED_DIM],
                     const float bias[EMBED_DIM],
                     float out[TEMP_LEN][EMBED_DIM]);
    void attention_sw(const float in[TEMP_LEN][EMBED_DIM],
                  const int8_t Wq[EMBED_DIM][EMBED_DIM],
                  const int8_t Wk[EMBED_DIM][EMBED_DIM],
                  const int8_t Wv[EMBED_DIM][EMBED_DIM],
                  const int8_t Wo[EMBED_DIM][EMBED_DIM],
                  const float q_weight_scale[EMBED_DIM],
                  const float k_weight_scale[EMBED_DIM],
                  const float v_weight_scale[EMBED_DIM],
                  const float o_weight_scale[EMBED_DIM],
                  const float q_bias[EMBED_DIM],
                  const float k_bias[EMBED_DIM],
                  const float v_bias[EMBED_DIM],
                  const float o_bias[EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM]);
    void mlp_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
            const float scale_input[TEMP_LEN],
            const int8_t W1[EMBED_DIM][MLP_DIM],
            const float b1[MLP_DIM],
            const float scale_w1[MLP_DIM],
            const int8_t W2[MLP_DIM][EMBED_DIM],
            const float b2[EMBED_DIM],
            const float scale_w2[EMBED_DIM],
            float out[TEMP_LEN][EMBED_DIM]);

    void residual_add(const float a[TEMP_LEN][EMBED_DIM],
                  const float b[TEMP_LEN][EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM]);
}

#endif