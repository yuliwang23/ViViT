#include <CL/cl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
// openmp
#include <omp.h>
#include "host.h"

using namespace std;


int gelu_int(int x);
void layernorm_host(int in[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM]);
void mlp_quant_host(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t W1[EMBED_DIM][MLP_DIM],
    float b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    int8_t W2[MLP_DIM][EMBED_DIM],
    float b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);

void residual_add(int a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM]);
void residual_add(float a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM]);
void residual_add(float a[SEQ_LEN][EMBED_DIM], float b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM]);

// Quantize attention outputs with one activation scale per token across all heads.
void quantize_attention_heads(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN]);

void linear_quant_host_fp32_input(
    int in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM]);


void concat_attention_heads(
    int8_t attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_concat[SEQ_LEN][EMBED_DIM]);


// opt. version

void quantize_attention_heads_mp(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN]);
void concat_attention_heads_mp(
    int8_t attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_concat[SEQ_LEN][EMBED_DIM]);

void residual_add_mp(
    const int a[SEQ_LEN][EMBED_DIM],
    const int b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);
void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const int   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);
void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const float   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);
void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const float   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);

void linear_quant_host_fp32_input_mp(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM]);

void mlp_quant_host_mp(
    const int8_t in[SEQ_LEN][EMBED_DIM],
    const float scale_input[SEQ_LEN],
    const int8_t W1[EMBED_DIM][MLP_DIM],
    const float b1[MLP_DIM],
    const float scale_w1[MLP_DIM],
    const int8_t W2[MLP_DIM][EMBED_DIM],
    const float b2[EMBED_DIM],
    const float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);


/* ====================================================
spatial encoder's layernorm
posembedding
>temporal encoder * 12
    layernorm
    attention_layer: quant->QKV->dequant->SoftMax->Quant->Out Projection & Concat->dequant->Residual
    layernorm
    MLP: quant->matmul->dequant->act. func->quant->matmul->dequant->act. func->residual
temportal encoder's layernorm
 ====================================================== */
/*
void temporal_layernorm(
    float in[TEMP_LEN][EMBED_DIM],
    float kernel[EMBED_DIM],
    float bias[EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM],
    float out_scale[TEMP_LEN]
);

void temporal_attention(
    float in[TEMP_LEN][EMBED_DIM],
    float in_scale[TEMP_LEN],
    int   Wq[EMBED_DIM][EMBED_DIM],
    int   Wk[EMBED_DIM][EMBED_DIM],
    int   Wv[EMBED_DIM][EMBED_DIM],
    int   Wo[EMBED_DIM][EMBED_DIM],
    float w_scale[EMBED_DIM],
    float q_scale[EMBED_DIM],
    float v_scale[EMBED_DIM],
    float o_scale[EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM]
);

void temporal_residual(
    float in1[TEMP_LEN][EMBED_DIM],
    float in2[TEMP_LEN][EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM]
);

void temporal_posEmbd(
    float in[TEMP_LEN][EMBED_DIM],
    float pos_embd[TEMP_LEN][EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM]
);
*/

