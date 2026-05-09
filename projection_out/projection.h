#ifndef PROJECTION_OUT_H
#define PROJECTION_OUT_H

#include <cmath>
#include <stdint.h>

constexpr int SEQ_LEN = 14 * 14 + 1;           // 197
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 64
constexpr int OUT_TILE = 64;
constexpr int PACK_INT8 = 32;

extern "C" {
void projection_out(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM]);
}

void quantize_attention_token(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    int8_t q_token[EMBED_DIM],
    float *scale);

void project_token(
    int8_t q_token[EMBED_DIM],
    float scale_input,
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out_token[EMBED_DIM]);

#endif
