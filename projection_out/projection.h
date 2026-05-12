#ifndef PROJECTION_OUT_H
#define PROJECTION_OUT_H

#include <cmath>
#include <stdint.h>

//constexpr int SEQ_LEN = 14 * 14 + 1;           // 197
//constexpr int EMBED_DIM = 768;
//constexpr int NUM_HEADS = 12;
//constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 64
//constexpr int OUT_TILE = 64;
//constexpr int DOT_UNROLL = 16;

constexpr int SEQ_LEN = 4;
constexpr int EMBED_DIM = 64;
constexpr int NUM_HEADS = 2;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int OUT_TILE = 16;
constexpr int DOT_UNROLL = 8;


extern "C" {
void projection_out(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);
}

void quantize_attention_token(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    int8_t q_token[EMBED_DIM],
    float *scale);

void quantize_attention_all(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t q_tokens[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN]);

void load_weight_tile(
    int8_t weight[EMBED_DIM][EMBED_DIM],
    int o_tile,
    int8_t weight_tile[EMBED_DIM][OUT_TILE]);

void project_tile(
    int8_t q_tokens[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight_tile[EMBED_DIM][OUT_TILE],
    int o_tile,
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);

#endif
