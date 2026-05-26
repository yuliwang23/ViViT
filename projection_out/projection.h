#ifndef PROJECTION_OUT_H
#define PROJECTION_OUT_H

#include <cmath>
#include <stdint.h>
#include "../vivit_config.h"

constexpr int SEQ_LEN = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int HEAD_DIM = VIVIT_HEAD_DIM;
constexpr int OUT_TILE = VIVIT_OUT_TILE;
constexpr int DOT_UNROLL = VIVIT_DOT_UNROLL;
constexpr int PACK_FACTOR = 8;
constexpr int EMBED_DIM_PACKED = EMBED_DIM / PACK_FACTOR;
constexpr int PACKS_PER_DOT = DOT_UNROLL / PACK_FACTOR;

// Small C/RTL co-simulation setting:
//  constexpr int SEQ_LEN = 4;
//  constexpr int EMBED_DIM = 64;
//  constexpr int NUM_HEADS = 2;
//  constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS; // 32
//  constexpr int OUT_TILE = 16;
//  constexpr int DOT_UNROLL = 8;
//  constexpr int PACK_FACTOR = 8;
//  constexpr int EMBED_DIM_PACKED = EMBED_DIM / PACK_FACTOR; // 8
//  constexpr int PACKS_PER_DOT = DOT_UNROLL / PACK_FACTOR;   // 1

extern "C"
{
    void projection_out(
        float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
        int8_t weight[EMBED_DIM][EMBED_DIM],
        float bias[EMBED_DIM],
        float scale_weight[EMBED_DIM],
        float out[SEQ_LEN][EMBED_DIM]);
}

#endif
