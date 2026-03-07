#ifndef ATTENTION_H
#define ATTENTION_H
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <cmath>
typedef ap_uint<256> uint256_dt;
constexpr int SEQ_LEN = 14 * 14 + 1; // 14*14*16
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float scale = 0.125; // for Base param. (FE_B_HEAD_DIM = 64)
constexpr int VEC_MAX = SEQ_LEN;

void attention_opt(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int Out[SEQ_LEN][HEAD_DIM]);

void softmax_software(float in[VEC_MAX],
            float out[VEC_MAX],
            int length);

#endif // ATTENTION_H