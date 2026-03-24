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
// #define MAX(a, b) ((a) > (b) ? (a) : (b))
typedef ap_uint<256> uint256_dt;
constexpr int SEQ_LEN = 14 * 14 + 1; // 14*14*16
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float scale = 0.125; // for Base param. (FE_B_HEAD_DIM = 64)
constexpr int VEC_MAX = SEQ_LEN;
constexpr int tile_size = 16; 


void attention_opt(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int Out[SEQ_LEN][HEAD_DIM]);

void softmax_software(float in[VEC_MAX],
            float out[VEC_MAX],
            int length);

void load_QKV_HBM2FIFO( //Q will be call with 13 times
    int QKV[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_QKV);

void load_QKV_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_QKV,
    int BRAM_QKV[tile_size][HEAD_DIM]);

void load_O_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_O,
    float BRAM_O[tile_size][HEAD_DIM]);

void compute_score(
    int BRAM_Q[tile_size][HEAD_DIM],
    int BRAM_K[tile_size][HEAD_DIM],
    hls::stream<float> &FIFO_score
);

void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[tile_size][tile_size]);

void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_P,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_m,
    bool last_tile);

void compute_output(
    hls::stream<float> &FIFO_m,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_P,
    hls::stream<uint256_dt> &FIFO_O_out,
    float BRAM_m[SEQ_LEN],
    float BRAM_l[SEQ_LEN],
    float BRAM_O[tile_size][HEAD_DIM],
    int BRAM_V[tile_size][HEAD_DIM],
    int tile_idx
);

void store_O_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_O_out,
    int O[SEQ_LEN][HEAD_DIM]
);

#endif // ATTENTION_H