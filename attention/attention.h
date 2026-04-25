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

constexpr int SEQ_LEN = 14 * 14 + 1;
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float scale = 0.125;
constexpr int VEC_MAX = SEQ_LEN;
constexpr int tile_size = 16;


void attention_opt(
    float Q[SEQ_LEN][HEAD_DIM],
    float K[SEQ_LEN][HEAD_DIM],
    float V[SEQ_LEN][HEAD_DIM],
    float Out[SEQ_LEN][HEAD_DIM]);

// Initialize state BRAMs (m, l, O) for a new Q tile
void init_states(
    float BRAM_m[tile_size],
    float BRAM_l[tile_size],
    float BRAM_O[tile_size][HEAD_DIM]
);

// Load a single tile of Q, K, or V from HBM to FIFO with padding logic
void load_tile_HBM2FIFO(
    uint256_dt *HBM_ptr, 
    hls::stream<uint256_dt> &FIFO_QKV,
    bool last_tile
);

// Unpack 256-bit chunks from FIFO and store them into local BRAM
void load_QKV_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_QKV,
    float BRAM_QKV[tile_size][HEAD_DIM]
);

// ================= Score =================
void compute_score(
    float BRAM_Q[tile_size][HEAD_DIM],
    float BRAM_K[tile_size][HEAD_DIM],
    hls::stream<float> &FIFO_score
);

// ================= Softmax =================
void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11]);

void compute_max(
    float BRAM_score[SEQ_LEN + 11],
    float &max_score);

void compute_sum_exp(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float &sum_exp);

void compute_softmax_output(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float sum_exp,
    hls::stream<float> &FIFO_softmax_out);

void compute_softmax_one_dim(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11],
    hls::stream<float> &FIFO_softmax_out);

void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_softmax_out);

// Update unnormalized Output and defer division until the last tile
void compute_output(
    hls::stream<float> &FIFO_m,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_P,
    hls::stream<uint256_dt> &FIFO_O_out,
    float BRAM_m[tile_size],
    float BRAM_l[tile_size],
    float BRAM_O[tile_size][HEAD_DIM],
    float BRAM_V[tile_size][HEAD_DIM],
    bool last_tile
);

// Store the finalized and padded Output tile from FIFO back to HBM
void store_tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_O_out,
    uint256_dt *HBM_ptr,
    bool last_tile_Q 
);

#endif // ATTENTION_H