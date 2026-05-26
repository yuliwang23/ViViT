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
#include "../vivit_config.h"
// #define MAX(a, b) ((a) > (b) ? (a) : (b))
typedef ap_uint<256> uint256_dt;
constexpr int SEQ_LEN = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int HEAD_DIM = VIVIT_HEAD_DIM;
constexpr float scale = VIVIT_ATTENTION_SCALE;
constexpr int VEC_MAX = SEQ_LEN;
constexpr int tile_size = VIVIT_ATTENTION_TILE_SIZE;
constexpr int num_tiles = VIVIT_ATTENTION_NUM_TILES;

extern "C" void attention_opt(
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

// Compute Attention Score (S = Q * K^T)
void compute_score(
    float BRAM_Q[tile_size][HEAD_DIM],
    float BRAM_K[tile_size][HEAD_DIM],
    hls::stream<float> &FIFO_score
);

// Load computed scores from stream to local BRAM for softmax processing
void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[tile_size][tile_size]
);

// Compute FlashAttention-2 online softmax parameters (max and sum of exp)
void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_P,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_m,
    bool last_tile
);

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
