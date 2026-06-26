#ifndef MLP_HLS_H
#define MLP_HLS_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <cstdint>
#include <algorithm>
#include <cmath>

typedef ap_uint<256> uint256_dt; // 8 x float32  OR  32 x int8
typedef ap_uint<128> uint128_dt; // 8 x half_dt packed
typedef ap_int<64> int64_dt;     // 8 x int8 packed
typedef ap_int<8> int8_dt;
typedef ap_int<26> int26_dt; // matmul accumulator (int8*int8 sum)
typedef ap_int<32> int32_dt;

// Compact storage type for the hidden-layer GELU activation buffer.
// 16-bit fixed-point, 8 integer bits (range [-128, 127.996], step 1/256).
typedef ap_fixed<16, 8> half_dt;

#define SEQ_LEN 197
#define TOKEN_TILE 4
#define SEQ_LEN_PADDED 200   // = ((197 + 3) / 4) * 4

#define EMBED_DIM 768   // = IN_DIM = OUT_DIM
#define HIDDEN_DIM 3072 // 4 * EMBED_DIM (standard GPT MLP)

#define EMBED_DIM_package (EMBED_DIM / 8)   // 96   (each = 8 int8)
#define HIDDEN_DIM_package (HIDDEN_DIM / 8) // 384

// Tile size on the *output-feature* dimension (rows of W^T we compute per pass)
#define TILE_SIZE 32

#define EPSILON 1e-5f
#define EMBED_DIM_INV (1.0f / (float)EMBED_DIM)
#define HIDDEN_DIM_INV (1.0f / (float)HIDDEN_DIM)

constexpr int NUM_ITERATIONS = 48;

// MEAN_WT[i] = 1/(i+1)
static const float MEAN_WT[NUM_ITERATIONS] = {
    1.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 4, 1.0f / 5, 1.0f / 6,
    1.0f / 7, 1.0f / 8, 1.0f / 9, 1.0f / 10, 1.0f / 11, 1.0f / 12,
    1.0f / 13, 1.0f / 14, 1.0f / 15, 1.0f / 16, 1.0f / 17, 1.0f / 18,
    1.0f / 19, 1.0f / 20, 1.0f / 21, 1.0f / 22, 1.0f / 23, 1.0f / 24,
    1.0f / 25, 1.0f / 26, 1.0f / 27, 1.0f / 28, 1.0f / 29, 1.0f / 30,
    1.0f / 31, 1.0f / 32, 1.0f / 33, 1.0f / 34, 1.0f / 35, 1.0f / 36,
    1.0f / 37, 1.0f / 38, 1.0f / 39, 1.0f / 40, 1.0f / 41, 1.0f / 42,
    1.0f / 43, 1.0f / 44, 1.0f / 45, 1.0f / 46, 1.0f / 47, 1.0f / 48};

// M2_WT[i] = 16*i/(i+1)
static const float M2_WT[NUM_ITERATIONS] = {
    0.0f,
    16.0f * 1 / 2, 16.0f * 2 / 3, 16.0f * 3 / 4, 16.0f * 4 / 5, 16.0f * 5 / 6,
    16.0f * 6 / 7, 16.0f * 7 / 8, 16.0f * 8 / 9, 16.0f * 9 / 10, 16.0f * 10 / 11,
    16.0f * 11 / 12, 16.0f * 12 / 13, 16.0f * 13 / 14, 16.0f * 14 / 15, 16.0f * 15 / 16,
    16.0f * 16 / 17, 16.0f * 17 / 18, 16.0f * 18 / 19, 16.0f * 19 / 20, 16.0f * 20 / 21,
    16.0f * 21 / 22, 16.0f * 22 / 23, 16.0f * 23 / 24, 16.0f * 24 / 25, 16.0f * 25 / 26,
    16.0f * 26 / 27, 16.0f * 27 / 28, 16.0f * 28 / 29, 16.0f * 29 / 30, 16.0f * 30 / 31,
    16.0f * 31 / 32, 16.0f * 32 / 33, 16.0f * 33 / 34, 16.0f * 34 / 35, 16.0f * 35 / 36,
    16.0f * 36 / 37, 16.0f * 37 / 38, 16.0f * 38 / 39, 16.0f * 39 / 40, 16.0f * 40 / 41,
    16.0f * 41 / 42, 16.0f * 42 / 43, 16.0f * 43 / 44, 16.0f * 44 / 45, 16.0f * 45 / 46,
    16.0f * 46 / 47, 16.0f * 47 / 48};

extern "C"
{
    void mlp_q(
        uint256_dt *input,   // FP32 input packed 8 values/chunk
        uint256_dt *gamma,
        uint256_dt *beta,
        uint256_dt *W1,      // [HIDDEN_DIM * EMBED_DIM  / 32] int8 packed
        uint256_dt *W2,      // [EMBED_DIM  * HIDDEN_DIM / 32] int8 packed
        float *b1,           // [HIDDEN_DIM]
        float *b2,           // [EMBED_DIM]
        float *scale_w1,     // [HIDDEN_DIM]
        float *scale_w2,     // [EMBED_DIM]
        uint256_dt *output,  // [SEQ_LEN * EMBED_DIM / 8]   FP32 packed
        int token_offset,
        int valid_seq_len);
}

void do_layernorm_dataflow(
    uint256_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN_PADDED],
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    int token_offset,
    int valid_seq_len);

void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X,
    int token_offset,
    int valid_seq_len);

void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]);

void load_beta_gamma_HBM2SRAM(
    uint256_dt HBM_gamma[EMBED_DIM / 8],
    uint256_dt HBM_beta[EMBED_DIM / 8],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]);

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN_PADDED],
    int64_dt X_package_array[SEQ_LEN_PADDED][EMBED_DIM_package],
    int seq_idx);

void layernorm_compute(
    hls::stream<uint256_dt> &FIFO_X,
    float scaler[SEQ_LEN_PADDED],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    int token_offset,
    int valid_seq_len);

void load_Bias_HBM2BRAM(
    float *b,
    float BRAM_b[],
    int N);

void load_Weight_tile_HBM2FIFO(
    uint256_dt *W,
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset,
    int K_dim);

void quantize_row_dynamic(
    float row_fp[],
    int64_dt row_pkg[],
    int length,
    float *out_scaler);

static void dense0_dataflow_tile(
    uint256_dt *W1,
    int64_dt    X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    float       scaler_X [SEQ_LEN_PADDED],
    float       SRAM_b1  [HIDDEN_DIM],
    float       SRAM_sw1 [HIDDEN_DIM],
    uint128_dt  H_full_pack[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float       scaler_H [SEQ_LEN_PADDED],
    int         iter);

static void dense1_dataflow_tile(
    uint256_dt *W2,
    uint256_dt *output,
    int64_dt    H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float       scaler_H [SEQ_LEN_PADDED],
    float       SRAM_b2  [EMBED_DIM],
    float       SRAM_sw2 [EMBED_DIM],
    int         iter,
    int         token_offset,
    int         valid_seq_len);

void dense0_one_tile(
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    float scaler_X[SEQ_LEN_PADDED],
    hls::stream<uint256_dt> &FIFO_W1,
    float SRAM_b1[HIDDEN_DIM],
    float SRAM_sw1[HIDDEN_DIM],
    uint128_dt H_full_pack[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float scaler_H[SEQ_LEN_PADDED],
    int tile_idx);

void dense1_one_tile(
    int64_dt H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float scaler_H[SEQ_LEN_PADDED],
    hls::stream<uint256_dt> &FIFO_W2,
    float SRAM_b2[EMBED_DIM],
    float SRAM_sw2[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_OUT,
    int tile_idx);

void store_Output_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_OUT,
    uint256_dt *output,
    int tile_idx,
    int token_offset,
    int valid_seq_len);

#endif