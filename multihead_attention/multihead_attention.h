#ifndef MULTIHEAD_ATTENTION_H
#define MULTIHEAD_ATTENTION_H

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <cmath>
#include <array>
#include <cfloat>
#include <algorithm>
#include "./vivit_config.h"

// ============================================================================
// Constants Definitions
// ============================================================================
constexpr int SEQ_LEN = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int HEAD_DIM = VIVIT_HEAD_DIM;
constexpr int TILE_SIZE = VIVIT_LINEAR_TILE_SIZE;

constexpr int EMBED_DIM_package = VIVIT_EMBED_DIM_PACKAGE;
constexpr int DSP_PER_CYCLE = VIVIT_DSP_PER_CYCLE;
constexpr int SEQ_LEN_PADDED = ((SEQ_LEN + 7) / 8) * 8;

constexpr int INT8_PACK = 8;
constexpr int FP32_PACK = 2;
constexpr float NEG_INF = -1e9f;

constexpr int NUM_ITERATIONS = VIVIT_LAYER_NORM_ITERATIONS;
constexpr float EPSILON = 1e-5f;
constexpr float EMBED_DIM_INV = 1.0f / static_cast<float>(EMBED_DIM);
constexpr int SOFTMAX_PACKS = SEQ_LEN_PADDED / INT8_PACK;
constexpr int SOFTMAX_PACK_STEPS = (SOFTMAX_PACKS + 3) / 4;

// ============================================================================
// Type Definitions
// ============================================================================
typedef ap_uint<256> uint256_dt;
typedef ap_int<8> int8_dt;
typedef ap_int<16> int16_dt;
typedef ap_int<26> int26_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<64> int64_dt;

// ============================================================================
// LayerNorm Look-Up Tables
// ============================================================================
// MEAN_WT[i] = 1/(i+1)
// static const float MEAN_WT[NUM_ITERATIONS] = {
//     1.0f / 1, 1.0f / 2, 1.0f / 3, 1.0f / 4, 1.0f / 5, 1.0f / 6,
//     1.0f / 7, 1.0f / 8, 1.0f / 9, 1.0f / 10, 1.0f / 11, 1.0f / 12,
//     1.0f / 13, 1.0f / 14, 1.0f / 15, 1.0f / 16, 1.0f / 17, 1.0f / 18,
//     1.0f / 19, 1.0f / 20, 1.0f / 21, 1.0f / 22, 1.0f / 23, 1.0f / 24,
//     1.0f / 25, 1.0f / 26, 1.0f / 27, 1.0f / 28, 1.0f / 29, 1.0f / 30,
//     1.0f / 31, 1.0f / 32, 1.0f / 33, 1.0f / 34, 1.0f / 35, 1.0f / 36,
//     1.0f / 37, 1.0f / 38, 1.0f / 39, 1.0f / 40, 1.0f / 41, 1.0f / 42,
//     1.0f / 43, 1.0f / 44, 1.0f / 45, 1.0f / 46, 1.0f / 47, 1.0f / 48};

// // M2_WT[i] = 16*i/(i+1)
// static const float M2_WT[NUM_ITERATIONS] = {
//     0.0f,
//     16.0f * 1 / 2, 16.0f * 2 / 3, 16.0f * 3 / 4, 16.0f * 4 / 5, 16.0f * 5 / 6,
//     16.0f * 6 / 7, 16.0f * 7 / 8, 16.0f * 8 / 9, 16.0f * 9 / 10, 16.0f * 10 / 11,
//     16.0f * 11 / 12, 16.0f * 12 / 13, 16.0f * 13 / 14, 16.0f * 14 / 15, 16.0f * 15 / 16,
//     16.0f * 16 / 17, 16.0f * 17 / 18, 16.0f * 18 / 19, 16.0f * 19 / 20, 16.0f * 20 / 21,
//     16.0f * 21 / 22, 16.0f * 22 / 23, 16.0f * 23 / 24, 16.0f * 24 / 25, 16.0f * 25 / 26,
//     16.0f * 26 / 27, 16.0f * 27 / 28, 16.0f * 28 / 29, 16.0f * 29 / 30, 16.0f * 30 / 31,
//     16.0f * 31 / 32, 16.0f * 32 / 33, 16.0f * 33 / 34, 16.0f * 34 / 35, 16.0f * 35 / 36,
//     16.0f * 36 / 37, 16.0f * 37 / 38, 16.0f * 38 / 39, 16.0f * 39 / 40, 16.0f * 40 / 41,
//     16.0f * 41 / 42, 16.0f * 42 / 43, 16.0f * 43 / 44, 16.0f * 44 / 45, 16.0f * 45 / 46,
//     16.0f * 46 / 47, 16.0f * 47 / 48};

// ============================================================================
// Top-Level Module Declaration
// ============================================================================
extern "C"
{
    void multihead_attention(
        // Input Feature Map (Used for both LayerNorm and Residual)
        uint256_dt *x,

        // LayerNorm Weights
        uint256_dt *gamma,
        uint256_dt *beta,

        // QKV & Output Projection Weights
        uint256_dt *Wq,
        uint256_dt *Wk,
        uint256_dt *Wv,
        uint256_dt *Wo,

        // Biases
        float *bq,
        float *bk,
        float *bv,
        float *bo,

        // Scales
        float *scale_wq,
        float *scale_wk,
        float *scale_wv,
        float *scale_wo,

        // Final Output
        uint256_dt *out);
}

// ============================================================================
// Sub-Function Declarations
// ============================================================================

// ----------------------------------------------------------------------------
// LayerNorm and Pre-Processing
// ----------------------------------------------------------------------------
static void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X);

static void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]);

static void load_beta_gamma_HBM2SRAM(
    uint256_dt HBM_gamma[EMBED_DIM / 8],
    uint256_dt HBM_beta[EMBED_DIM / 8],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]);

static void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package],
    int seq_idx);

static void layernorm_compute(
    hls::stream<uint256_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);

static void do_layernorm_dataflow(
    uint256_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);

// ----------------------------------------------------------------------------
// QKV Linear Projection
// ----------------------------------------------------------------------------
static void load_Bias_HBM2BRAM(
    float b[EMBED_DIM],
    float BRAM_b[EMBED_DIM]);

void load_Weight_tile_HBM2FIFO(
    uint256_dt *W,
    hls::stream<uint256_dt> &FIFO_W,
    int iter);

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int64_dt URAM_Wq[TILE_SIZE][EMBED_DIM_package],
    int64_dt URAM_Wk[TILE_SIZE][EMBED_DIM_package],
    int64_dt URAM_Wv[TILE_SIZE][EMBED_DIM_package]);

void compute_linear_qkv_one_head(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    float scale_wq[EMBED_DIM],
    float scale_wk[EMBED_DIM],
    float scale_wv[EMBED_DIM],
    float scale_X[SEQ_LEN],
    int iter);

// ----------------------------------------------------------------------------
// Attention Score and Context Computation
// ----------------------------------------------------------------------------
void qkv_sram_and_quantize(
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,
    int64_dt URAM_Q[SEQ_LEN][HEAD_DIM / INT8_PACK],
    int64_dt URAM_K[SEQ_LEN][HEAD_DIM / INT8_PACK],
    int64_dt URAM_V_T[HEAD_DIM][SEQ_LEN_PADDED / INT8_PACK],
    float scaler_Q[SEQ_LEN],
    float scaler_K[SEQ_LEN],
    float scaler_V[HEAD_DIM]);

void compute_score_and_softmax(
    int64_dt URAM_Q[SEQ_LEN][HEAD_DIM / INT8_PACK],
    int64_dt URAM_K[SEQ_LEN][HEAD_DIM / INT8_PACK],
    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],
    hls::stream<int8_dt> &FIFO_Softmax_out,
    hls::stream<float> &FIFO_Softmax_scaler_out);

void compute_attention_output_one_head(
    hls::stream<int8_dt> &FIFO_Softmax_out,
    hls::stream<float> &FIFO_Softmax_scaler_out,
    int64_dt URAM_V_T[HEAD_DIM][SEQ_LEN_PADDED / INT8_PACK],
    float scale_V[HEAD_DIM],
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],
    float global_max[SEQ_LEN],
    int head_idx);

void process_attention_one_head_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *Wq, uint256_dt *Wk, uint256_dt *Wv,
    float SRAM_bq[EMBED_DIM], float SRAM_bk[EMBED_DIM], float SRAM_bv[EMBED_DIM],
    float SRAM_swq[EMBED_DIM], float SRAM_swk[EMBED_DIM], float SRAM_swv[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],
    float global_max[SEQ_LEN],
    int head_idx);

void quantize_attention_output(
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],
    float global_max[SEQ_LEN],
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK],
    float out_scaler[SEQ_LEN]);

// ----------------------------------------------------------------------------
// Output Projection and Residual
// ----------------------------------------------------------------------------
static void load_Residual_Tile_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter);

void compute_output_proj_and_residual(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK],
    hls::stream<uint256_dt> &FIFO_Wo,
    hls::stream<uint256_dt> &FIFO_Res,
    hls::stream<uint256_dt> &FIFO_Final_Out,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],
    int iter);

static void store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint256_dt *HBM_Out,
    int iter);

void process_o_proj_single_tile_dataflow(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK],
    uint256_dt *HBM_Wo,
    uint256_dt *HBM_X_Res,
    uint256_dt *HBM_Out,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],
    int iter);

void run_full_output_projection(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK],
    uint256_dt *HBM_Wo,
    uint256_dt *HBM_X_Res,
    uint256_dt *HBM_Out,
    float out_scaler[SEQ_LEN],
    float *HBM_bo,
    float *HBM_scale_wo);

#endif // MULTIHEAD_ATTENTION_H
