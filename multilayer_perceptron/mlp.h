#ifndef MLP_H
#define MLP_H

#include "hls_stream.h"
#include "hls_math.h"
#include "ap_int.h"
#include <cmath>
#include <algorithm>
#include <stdint.h>
#include "./vivit_config.h"

// =======================================================================
//   Macros and Constants
// =======================================================================
#define SEQ_LEN VIVIT_SEQ_LEN
#define EMBED_DIM VIVIT_EMBED_DIM
#define MLP_DIM VIVIT_MLP_DIM
#define TILE_OUT_DIM VIVIT_MLP_TILE_OUT_DIM
#define TILE_SIZE VIVIT_LINEAR_TILE_SIZE
#define EMBED_DIM_package VIVIT_EMBED_DIM_PACKAGE
#define MLP_DIM_package VIVIT_MLP_DIM_PACKAGE
#define INT8_PACK 8
#define FP32_PACK 8
#define MLP1_NUM_TILES (MLP_DIM / TILE_OUT_DIM)
#define MLP2_NUM_TILES (EMBED_DIM / TILE_SIZE)
#define MLP_PACK_PARALLEL VIVIT_MLP_PACK_PARALLEL
#define EPSILON 1e-5f
#define EMBED_DIM_INV (1.0f / static_cast<float>(EMBED_DIM))

// =======================================================================
//   Typedefs
// =======================================================================
typedef ap_int<8> int8_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<64> int64_dt;
typedef ap_uint<256> uint256_dt;

// =======================================================================
//   Function Prototypes
// =======================================================================
extern "C"
{
    void mlp(
        uint256_dt *x,
        uint256_dt *gamma, uint256_dt *beta,
        uint256_dt *W1, uint256_dt *W2,
        float *b1, float *b2,
        float *scale_w1, float *scale_w2,
        uint256_dt *out);
}

// Memory Loading Helpers
static void load_X_HBM2FIFO(uint256_dt *HBM_X, hls::stream<uint256_dt> &FIFO_X);
static void load_X_FIFO2BRAM(hls::stream<uint256_dt> &FIFO_X, float BRAM_X[EMBED_DIM]);
static void load_beta_gamma_HBM2SRAM(uint256_dt HBM_gamma[EMBED_DIM / 8], uint256_dt HBM_beta[EMBED_DIM / 8], float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM]);
static void load_Bias_HBM2BRAM(float *HBM_b, float BRAM_b[EMBED_DIM]);
void load_Bias_HBM2BRAM_MLP(float *HBM_b, float BRAM_b[MLP_DIM]);
void load_W1_tile_HBM2FIFO(uint256_dt *HBM_W1, hls::stream<uint256_dt> &FIFO_W1, int iter);
void load_W2_tile_HBM2FIFO(uint256_dt *HBM_W2, hls::stream<uint256_dt> &FIFO_W2, int iter);
static void load_Residual_Tile_HBM2FIFO(uint256_dt *HBM_X, hls::stream<uint256_dt> &FIFO_Res, int iter);
static void store_Final_Tile_FIFO2HBM(hls::stream<uint256_dt> &FIFO_Final_Out, uint256_dt *HBM_Out, int iter);

// LayerNorm
static void compute_mean_variance(float BRAM_X[EMBED_DIM], float gamma[EMBED_DIM], float beta[EMBED_DIM], float scaler_array[SEQ_LEN], int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package], int seq_idx);
static void layernorm_compute(hls::stream<uint256_dt> &FIFO_X, float scaler[SEQ_LEN], float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM], int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);
static void do_layernorm_dataflow(uint256_dt *x, float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM], float scaler_X[SEQ_LEN], int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);

// MLP Layer 1 & GELU
inline float gelu_hw(float x);
void compute_linear_mlp1_tile(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], hls::stream<uint256_dt> &FIFO_W1, hls::stream<float> &FIFO_GELU_in, float BRAM_b1[MLP_DIM], float scale_w1[MLP_DIM], float scale_X[SEQ_LEN], int iter);
void process_mlp1_single_tile_dataflow(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, hls::stream<float> &FIFO_GELU_in, float BRAM_b1[MLP_DIM], float scale_w1[MLP_DIM], float scale_X[SEQ_LEN], int iter);
void run_full_mlp_layer1(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, hls::stream<float> &FIFO_GELU_in, float *HBM_b1, float *HBM_scale_w1, float scale_X[SEQ_LEN]);
void run_mlp1_and_gelu_dataflow(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, float *HBM_b1, float *HBM_scale_w1, float scale_X[SEQ_LEN], int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN]);
void compute_gelu_and_buffer_fp32(hls::stream<float> &FIFO_GELU_in, int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN]);
void quantize_mlp1_output(int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN], int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], float scaler_MLP1[SEQ_LEN]);

// MLP Layer 2 & Residual
void compute_mlp2_proj_and_residual(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], hls::stream<uint256_dt> &FIFO_W2, hls::stream<uint256_dt> &FIFO_Res, hls::stream<uint256_dt> &FIFO_Final_Out, float scaler_MLP1[SEQ_LEN], float BRAM_b2[EMBED_DIM], float scale_w2[EMBED_DIM], int iter);
void process_mlp2_single_tile_dataflow(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], uint256_dt *HBM_W2, uint256_dt *HBM_X_Res, uint256_dt *HBM_Out, float scaler_MLP1[SEQ_LEN], float BRAM_b2[EMBED_DIM], float scale_w2[EMBED_DIM], int iter);
void run_full_mlp_layer2(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], uint256_dt *HBM_W2, uint256_dt *HBM_X_Res, uint256_dt *HBM_Out, float scaler_MLP1[SEQ_LEN], float *HBM_b2, float *HBM_scale_w2);

void load_w2_and_res_sequential(
    uint256_dt *HBM_W2,
    uint256_dt *HBM_X_Res,
    hls::stream<uint256_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
);

#endif // MLP_H
