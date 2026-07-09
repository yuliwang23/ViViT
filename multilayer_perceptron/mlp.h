#ifndef MLP_H
#define MLP_H

#include "hls_stream.h"
#include "hls_math.h"
#include "ap_int.h"
#include <cmath>
#include <algorithm>
#include <stdint.h>

// =======================================================================
//   Macros and Constants
// =======================================================================
#define SEQ_LEN 197
#define EMBED_DIM 768
#define MLP_DIM 3072
#define TILE_OUT_DIM 512       // MLP_DIM / 6 = 512 (Tile size for W1 output)
#define TILE_SIZE 64           // Tile size for W2 output rows
#define EMBED_DIM_package 96   // 768 / 8 = 96
#define MLP_DIM_package 384    // 3072 / 8 = 384
#define INT8_PACK 8
#define FP32_PACK 8
#define MLP1_NUM_TILES 6       // 3072 / 512
#define MLP2_NUM_TILES 12      // 768 / 64
#define EPSILON 1e-5f
#define EMBED_DIM_INV (1.0f / 768.0f)

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
extern "C" {
    void mlp(
    // Input Feature Map (197 * 768)
    uint256_dt *x,
    
    // LayerNorm Weights
    uint256_dt *gamma,
    uint256_dt *beta,
    
    // MLP Layer 1 & 2 Weights
    uint256_dt *W1,
    uint256_dt *W2,
    
    // Biases
    float *b1,
    float *b2,
    
    // Scales
    float *scale_w1,
    float *scale_w2,
    
    // Final Output (197 * 768)
    uint256_dt *out,
    uint256_dt *HBM_FP32_Intermediate
);
}

// Memory Loading Helpers
void load_X_HBM2FIFO(uint256_dt *HBM_X, hls::stream<uint256_dt> &FIFO_X);
void load_X_FIFO2BRAM(hls::stream<uint256_dt> &FIFO_X, float BRAM_X[EMBED_DIM]);
void load_beta_gamma_HBM2SRAM(uint256_dt HBM_gamma[EMBED_DIM / 8], uint256_dt HBM_beta[EMBED_DIM / 8], float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM]);
void load_Bias_HBM2BRAM(float *HBM_b, float BRAM_b[EMBED_DIM]);
void load_Bias_HBM2BRAM_MLP(float *HBM_b, float BRAM_b[MLP_DIM]);
void load_W1_tile_HBM2FIFO(uint256_dt *HBM_W1, hls::stream<uint256_dt> &FIFO_W1, int iter);
void load_W2_tile_HBM2FIFO(uint256_dt *HBM_W2, hls::stream<uint256_dt> &FIFO_W2, int iter);
void load_Residual_Tile_HBM2FIFO(uint256_dt *HBM_X, hls::stream<uint256_dt> &FIFO_Res, int iter);
void store_Final_Tile_FIFO2HBM(hls::stream<uint256_dt> &FIFO_Final_Out, uint256_dt *HBM_Out, int iter);

// LayerNorm
void compute_mean_variance(float BRAM_X[EMBED_DIM], float gamma[EMBED_DIM], float beta[EMBED_DIM], float scaler_array[SEQ_LEN], int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package], int seq_idx);
void layernorm_compute(hls::stream<uint256_dt> &FIFO_X, float scaler[SEQ_LEN], float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM], int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);
void do_layernorm_dataflow(uint256_dt *x, float SRAM_gamma[EMBED_DIM], float SRAM_beta[EMBED_DIM], float scaler_X[SEQ_LEN], int64_dt X_package[SEQ_LEN][EMBED_DIM_package]);

// MLP Layer 1 & GELU
inline float gelu_hw(float x);
void compute_linear_mlp1_tile(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], hls::stream<uint256_dt> &FIFO_W1, hls::stream<float> &FIFO_GELU_in, float BRAM_b1[MLP_DIM], float scale_w1[MLP_DIM], float scale_X[SEQ_LEN], int iter);
void process_mlp1_single_tile_dataflow(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, hls::stream<float> &FIFO_GELU_in, float BRAM_b1[MLP_DIM], float scale_w1[MLP_DIM], float scale_X[SEQ_LEN], int iter);
void run_full_mlp_layer1(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, hls::stream<float> &FIFO_GELU_in, float *HBM_b1, float *HBM_scale_w1, float scale_X[SEQ_LEN]);
void run_mlp1_and_gelu_dataflow(int64_dt X_package[SEQ_LEN][EMBED_DIM_package], uint256_dt *HBM_W1, float *HBM_b1, float *HBM_scale_w1, float scale_X[SEQ_LEN], int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN]);
// void compute_gelu_and_buffer_fp32(hls::stream<float> &FIFO_GELU_in, int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN]);
// void quantize_mlp1_output(int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], float global_max[SEQ_LEN], int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], float scaler_MLP1[SEQ_LEN]);

// MLP Layer 2 & Residual
void compute_mlp2_proj_and_residual(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], hls::stream<uint256_dt> &FIFO_W2, hls::stream<uint256_dt> &FIFO_Res, hls::stream<uint256_dt> &FIFO_Final_Out, float scaler_MLP1[SEQ_LEN], float BRAM_b2[EMBED_DIM], float scale_w2[EMBED_DIM], int iter);
void process_mlp2_single_tile_dataflow(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], uint256_dt *HBM_W2, uint256_dt *HBM_X_Res, uint256_dt *HBM_Out, float scaler_MLP1[SEQ_LEN], float BRAM_b2[EMBED_DIM], float scale_w2[EMBED_DIM], int iter);
void run_full_mlp_layer2(int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], uint256_dt *HBM_W2, uint256_dt *HBM_X_Res, uint256_dt *HBM_Out, float scaler_MLP1[SEQ_LEN], float *HBM_b2, float *HBM_scale_w2);

void compute_gelu_and_write_FIFO(
    hls::stream<float> &FIFO_GELU_in,
    hls::stream<uint256_dt> &FIFO_GELU_out, 
    float global_max[SEQ_LEN]
);

void store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint256_dt *HBM_FP32_Intermediate
);

void load_FP32_HBM2FIFO(
    uint256_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
);

void quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM / INT8_PACK],
    float scaler_MLP1[SEQ_LEN]
);

void run_mlp1_and_gelu_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package], // Input features (quantized)
    uint256_dt *HBM_W1,                             // MLP1 Weights
    float *HBM_b1,                                  // MLP1 Biases
    float *HBM_scale_w1,                            // MLP1 Weight Scales
    float scale_X[SEQ_LEN],                         // Input Feature Scales
    uint256_dt *HBM_FP32_Intermediate,              // Output: FP32 buffer in HBM
    float global_max[SEQ_LEN]                       // Output: Max values for quant
);

void run_quantize_mlp1_dataflow(
    uint256_dt *HBM_FP32_Intermediate,              // Input: FP32 buffer in HBM
    float global_max[SEQ_LEN],                      // Input: Max values from Phase 1
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM / INT8_PACK], // Output: Quantized URAM
    float scaler_MLP1[SEQ_LEN]                      // Output: Quantization scales
);

void load_w2_and_res_sequential(
    uint256_dt *HBM_W2,
    uint256_dt *HBM_X_Res,
    hls::stream<uint256_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
);

#endif // MLP_H