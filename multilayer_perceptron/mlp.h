#ifndef MLP_H
#define MLP_H

// =======================================================================
//   Includes
// =======================================================================

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

// =======================================================================
//   Data Types
// =======================================================================

// External HBM / AXI data width
typedef ap_uint<512> uint512_dt;

// Internal FP32 packing width:
// 8 × FP32 = 256 bits
typedef ap_uint<256> uint256_dt;

// Eight INT8 values packed into 64 bits.
//
// This is an unsigned container. Each individual 8-bit field is
// converted to int8_dt before arithmetic.
typedef ap_uint<64> int64_dt;

// Signed accumulator
typedef ap_int<32> int32_dt;

// Signed INT8 data
typedef ap_int<8> int8_dt;

// =======================================================================
//   Model Dimensions
// =======================================================================

static constexpr int SEQ_LEN   = 197;
static constexpr int EMBED_DIM = 768;
static constexpr int MLP_DIM   = 3072;

// Number of INT8 values stored in one 64-bit package
static constexpr int INT8_PACK = 8;

// Number of FP32 values stored in one internal 256-bit packet
static constexpr int FP32_PACK = 8;

// Number of 64-bit INT8 packages per embedding vector
static constexpr int EMBED_DIM_package =
    EMBED_DIM / INT8_PACK;

// Number of 64-bit INT8 packages per MLP hidden vector
static constexpr int MLP_DIM_package =
    MLP_DIM / INT8_PACK;

// =======================================================================
//   Tile Configuration
// =======================================================================

// MLP1 produces 64 output channels per tile
static constexpr int TILE_OUT_DIM = 64;

// MLP2 produces 64 output channels per tile
static constexpr int TILE_SIZE = 64;

// Number of MLP1 tiles:
// 3072 / 64 = 48
static constexpr int MLP1_NUM_TILES =
    MLP_DIM / TILE_OUT_DIM;

// Number of MLP2 tiles:
// 768 / 64 = 12
static constexpr int MLP2_NUM_TILES =
    EMBED_DIM / TILE_SIZE;

// =======================================================================
//   Compile-Time Configuration Checks
// =======================================================================

static_assert(INT8_PACK == 8,
              "INT8_PACK must be 8 for a 64-bit INT8 package");

static_assert(FP32_PACK == 8,
              "FP32_PACK must be 8 for a 256-bit FP32 packet");

static_assert(EMBED_DIM % INT8_PACK == 0,
              "EMBED_DIM must be divisible by INT8_PACK");

static_assert(MLP_DIM % INT8_PACK == 0,
              "MLP_DIM must be divisible by INT8_PACK");

static_assert(EMBED_DIM % 16 == 0,
              "EMBED_DIM must be divisible by 16");

static_assert(MLP_DIM % 16 == 0,
              "MLP_DIM must be divisible by 16");

static_assert(EMBED_DIM % 64 == 0,
              "EMBED_DIM must be divisible by 64 for 512-bit INT8 access");

static_assert(MLP_DIM % 64 == 0,
              "MLP_DIM must be divisible by 64 for 512-bit INT8 access");

static_assert(EMBED_DIM_package == EMBED_DIM / INT8_PACK,
              "Invalid EMBED_DIM_package");

static_assert(MLP_DIM_package == MLP_DIM / INT8_PACK,
              "Invalid MLP_DIM_package");

static_assert(EMBED_DIM_package % 16 == 0,
              "EMBED_DIM_package must be divisible by 16");

static_assert(MLP_DIM_package % 16 == 0,
              "MLP_DIM_package must be divisible by 16");

static_assert(MLP_DIM % TILE_OUT_DIM == 0,
              "MLP_DIM must be divisible by TILE_OUT_DIM");

static_assert(EMBED_DIM % TILE_SIZE == 0,
              "EMBED_DIM must be divisible by TILE_SIZE");

static_assert(TILE_OUT_DIM % INT8_PACK == 0,
              "TILE_OUT_DIM must be divisible by INT8_PACK");

static_assert(TILE_OUT_DIM % FP32_PACK == 0,
              "TILE_OUT_DIM must be divisible by FP32_PACK");

static_assert(TILE_SIZE % FP32_PACK == 0,
              "TILE_SIZE must be divisible by FP32_PACK");

static_assert(TILE_SIZE % 16 == 0,
              "TILE_SIZE must be divisible by 16");

static_assert(MLP1_NUM_TILES == MLP_DIM / TILE_OUT_DIM,
              "Invalid MLP1_NUM_TILES");

static_assert(MLP2_NUM_TILES == EMBED_DIM / TILE_SIZE,
              "Invalid MLP2_NUM_TILES");

// =======================================================================
//   Numeric Constants
// =======================================================================

static constexpr float EPSILON = 1.0e-5f;

static constexpr float EMBED_DIM_INV =
    1.0f / static_cast<float>(EMBED_DIM);

// =======================================================================
//   Top Function
// =======================================================================

extern "C" {

void mlp(
    uint512_dt *x,

    uint512_dt *gamma,
    uint512_dt *beta,

    uint512_dt *W1,
    uint512_dt *W2,

    float *b1,
    float *b2,

    float *scale_w1,
    float *scale_w2,

    uint512_dt *out,

    uint512_dt *HBM_FP32_Intermediate
);

}

// =======================================================================
//   General Memory Loading Functions
// =======================================================================

void load_X_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
);

void load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
);

void load_beta_gamma_HBM2SRAM(
    const uint512_dt HBM_gamma[EMBED_DIM / 16],
    const uint512_dt HBM_beta[EMBED_DIM / 16],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
);

void load_Bias_HBM2BRAM_MLP(
    const float *HBM_b,
    float BRAM_b[MLP_DIM]
);

void load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
);

// =======================================================================
//   LayerNorm Functions
// =======================================================================

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array
        [SEQ_LEN][EMBED_DIM_package],
    int seq_idx
);

void layernorm_compute(
    hls::stream<uint512_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package]
);

void do_layernorm_dataflow(
    const uint512_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package]
);

// =======================================================================
//   MLP Layer 1 Weight Loading
// =======================================================================

void load_W1_tile_HBM2FIFO(
    const uint512_dt *HBM_W1,
    hls::stream<uint512_dt> &FIFO_W1,
    int iter
);

// =======================================================================
//   MLP Layer 1 Compute
// =======================================================================

void compute_linear_mlp1_tile(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],

    hls::stream<uint512_dt> &FIFO_W1,
    hls::stream<float> &FIFO_GELU_in,

    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],

    int iter
);

void process_mlp1_single_tile_dataflow(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],

    const uint512_dt *HBM_W1,

    hls::stream<float> &FIFO_GELU_in,

    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],

    int iter
);

void run_full_mlp_layer1(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],

    const uint512_dt *HBM_W1,

    hls::stream<float> &FIFO_GELU_in,

    const float *HBM_b1,
    const float *HBM_scale_w1,

    float scale_X[SEQ_LEN]
);

// =======================================================================
//   GELU
// =======================================================================

float gelu_hw(float x);

void compute_gelu_and_write_FIFO(
    hls::stream<float> &FIFO_GELU_in,
    hls::stream<uint256_dt> &FIFO_GELU_out,
    float global_max[SEQ_LEN]
);

// =======================================================================
//   MLP1 Phase 1:
//   MLP1 -> GELU -> HBM
// =======================================================================

void store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint512_dt *HBM_FP32_Intermediate
);

void run_mlp1_and_gelu_dataflow(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],

    const uint512_dt *HBM_W1,

    const float *HBM_b1,
    const float *HBM_scale_w1,

    float scale_X[SEQ_LEN],

    uint512_dt *HBM_FP32_Intermediate,

    float global_max[SEQ_LEN]
);

// =======================================================================
//   MLP1 Phase 2:
//   HBM -> Dynamic INT8 Quantization
// =======================================================================

void load_FP32_HBM2FIFO(
    const uint512_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
);

void quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,

    float global_max[SEQ_LEN],

    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    float scaler_MLP1[SEQ_LEN]
);

void run_quantize_mlp1_dataflow(
    const uint512_dt *HBM_FP32_Intermediate,

    float global_max[SEQ_LEN],

    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    float scaler_MLP1[SEQ_LEN]
);

// =======================================================================
//   MLP Layer 2 Memory Loading
// =======================================================================

void load_W2_tile_HBM2FIFO(
    const uint512_dt *HBM_W2,
    hls::stream<uint512_dt> &FIFO_W2,
    int iter
);

void load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
);

void store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint512_dt *HBM_Out,
    int iter
);

void load_w2_and_res_sequential(
    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,

    int iter
);

// =======================================================================
//   MLP Layer 2 Compute
// =======================================================================

void compute_mlp2_proj_and_residual(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    hls::stream<uint256_dt> &FIFO_Final_Out,

    float scaler_MLP1[SEQ_LEN],
    float BRAM_b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],

    int iter
);

void process_mlp2_single_tile_dataflow(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,

    uint512_dt *HBM_Out,

    float scaler_MLP1[SEQ_LEN],
    float BRAM_b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],

    int iter
);

void run_full_mlp_layer2(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,

    uint512_dt *HBM_Out,

    float scaler_MLP1[SEQ_LEN],

    const float *HBM_b2,
    const float *HBM_scale_w2
);

#endif  // MLP_H