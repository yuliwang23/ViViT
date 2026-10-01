#ifndef FFN1_H
#define FFN1_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<64> int64_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<8> int8_dt;

static constexpr int SEQ_LEN =
    197;

static constexpr int EMBED_DIM =
    768;

static constexpr int MLP_DIM =
    3072;

static constexpr int INT8_PACK =
    8;

static constexpr int FP32_PACK =
    8;

static constexpr int EMBED_DIM_package =
    EMBED_DIM /
    INT8_PACK;

static constexpr int TILE_OUT_DIM =
    64;

static constexpr int MLP1_NUM_TILES =
    MLP_DIM /
    TILE_OUT_DIM;

static constexpr float EPSILON =
    1.0e-5f;

static constexpr float EMBED_DIM_INV =
    1.0f /
    static_cast<float>(EMBED_DIM);

static_assert(
    INT8_PACK == 8,
    "INT8_PACK must be 8"
);

static_assert(
    FP32_PACK == 8,
    "FP32_PACK must be 8"
);

static_assert(
    EMBED_DIM % INT8_PACK == 0,
    "EMBED_DIM must be divisible by INT8_PACK"
);

static_assert(
    EMBED_DIM % 16 == 0,
    "EMBED_DIM must be divisible by 16"
);

static_assert(
    EMBED_DIM % 64 == 0,
    "EMBED_DIM must be divisible by 64"
);

static_assert(
    MLP_DIM % 16 == 0,
    "MLP_DIM must be divisible by 16"
);

static_assert(
    MLP_DIM % 64 == 0,
    "MLP_DIM must be divisible by 64"
);

static_assert(
    EMBED_DIM_package % 16 == 0,
    "EMBED_DIM_package must be divisible by 16"
);

static_assert(
    MLP_DIM % TILE_OUT_DIM == 0,
    "MLP_DIM must be divisible by TILE_OUT_DIM"
);

static_assert(
    TILE_OUT_DIM % INT8_PACK == 0,
    "TILE_OUT_DIM must be divisible by INT8_PACK"
);

static_assert(
    TILE_OUT_DIM % FP32_PACK == 0,
    "TILE_OUT_DIM must be divisible by FP32_PACK"
);

static_assert(
    MLP1_NUM_TILES ==
        MLP_DIM / TILE_OUT_DIM,
    "Invalid MLP1_NUM_TILES"
);

extern "C" {

void FFN1(
    uint512_dt *x,
    uint512_dt *gamma,
    uint512_dt *beta,
    uint512_dt *W1,
    float *b1,
    float *scale_w1,
    uint512_dt *HBM_FP32_Intermediate,
    float *HBM_GELU_Max
);

}

void ffn1_load_X_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
);

void ffn1_load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
);

void ffn1_load_beta_gamma_HBM2SRAM(
    const uint512_dt HBM_gamma[EMBED_DIM / 16],
    const uint512_dt HBM_beta[EMBED_DIM / 16],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
);

void ffn1_load_Bias_HBM2BRAM_MLP(
    const float *HBM_b,
    float BRAM_b[MLP_DIM]
);

void ffn1_load_W1_tile_HBM2FIFO(
    const uint512_dt *HBM_W1,
    hls::stream<uint512_dt> &FIFO_W1,
    int iter
);

void ffn1_compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package],
    int seq_idx
);

void ffn1_layernorm_compute(
    hls::stream<uint512_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]
);

void ffn1_do_layernorm_dataflow(
    const uint512_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]
);

void ffn1_compute_linear_mlp1_tile(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    hls::stream<uint512_dt> &FIFO_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
);

void ffn1_process_mlp1_single_tile_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
);

void ffn1_run_full_mlp_layer1(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    const float *HBM_b1,
    const float *HBM_scale_w1,
    float scale_X[SEQ_LEN]
);

float ffn1_gelu_hw(float x);

void ffn1_compute_gelu_and_write_FIFO(
    hls::stream<float> &FIFO_GELU_in,
    hls::stream<uint256_dt> &FIFO_GELU_out,
    float global_max[SEQ_LEN]
);

void ffn1_store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint512_dt *HBM_FP32_Intermediate
);

void ffn1_run_mlp1_and_gelu_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    const float *HBM_b1,
    const float *HBM_scale_w1,
    float scale_X[SEQ_LEN],
    uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN]
);
    
void ffn1_store_GELU_max_HBM(
    float global_max[SEQ_LEN],
    float *HBM_GELU_Max
);

void ffn2_load_GELU_max_HBM2BRAM(
    const float *HBM_GELU_Max,
    float global_max[SEQ_LEN]
);

void ffn2_load_FP32_HBM2FIFO(
    const uint512_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
);

void ffn2_quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    float scaler_MLP1[SEQ_LEN]
);

void ffn2_run_quantize_mlp1_dataflow(
    const uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    float scaler_MLP1[SEQ_LEN]
);

void ffn2_load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
);

void ffn2_load_W2_tile_HBM2FIFO(
    const uint512_dt *HBM_W2,
    hls::stream<uint512_dt> &FIFO_W2,
    int iter
);

void ffn2_load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
);

void ffn2_store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint512_dt *HBM_Out,
    int iter
);

void ffn2_compute_mlp2_proj_and_residual(
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

void ffn2_load_w2_and_res_sequential(
    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,
    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
);

void ffn2_process_mlp2_single_tile_dataflow(
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

void ffn2_run_full_mlp_layer2(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    float scaler_MLP1[SEQ_LEN],
    const float *HBM_b2,
    const float *HBM_scale_w2
);


#endif
