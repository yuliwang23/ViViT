#ifndef MHA_ATTENTION_H
#define MMA_ATTENTION_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>


#define QK_HEAD_FIFO_DEPTH 1576
#define V_HEAD_FIFO_DEPTH 1600
#define QK_SCALE_FIFO_DEPTH 197
#define V_SCALE_FIFO_DEPTH 64

#define SOFTMAX_ACC_SLOTS 16
#define ROW_MAX_ACC_SLOTS 8
#define SV_PACKS_PER_CYCLE 5
#define SV_CHANNELS_PER_CYCLE 2
#define SOFTMAX_VALUES_PER_CYCLE (SV_PACKS_PER_CYCLE * INT8_PACK)
#define SOFTMAX_PACKED_FIFO_DEPTH (2 * (SEQ_LEN_PADDED / INT8_PACK))

typedef ap_uint<512> uint512_dt;

typedef ap_int<8> int8_dt;

typedef ap_int<16> int16_dt;
typedef ap_int<28> int28_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<64> int64_dt;

constexpr int SEQ_LEN   = 197;
constexpr int EMBED_DIM = 768;

constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM  = EMBED_DIM / NUM_HEADS;

constexpr int TILE_SIZE = 64;

constexpr int SEQ_LEN_PADDED = 200;

constexpr int INT8_PACK = 8;

constexpr int EMBED_DIM_package =
    EMBED_DIM / INT8_PACK;

constexpr int FP32_PER_512 = 16;

constexpr int INT8_PER_512 = 64;

constexpr int INT64_PER_512 = 8;

constexpr int QKV_INPUT_PARALLEL =
    96;

constexpr int QKV_PACKS_PER_STEP =
    QKV_INPUT_PARALLEL / INT8_PACK;

constexpr int QKV_DOT_STEPS =
    EMBED_DIM / QKV_INPUT_PARALLEL;

constexpr int QKV_MEMORY_BANKS =
    QKV_PACKS_PER_STEP / 2;

constexpr int OPROJ_INPUT_PARALLEL =
    128;

constexpr int OPROJ_WORDS_PER_STEP =
    OPROJ_INPUT_PARALLEL / INT8_PER_512;

constexpr int OPROJ_DOT_STEPS =
    EMBED_DIM / OPROJ_INPUT_PARALLEL;

constexpr int OPROJ_INT8_WORDS_PER_ROW =
    EMBED_DIM / INT8_PER_512;

constexpr int OPROJ_ATTENTION_TOTAL_WORDS =
    SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW;

constexpr int ATTN_WORDS_PER_TOKEN =
    EMBED_DIM / FP32_PER_512;

constexpr int ATTN_WORDS_PER_HEAD =
    HEAD_DIM / FP32_PER_512;

constexpr int WEIGHT_WORDS_PER_ROW =
    EMBED_DIM / INT8_PER_512;

constexpr int FP32_WORDS_PER_TILE =
    TILE_SIZE / FP32_PER_512;

constexpr float NEG_INF = -1.0e9f;
constexpr float EPSILON = 1.0e-5f;

constexpr float EMBED_DIM_INV =
    1.0f / static_cast<float>(EMBED_DIM);

constexpr float ATTENTION_SCALE =
    0.125f;

static_assert(
    EMBED_DIM == NUM_HEADS * HEAD_DIM,
    "EMBED_DIM must equal NUM_HEADS * HEAD_DIM"
);

static_assert(
    EMBED_DIM % INT8_PACK == 0,
    "EMBED_DIM must be divisible by INT8_PACK"
);

static_assert(
    EMBED_DIM % FP32_PER_512 == 0,
    "EMBED_DIM must be divisible by FP32_PER_512"
);

static_assert(
    EMBED_DIM % INT8_PER_512 == 0,
    "EMBED_DIM must be divisible by INT8_PER_512"
);

static_assert(
    HEAD_DIM % FP32_PER_512 == 0,
    "HEAD_DIM must be divisible by FP32_PER_512"
);

static_assert(
    TILE_SIZE % FP32_PER_512 == 0,
    "TILE_SIZE must be divisible by FP32_PER_512"
);

static_assert(
    TILE_SIZE == HEAD_DIM,
    "Current implementation requires TILE_SIZE == HEAD_DIM"
);

static_assert(
    SEQ_LEN_PADDED % INT8_PACK == 0,
    "SEQ_LEN_PADDED must be divisible by INT8_PACK"
);

static_assert(
    QKV_INPUT_PARALLEL % INT8_PACK == 0,
    "QKV_INPUT_PARALLEL must be divisible by INT8_PACK"
);

static_assert(
    EMBED_DIM % QKV_INPUT_PARALLEL == 0,
    "EMBED_DIM must be divisible by QKV_INPUT_PARALLEL"
);

static_assert(
    QKV_PACKS_PER_STEP == 12,
    "Current QKV implementation requires 12 packages per step"
);

static_assert(
    QKV_MEMORY_BANKS == 6,
    "Current QKV implementation requires 6 memory banks"
);

static_assert(
    QKV_DOT_STEPS * QKV_INPUT_PARALLEL == EMBED_DIM,
    "QKV parallel configuration must cover the complete embedding dimension"
);

static_assert(
    QKV_MEMORY_BANKS * 2 == QKV_PACKS_PER_STEP,
    "QKV memory bank configuration must provide two packages per bank"
);

static_assert(
    OPROJ_INPUT_PARALLEL % INT8_PER_512 == 0,
    "OPROJ_INPUT_PARALLEL must be divisible by INT8_PER_512"
);

static_assert(
    EMBED_DIM % OPROJ_INPUT_PARALLEL == 0,
    "EMBED_DIM must be divisible by OPROJ_INPUT_PARALLEL"
);

static_assert(
    OPROJ_WORDS_PER_STEP == 2,
    "Current output projection implementation requires two 512-bit words per step"
);

static_assert(
    OPROJ_DOT_STEPS * OPROJ_INPUT_PARALLEL == EMBED_DIM,
    "Output projection parallel configuration must cover the complete embedding dimension"
);

static_assert(
    OPROJ_INT8_WORDS_PER_ROW ==
        OPROJ_DOT_STEPS * OPROJ_WORDS_PER_STEP,
    "Output projection word count configuration is inconsistent"
);

static_assert(
    QK_HEAD_FIFO_DEPTH ==
        SEQ_LEN * (HEAD_DIM / INT8_PACK),
    "QK_HEAD_FIFO_DEPTH does not match model dimensions"
);

static_assert(
    V_HEAD_FIFO_DEPTH ==
        HEAD_DIM * (SEQ_LEN_PADDED / INT8_PACK),
    "V_HEAD_FIFO_DEPTH does not match model dimensions"
);

static_assert(
    QK_SCALE_FIFO_DEPTH == SEQ_LEN,
    "QK_SCALE_FIFO_DEPTH does not match SEQ_LEN"
);

static_assert(
    V_SCALE_FIFO_DEPTH == HEAD_DIM,
    "V_SCALE_FIFO_DEPTH does not match HEAD_DIM"
);

static_assert(
    (SEQ_LEN_PADDED / INT8_PACK) % SV_PACKS_PER_CYCLE == 0,
    "SV_PACKS_PER_CYCLE must divide the number of V packs"
);

static_assert(
    SEQ_LEN_PADDED % SOFTMAX_VALUES_PER_CYCLE == 0,
    "SOFTMAX_VALUES_PER_CYCLE must divide SEQ_LEN_PADDED"
);

static_assert(
    (SOFTMAX_ACC_SLOTS & (SOFTMAX_ACC_SLOTS - 1)) == 0,
    "SOFTMAX_ACC_SLOTS must be a power of two"
);

static_assert(
    (ROW_MAX_ACC_SLOTS & (ROW_MAX_ACC_SLOTS - 1)) == 0,
    "ROW_MAX_ACC_SLOTS must be a power of two"
);

static_assert(
    HEAD_DIM % SV_CHANNELS_PER_CYCLE == 0,
    "SV_CHANNELS_PER_CYCLE must divide HEAD_DIM"
);


extern "C" {

void multihead_attention(
    uint512_dt *x,

    uint512_dt *gamma,
    uint512_dt *beta,

    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,
    uint512_dt *Wo,

    float *bq,
    float *bk,
    float *bv,
    float *bo,

    float *scale_wq,
    float *scale_wk,
    float *scale_wv,
    float *scale_wo,

    uint512_dt *HBM_Attention_Out,

    uint512_dt *out
);

}

void load_X_HBM2FIFO(
    uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
);

void load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
);

void load_beta_gamma_HBM2SRAM(
    uint512_dt HBM_gamma[
        EMBED_DIM / FP32_PER_512
    ],

    uint512_dt HBM_beta[
        EMBED_DIM / FP32_PER_512
    ],

    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
);

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],

    float scaler_array[SEQ_LEN],

    int64_dt X_package_array
        [SEQ_LEN]
        [EMBED_DIM_package],

    int seq_idx
);

void layernorm_compute(
    hls::stream<uint512_dt> &FIFO_X,

    float scaler[SEQ_LEN],

    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],

    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package]
);

void do_layernorm_dataflow(
    uint512_dt *x,

    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],

    float scaler_X[SEQ_LEN],

    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package]
);

void load_Bias_HBM2BRAM(
    float b[EMBED_DIM],
    float BRAM_b[EMBED_DIM]
);

void load_Weight_tile_HBM2FIFO(
    uint512_dt *W,

    hls::stream<uint512_dt> &FIFO_W,

    int iter
);

void load_All_Weights_Parallel(
    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

    int64_dt URAM_Wq
        [TILE_SIZE]
        [EMBED_DIM_package],

    int64_dt URAM_Wk
        [TILE_SIZE]
        [EMBED_DIM_package],

    int64_dt URAM_Wv
        [TILE_SIZE]
        [EMBED_DIM_package]
);

void load_qkv_weights_sequential(
    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,

    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

    int head_idx
);

void compute_linear_qkv_one_head(
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package],

    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

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

    int iter
);

void qkv_sram_and_quantize(
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,

    int64_dt URAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],

    float scaler_Q[SEQ_LEN],
    float scaler_K[SEQ_LEN],
    float scaler_V[HEAD_DIM]
);

void compute_score_and_softmax(
    int64_dt URAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],

    hls::stream<int8_dt> &FIFO_Softmax_out,

    hls::stream<float>
        &FIFO_Softmax_scaler_out
);

void compute_attention_output_one_head(
    hls::stream<int8_dt> &FIFO_Softmax_out,

    hls::stream<float>
        &FIFO_Softmax_scaler_out,

    int64_dt URAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],

    float scale_V[HEAD_DIM],

    uint512_dt *HBM_Attention_Out,

    float global_max[SEQ_LEN],

    int head_idx
);

void process_attention_one_head_dataflow(
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package],

    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,

    float SRAM_bq[EMBED_DIM],
    float SRAM_bk[EMBED_DIM],
    float SRAM_bv[EMBED_DIM],

    float SRAM_swq[EMBED_DIM],
    float SRAM_swk[EMBED_DIM],
    float SRAM_swv[EMBED_DIM],

    float scaler_X[SEQ_LEN],

    uint512_dt *HBM_Attention_Out,

    float global_max[SEQ_LEN],

    int head_idx
);

void prepare_attention_quant_params(
    float global_max[SEQ_LEN],
    float quant_scale[SEQ_LEN],
    float out_scaler[SEQ_LEN]
);

void load_quantize_attention_HBM2URAM(
    uint512_dt *HBM_Attention_Out,

    float quant_scale[SEQ_LEN],

    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ]
);

void load_Residual_Tile_HBM2FIFO(
    uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_Res,

    int iter
);

void store_Final_Tile_FIFO2HBM(
    hls::stream<uint512_dt> &FIFO_Final_Out,

    uint512_dt *HBM_Out,

    int iter
);

void compute_output_proj_and_residual(
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ],

    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,
    hls::stream<uint512_dt> &FIFO_Final_Out,

    float out_scaler[SEQ_LEN],

    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],

    int iter
);

void load_wo_residual_sequential(
    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,

    int iter
);

void process_o_proj_single_tile_dataflow(
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ],

    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float out_scaler[SEQ_LEN],

    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],

    int iter
);

void run_full_output_projection(
    uint512_dt *HBM_Attention_Out,
    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float global_max[SEQ_LEN],

    float *HBM_bo,
    float *HBM_scale_wo
);

void store_attention_max_HBM(
    float global_max[SEQ_LEN],
    float *HBM_Attention_Max
);

void process_all_heads_fifo_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,
    float SRAM_bq[EMBED_DIM],
    float SRAM_bk[EMBED_DIM],
    float SRAM_bv[EMBED_DIM],
    float SRAM_swq[EMBED_DIM],
    float SRAM_swk[EMBED_DIM],
    float SRAM_swv[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    uint512_dt *HBM_Attention_Out,
    float global_max[SEQ_LEN]
);


#endif