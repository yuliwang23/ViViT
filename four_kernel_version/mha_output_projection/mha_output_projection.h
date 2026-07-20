#ifndef MHA_OUTPUT_PROJECTION_H
#define MHA_OUTPUT_PROJECTION_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

typedef ap_uint<512> uint512_dt;

typedef ap_int<8> int8_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<64> int64_dt;

constexpr int SEQ_LEN = 197;
constexpr int EMBED_DIM = 768;
constexpr int TILE_SIZE = 64;

constexpr int INT8_PACK = 8;
constexpr int FP32_PER_512 = 16;
constexpr int INT8_PER_512 = 64;
constexpr int INT64_PER_512 = 8;

constexpr int OPROJ_INPUT_PARALLEL = 128;

constexpr int OPROJ_WORDS_PER_STEP =
    OPROJ_INPUT_PARALLEL / INT8_PER_512;

constexpr int OPROJ_DOT_STEPS =
    EMBED_DIM / OPROJ_INPUT_PARALLEL;

constexpr int OPROJ_INT8_WORDS_PER_ROW =
    EMBED_DIM / INT8_PER_512;

constexpr int OPROJ_ATTENTION_TOTAL_WORDS =
    SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW;

constexpr int FP32_WORDS_PER_ROW =
    EMBED_DIM / FP32_PER_512;

constexpr int FP32_WORDS_PER_TILE =
    TILE_SIZE / FP32_PER_512;

constexpr float EPSILON = 1.0e-5f;

static_assert(
    EMBED_DIM % FP32_PER_512 == 0,
    "EMBED_DIM must be divisible by FP32_PER_512"
);

static_assert(
    EMBED_DIM % INT8_PER_512 == 0,
    "EMBED_DIM must be divisible by INT8_PER_512"
);

static_assert(
    INT64_PER_512 * 64 == 512,
    "INT64_PER_512 packing is inconsistent"
);

static_assert(
    TILE_SIZE % FP32_PER_512 == 0,
    "TILE_SIZE must be divisible by FP32_PER_512"
);

static_assert(
    EMBED_DIM % TILE_SIZE == 0,
    "EMBED_DIM must be divisible by TILE_SIZE"
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

extern "C" {

void mha_output_projection(
    uint512_dt *HBM_Attention_Out,
    float *HBM_Attention_Max,
    uint512_dt *Wo,
    uint512_dt *x,
    float *bo,
    float *scale_wo,
    uint512_dt *out
);

}

void load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
);

void load_attention_max_HBM2BRAM(
    const float *HBM_Attention_Max,
    float BRAM_Attention_Max[SEQ_LEN]
);

void prepare_attention_quant_params(
    float attention_max[SEQ_LEN],
    float quant_scale[SEQ_LEN],
    float out_scaler[SEQ_LEN]
);

void load_quantize_attention_HBM2URAM(
    const uint512_dt *HBM_Attention_Out,
    float quant_scale[SEQ_LEN],
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ]
);

void load_Weight_tile_HBM2FIFO(
    const uint512_dt *HBM_Wo,
    hls::stream<uint512_dt> &FIFO_Wo,
    int iter
);

void load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X_Res,
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
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,
    int iter
);

void process_o_proj_single_tile_dataflow(
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ],
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],
    int iter
);

void run_full_output_projection(
    const uint512_dt *HBM_Attention_Out,
    const float *HBM_Attention_Max,
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    const float *HBM_bo,
    const float *HBM_scale_wo
);

#endif
