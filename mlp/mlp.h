#ifndef MLP_HLS_H
#define MLP_HLS_H

#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <cstdint>

typedef ap_uint<256> uint256_dt; // 8 x float32  OR  32 x int8
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

#define GELU_LUT_SIZE 512
#define GELU_X_MIN (-8.0f)
#define GELU_X_MAX (8.0f)
#define GELU_INV_STEP ((float)GELU_LUT_SIZE / (GELU_X_MAX - GELU_X_MIN))

#define EPSILON 1e-5f
#define EMBED_DIM_INV (1.0f / (float)EMBED_DIM)
#define HIDDEN_DIM_INV (1.0f / (float)HIDDEN_DIM)

extern "C"
{
    void mlp_q(
        uint256_dt *input,   // int8 packed LN output, 32 values/chunk
        float *input_scaler, // dequant scaler per source token
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

void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X);

void load_Bias_HBM2BRAM(
    float *b,
    float BRAM_b[],
    int N);

void load_Weight_tile_HBM2FIFO(
    uint256_dt *W,
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset,
    int K_dim); // = inner-dim of this matmul

// Quantize one FP32 row -> int8 packed; returns dequant scaler (= 1/scale).
void quantize_row_dynamic(
    float row_fp[],
    int64_dt row_pkg[],
    int length,
    float *out_scaler);

void dense0_one_tile(
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    float scaler_X[SEQ_LEN_PADDED],
    hls::stream<uint256_dt> &FIFO_W1,
    float SRAM_b1[HIDDEN_DIM],
    float SRAM_sw1[HIDDEN_DIM],
    half_dt H_full_fp[SEQ_LEN_PADDED][HIDDEN_DIM],
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

// LUT lookup with linear interpolation
float gelu_lut_lookup(float x);

#endif