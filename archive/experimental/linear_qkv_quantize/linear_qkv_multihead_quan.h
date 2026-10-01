#include <stdint.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

// ============================================================
// Design parameters
// ============================================================
constexpr int SEQ_LEN   = 14 * 14 + 1;          // 197
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM  = EMBED_DIM / NUM_HEADS; // 64
constexpr int TILE_SIZE = 64;                    // == HEAD_DIM

// FIFO depth constants (also defined in .cpp, kept here for reference)
// FIFO_W_DEPTH   = (EMBED_DIM/TILE_SIZE) * TILE_SIZE * (TILE_SIZE/32)
//                = 12 * 64 * 2 = 1536
// FIFO_OUT_DEPTH = SEQ_LEN * (TILE_SIZE/8)
//                = 197 * 8 = 1576
#define FIFO_W_DEPTH   1536
#define FIFO_OUT_DEPTH 1576

typedef ap_uint<256> uint256_dt;

// ============================================================
// Top-level kernel
// ============================================================
extern "C" {
void linear_qkv_multihead(
    int8_t X          [SEQ_LEN ][EMBED_DIM],
    int8_t Wq         [EMBED_DIM][EMBED_DIM],
    float  bq         [EMBED_DIM],
    int8_t Wk         [EMBED_DIM][EMBED_DIM],
    float  bk         [EMBED_DIM],
    int8_t Wv         [EMBED_DIM][EMBED_DIM],
    float  bv         [EMBED_DIM],
    float  scale_input[SEQ_LEN],
    float  scale_wq   [EMBED_DIM],
    float  scale_wk   [EMBED_DIM],
    float  scale_wv   [EMBED_DIM],
    float  Q          [NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float  K          [NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float  V          [NUM_HEADS][SEQ_LEN][HEAD_DIM]);
}

// ============================================================
// Pipeline stage functions (called only from top-level DATAFLOW)
// ============================================================
void stage_weight_stream(
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv);

void stage_compute(
    int8_t  BRAM_X[SEQ_LEN][EMBED_DIM],
    float   BRAM_bq[EMBED_DIM],
    float   BRAM_bk[EMBED_DIM],
    float   BRAM_bv[EMBED_DIM],
    float   BRAM_scale_input[SEQ_LEN],
    float   BRAM_scale_wq[EMBED_DIM],
    float   BRAM_scale_wk[EMBED_DIM],
    float   BRAM_scale_wv[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out);

void stage_store(
    hls::stream<uint256_dt> &FIFO_Q,
    hls::stream<uint256_dt> &FIFO_K,
    hls::stream<uint256_dt> &FIFO_V,
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM]);

// ============================================================
// Internal helper functions
// ============================================================
void load_X_HBM2FIFO(
    int8_t X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X);

void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM]);

void load_Weight_tile_HBM2FIFO(
    int8_t W[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset,
    int col_offset);

void load_Weight_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_W,
    int8_t BRAM_W[TILE_SIZE][TILE_SIZE]);

void load_Scale_Input_HBM2BRAM(
    float s_hbm[SEQ_LEN],
    float BRAM_s[SEQ_LEN]);

void load_Vec_HBM2BRAM(
    float v_hbm[EMBED_DIM],
    float BRAM_v[EMBED_DIM]);