/*#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

constexpr int SEQ_LEN = 14 * 14 + 1; // 14*14*16 --> per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int tile_size_dim = 64;
constexpr int TILE_SIZE = 64;
constexpr int tile_size_sql = 64;
constexpr int tile_sql_time = SEQ_LEN / tile_size_sql;
constexpr int tile_sql_remain = SEQ_LEN % tile_size_sql;
constexpr int DSP_PER_CYCLE = 16;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<8> uint8_dt;

extern "C" {
void linear_qkv_multihead(
    int8_t X[SEQ_LEN][EMBED_DIM],
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    float bq[EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    float bk[EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    float bv[EMBED_DIM],
    float scale_input[SEQ_LEN],
    float scale_wq[EMBED_DIM],
    float scale_wk[EMBED_DIM],
    float scale_wv[EMBED_DIM],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM]
);
}

void weight_streamer(
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv
);

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

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE]
);

void compute(
    int8_t BRAM_X [SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    float BRAM_scale_input[SEQ_LEN],
    float BRAM_scale_wq[EMBED_DIM],
    float BRAM_scale_wk[EMBED_DIM],
    float BRAM_scale_wv[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out
);

void task_compute_tile(
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM],
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE],
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv
);

void task_stream_out(
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    float BRAM_scale_input[SEQ_LEN],
    float BRAM_scale_wq[EMBED_DIM],
    float BRAM_scale_wk[EMBED_DIM],
    float BRAM_scale_wv[EMBED_DIM],
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    int j_tile,
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out
);

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    float Output_Global[NUM_HEADS][SEQ_LEN][HEAD_DIM]
);*/
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <stdint.h>

#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

constexpr int SEQ_LEN = 14 * 14 + 1; // 14*14*16 ¡÷ per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int tile_size_dim = 64;
constexpr int TILE_SIZE = 64;
constexpr int tile_size_sql = 64;
constexpr int tile_sql_time = SEQ_LEN / tile_size_sql;
constexpr int tile_sql_remain = SEQ_LEN % tile_size_sql;
constexpr int DSP_PER_CYCLE = 16;

typedef ap_uint<256> uint256_dt;
typedef ap_uint<8> uint8_dt;

extern "C"
{
    void linear_qkv_multihead(
        int8_t X[SEQ_LEN][EMBED_DIM],
        int8_t Wq[EMBED_DIM][EMBED_DIM],
        float bq[EMBED_DIM],
        int8_t Wk[EMBED_DIM][EMBED_DIM],
        float bk[EMBED_DIM],
        int8_t Wv[EMBED_DIM][EMBED_DIM],
        float bv[EMBED_DIM],
        float scale_input[SEQ_LEN],
        float scale_wq[EMBED_DIM],
        float scale_wk[EMBED_DIM],
        float scale_wv[EMBED_DIM],
        float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
        float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
        float V[NUM_HEADS][SEQ_LEN][HEAD_DIM]);
}

void weight_streamer(
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv);

void load_X_HBM2FIFO(int8_t X[SEQ_LEN][EMBED_DIM], hls::stream<uint256_dt> &FIFO_X);
void load_X_FIFO2BRAM(hls::stream<uint256_dt> &FIFO_X, int8_t BRAM_X[SEQ_LEN][EMBED_DIM]);

void load_W_HBM2FIFO(int8_t W[EMBED_DIM][EMBED_DIM], hls::stream<uint256_dt> &FIFO_W,
                     int j_offset, int k_offset);
void load_W_FIFO2BRAM(hls::stream<uint256_dt> &FIFO_W, int8_t BRAM_W[TILE_SIZE][TILE_SIZE]);

void load_Scale_Input_HBM2BRAM(float s_hbm[SEQ_LEN], float BRAM_s[SEQ_LEN]);

void load_Vec_HBM2BRAM(float v_hbm[EMBED_DIM], float BRAM_v[EMBED_DIM]);

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE]);

void compute(
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    float BRAM_scale_input[SEQ_LEN],
    float BRAM_scale_wq[EMBED_DIM],
    float BRAM_scale_wk[EMBED_DIM],
    float BRAM_scale_wv[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out);

void task_compute_tile(
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM],
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE],
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv);

void task_stream_out(
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    float BRAM_scale_input[SEQ_LEN],
    float BRAM_scale_wq[EMBED_DIM],
    float BRAM_scale_wk[EMBED_DIM],
    float BRAM_scale_wv[EMBED_DIM],
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    int j_tile,
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out);

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    float Output_Global[NUM_HEADS][SEQ_LEN][HEAD_DIM]);