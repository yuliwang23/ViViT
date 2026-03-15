#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>


constexpr int SEQ_LEN = 14*14 + 1; // 14*14*16 --> per frame // 14*15
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
 
extern "C"{
void linear_qkv_multihead(
    int X[SEQ_LEN][EMBED_DIM],
    int Wq[EMBED_DIM][EMBED_DIM],
    int bq[EMBED_DIM],
    int Wk[EMBED_DIM][EMBED_DIM],
    int bk[EMBED_DIM],
    int Wv[EMBED_DIM][EMBED_DIM],
    int bv[EMBED_DIM],
    int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM]);

void load_X_HBM2FIFO(
    int X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X);  
void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    int BRAM_X[SEQ_LEN][EMBED_DIM]);
void load_Weight_Q_tile_HBM2FIFO(
    int Wq[EMBED_DIM][EMBED_DIM], 
    hls::stream<uint256_dt> &FIFO_Wq,
    int row_offset, // tile start row (in HBM)
    int col_offset  // tile start col (in HBM)
);
}

void load_Weight_Q_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wq,
    int BRAM_Wq[TILE_SIZE][TILE_SIZE]
);

void load_Bias_Q_HBM2BRAM(
    int bq[EMBED_DIM],
    int BRAM_bq[EMBED_DIM]
);


void load_Weight_K_tile_HBM2FIFO(
    int Wk[EMBED_DIM][EMBED_DIM], 
    hls::stream<uint256_dt> &FIFO_Wk,
    int row_offset,
    int col_offset
);

void load_Weight_K_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wk,
    int BRAM_Wk[TILE_SIZE][TILE_SIZE]
);

void load_Bias_K_HBM2BRAM(
    int bk[EMBED_DIM],
    int BRAM_bk[EMBED_DIM]
);

void load_Weight_V_tile_HBM2FIFO(
    int Wv[EMBED_DIM][EMBED_DIM], 
    hls::stream<uint256_dt> &FIFO_Wv,
    int row_offset,
    int col_offset
);

void load_Weight_V_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wv,
    int BRAM_Wv[TILE_SIZE][TILE_SIZE]
);

void load_Bias_V_HBM2BRAM(
    int bv[EMBED_DIM],
    int BRAM_bv[EMBED_DIM]
);

void compute(
   int BRAM_X[SEQ_LEN][EMBED_DIM],
   hls::stream<uint256_dt> &FIFO_Wq,
   hls::stream<uint256_dt> &FIFO_Wk,
   hls::stream<uint256_dt> &FIFO_Wv,
   int BRAM_bq[EMBED_DIM],
   int BRAM_bk[EMBED_DIM],
   int BRAM_bv[EMBED_DIM],
   hls::stream<uint256_dt> &FIFO_Q_out,
   hls::stream<uint256_dt> &FIFO_K_out,
   hls::stream<uint256_dt> &FIFO_V_out
);

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int BRAM_Wv[TILE_SIZE][TILE_SIZE]
);

// void task_init_acc(
//     int BRAM_bq[EMBED_DIM],
//     int BRAM_bk[EMBED_DIM],
//     int BRAM_bv[EMBED_DIM],
//     int BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
//     int BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
//     int BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
//     int j_tile
// );

void task_compute_tile(
    int BRAM_X[SEQ_LEN][EMBED_DIM],
    int BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int BRAM_Wv[TILE_SIZE][TILE_SIZE],
    int BRAM_bq[EMBED_DIM],
    int BRAM_bk[EMBED_DIM],
    int BRAM_bv[EMBED_DIM],
    int BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int j_tile
);

void task_stream_out(
    int BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out
);

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    int Output_Global[NUM_HEADS][SEQ_LEN][HEAD_DIM] 
);