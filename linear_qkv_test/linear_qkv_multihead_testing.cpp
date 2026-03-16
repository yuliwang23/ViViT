#include "linear_qkv_multihead.h"

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
    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
// input buffer declaration
static int BRAM_X[SEQ_LEN][EMBED_DIM];
// static int BRAM_Wq[tile_size_dim][tile_size_dim];
// static int BRAM_Wk[tile_size_dim][tile_size_dim];
// static int BRAM_Wv[tile_size_dim][tile_size_dim];
static int BRAM_bq[EMBED_DIM];
static int BRAM_bk[EMBED_DIM];
static int BRAM_bv[EMBED_DIM];
// output buffer declaration
// static int BRAM_Q[SEQ_LEN][HEAD_DIM];
// static int BRAM_K[SEQ_LEN][HEAD_DIM];
// static int BRAM_V[SEQ_LEN][HEAD_DIM];

// gmem interface 0: X, bq, bk, bv = 197*768+768+768+768=153984
// gmem interface 1: Wq, Q = 768*768+12*197*64=611584
// gmem interface 2: Wk, K = 768*768+12*197*64=611584
// gmem interface 3: Wv, V = 768*768+12*197*64=611584
// suppose data in HBM is int32
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wq offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wk offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wv offset=slave bundle=gmem3 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bq offset=slave bundle=gmem4 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bk offset=slave bundle=gmem5 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bv offset=slave bundle=gmem6 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem4 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=gmem5 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=gmem6 max_widen_bitwidth=256
// control interface
#pragma HLS INTERFACE s_axilite port=X bundle=control
#pragma HLS INTERFACE s_axilite port=Wq bundle=control
#pragma HLS INTERFACE s_axilite port=Wk bundle=control
#pragma HLS INTERFACE s_axilite port=Wv bundle=control
#pragma HLS INTERFACE s_axilite port=bq bundle=control
#pragma HLS INTERFACE s_axilite port=bk bundle=control
#pragma HLS INTERFACE s_axilite port=bv bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=V bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
// partitioning
#pragma HLS ARRAY_PARTITION variable=BRAM_X dim=2 type=cyclic factor=8
// #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=1 type=cyclic factor=8
// #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=1 type=cyclic factor=8
// #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=1 type=cyclic factor=8
// #pragma HLS ARRAY_PARTITION variable=buffer_Q dim=3 type=cyclic factor=DSP_PER_CYCLE
// #pragma HLS ARRAY_PARTITION variable=buffer_K dim=3 type=cyclic factor=DSP_PER_CYCLE
// #pragma HLS ARRAY_PARTITION variable=buffer_V dim=3 type=cyclic factor=DSP_PER_CYCLE
// dataflow
#pragma HLS DATAFLOW
// storage binding
#pragma HLS BIND_STORAGE variable=BRAM_X type=ram_2p impl=uram
// #pragma HLS BIND_STORAGE variable=BRAM_Wq type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=BRAM_Wk type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=BRAM_Wv type=ram_2p impl=bram
// #pragma HLS BIND_STORAGE variable=BRAM_Wq type=ram_2p impl=uram
// #pragma HLS BIND_STORAGE variable=BRAM_Wk type=ram_2p impl=uram
// #pragma HLS BIND_STORAGE variable=BRAM_Wv type=ram_2p impl=uram
// FIFO buffer
hls::stream<uint256_dt> FIFO_X;
hls::stream<uint256_dt> FIFO_Wq;
hls::stream<uint256_dt> FIFO_Wk;
hls::stream<uint256_dt> FIFO_Wv;
hls::stream<uint256_dt> FIFO_Q;
hls::stream<uint256_dt> FIFO_K;
hls::stream<uint256_dt> FIFO_V;
// FIFO depth
#pragma HLS STREAM variable=FIFO_X depth=256
#pragma HLS STREAM variable=FIFO_Wq depth=256
#pragma HLS STREAM variable=FIFO_Wk depth=256
#pragma HLS STREAM variable=FIFO_Wv depth=256
#pragma HLS STREAM variable=FIFO_Q depth=256
#pragma HLS STREAM variable=FIFO_K depth=256
#pragma HLS STREAM variable=FIFO_V depth=256

   {
      #pragma HLS DATAFLOW
      load_X_HBM2FIFO(X, FIFO_X);
      load_X_FIFO2BRAM(FIFO_X, BRAM_X);
      load_Bias_Q_HBM2BRAM(bq, BRAM_bq);
      load_Bias_K_HBM2BRAM(bk, BRAM_bk);
      load_Bias_V_HBM2BRAM(bv, BRAM_bv);
   }

   #pragma HLS DATAFLOW

   {
      for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE) {
         for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE) {
               load_Weight_Q_tile_HBM2FIFO(Wq, FIFO_Wq, k_tile, j_tile);
               load_Weight_K_tile_HBM2FIFO(Wk, FIFO_Wk, k_tile, j_tile);
               load_Weight_V_tile_HBM2FIFO(Wv, FIFO_Wv, k_tile, j_tile);
         }
      }
   }
   compute(BRAM_X, FIFO_Wq, FIFO_Wk, FIFO_Wv, BRAM_bq, BRAM_bk, BRAM_bv, FIFO_Q, FIFO_K, FIFO_V);
   store_result_FIFO2HBM(FIFO_Q, Q);
   store_result_FIFO2HBM(FIFO_K, K);
   store_result_FIFO2HBM(FIFO_V, V);
}
}

void load_X_HBM2FIFO(
    int X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X)
{
   uint256_dt *X_256_ptr = (uint256_dt*)X;
   load_X_loop:
   for (int i = 0; i < (SEQ_LEN * EMBED_DIM) / 8; i++) {
   #pragma HLS PIPELINE II=1
      FIFO_X.write(X_256_ptr[i]);
   }
}

void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    int BRAM_X[SEQ_LEN][EMBED_DIM])
{
   load_X_row:
   for (int i = 0; i < SEQ_LEN; i++){
      load_X_col:
      for (int j = 0; j < EMBED_DIM/8; j++){
      #pragma HLS PIPELINE II=1
         int reg_buffer[8];
         #pragma HLS ARRAY_PARTITION variable=reg_buffer type=complete
         uint256_dt temp_chunk = FIFO_X.read();
         unpack_loop:
         for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            reg_buffer[k] = temp_chunk.range(k*32+31, k*32);
         }
         write_bram_loop:
         for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            BRAM_X[i][j*8+k] = reg_buffer[k];
         }
      }
   }
}

// row_offset = k_tile (weight row start)
// col_offset = j_tile (weight col start)
// Row-major addressing: Wq[row_offset+i][col_offset] in 256-bit ptr space
void load_Weight_Q_tile_HBM2FIFO(
    int Wq[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    int row_offset,
    int col_offset
) {
    uint256_dt *Wq_256_ptr = (uint256_dt*)Wq;
    int global_stride = EMBED_DIM / 8;  // 96

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start = (row_offset + i) * global_stride + (col_offset / 8);
        load_tile_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            FIFO_Wq.write(Wq_256_ptr[current_row_start + j]);
        }
    }
}

// Outer=row(i), inner=col(j): matches HBM2FIFO row-major write order.
// BRAM_Wq[i][j*8+k] — same row i, 8 consecutive cols per iteration.
// Requires dim=2 cyclic factor=8 on BRAM_Wq so the 8 writes in one
// PIPELINE cycle each go to a different bank → II=1 with no port conflict.
void load_Weight_Q_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wq,
    int BRAM_Wq[TILE_SIZE][TILE_SIZE]
) {
    load_local_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        load_local_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wq.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
               #pragma HLS UNROLL
               BRAM_Wq[i][j*8+k] = temp_chunk.range(k*32+31, k*32);
            }
        }
    }
}

void load_Bias_Q_HBM2BRAM(int bq[EMBED_DIM], int BRAM_bq[EMBED_DIM])
{
    load_bq_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
    #pragma HLS PIPELINE II=1
        BRAM_bq[i] = bq[i];
    }
}

void load_Weight_K_tile_HBM2FIFO(
    int Wk[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wk,
    int row_offset,
    int col_offset
) {
    uint256_dt *Wk_256_ptr = (uint256_dt*)Wk;
    int global_stride = EMBED_DIM / 8;

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start = (row_offset + i) * global_stride + (col_offset / 8);
        load_tile_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            FIFO_Wk.write(Wk_256_ptr[current_row_start + j]);
        }
    }
}

void load_Weight_K_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wk,
    int BRAM_Wk[TILE_SIZE][TILE_SIZE]
) {
    load_local_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        load_local_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wk.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                BRAM_Wk[i][j*8+k] = temp_chunk.range(k*32+31, k*32);
            }
        }
    }
}

void load_Bias_K_HBM2BRAM(int bk[EMBED_DIM], int BRAM_bk[EMBED_DIM])
{
    load_bk_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
    #pragma HLS PIPELINE II=1
        BRAM_bk[i] = bk[i];
    }
}

void load_Weight_V_tile_HBM2FIFO(
    int Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wv,
    int row_offset,
    int col_offset
) {
    uint256_dt *Wv_256_ptr = (uint256_dt*)Wv;
    int global_stride = EMBED_DIM / 8;

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start = (row_offset + i) * global_stride + (col_offset / 8);
        load_tile_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            FIFO_Wv.write(Wv_256_ptr[current_row_start + j]);
        }
    }
}

void load_Weight_V_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wv,
    int BRAM_Wv[TILE_SIZE][TILE_SIZE]
) {
    load_local_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        load_local_col:
        for (int j = 0; j < TILE_SIZE / 8; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wv.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                BRAM_Wv[i][j*8+k] = temp_chunk.range(k*32+31, k*32);
            }
        }
    }
}

void load_Bias_V_HBM2BRAM(int bv[EMBED_DIM], int BRAM_bv[EMBED_DIM])
{
    load_bv_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
    #pragma HLS PIPELINE II=1
        BRAM_bv[i] = bv[i];
    }
}

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
) {
   int BRAM_Acc_Q[SEQ_LEN][TILE_SIZE];
   int BRAM_Acc_K[SEQ_LEN][TILE_SIZE];
   int BRAM_Acc_V[SEQ_LEN][TILE_SIZE];
   #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_Q dim=2 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_K dim=2 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_V dim=2 type=cyclic factor=8

   int BRAM_Wq[TILE_SIZE][TILE_SIZE];
   int BRAM_Wk[TILE_SIZE][TILE_SIZE];
   int BRAM_Wv[TILE_SIZE][TILE_SIZE];
   // dim=2 factor=8  → FIFO2BRAM writes 8 cols of same row → 8 different banks → II=1
   // dim=1 factor=16 → compute_dot reads 16 rows of same col → 16 different banks → no conflict
   // Combined (16*8=128 banks for 64*64): HLS maps to registers, so BIND_STORAGE removed.
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=1 type=cyclic factor=16
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=2 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=1 type=cyclic factor=16
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=2 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=1 type=cyclic factor=16
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=2 type=cyclic factor=8

   loop_out_tile:
   for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE) {
      #pragma HLS DATAFLOW

      task_compute_tile(
         BRAM_X, BRAM_Wq, BRAM_Wk, BRAM_Wv,
         BRAM_bq, BRAM_bk, BRAM_bv,
         BRAM_Acc_Q, BRAM_Acc_K, BRAM_Acc_V,
         FIFO_Wq, FIFO_Wk, FIFO_Wv,
         j_tile
      );

      task_stream_out(
         BRAM_Acc_Q, BRAM_Acc_K, BRAM_Acc_V,
         FIFO_Q_out, FIFO_K_out, FIFO_V_out
      );
   }
}

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int BRAM_Wv[TILE_SIZE][TILE_SIZE]
) {
    #pragma HLS DATAFLOW
    load_Weight_Q_tile_FIFO2BRAM(FIFO_Wq, BRAM_Wq);
    load_Weight_K_tile_FIFO2BRAM(FIFO_Wk, BRAM_Wk);
    load_Weight_V_tile_FIFO2BRAM(FIFO_Wv, BRAM_Wv);
}

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
){
#pragma HLS INLINE off

   init_acc_loop:
   for (int i = 0; i < SEQ_LEN; i++) {
      for (int j = 0; j < TILE_SIZE; j += 8) {
         #pragma HLS PIPELINE II=1
         for (int r = 0; r < 8; r++) {
            #pragma HLS UNROLL
            BRAM_Acc_Q[i][j+r] = BRAM_bq[j_tile+j+r];
            BRAM_Acc_K[i][j+r] = BRAM_bk[j_tile+j+r];
            BRAM_Acc_V[i][j+r] = BRAM_bv[j_tile+j+r];
         }
      }
   }

   loop_in_tile:
   for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE) {

      load_All_Weights_Parallel(
            FIFO_Wq, FIFO_Wk, FIFO_Wv,
            BRAM_Wq, BRAM_Wk, BRAM_Wv
      );

      // BRAM_Wq[k+r][j]: 16 different rows, same col j.
      // dim=1 cyclic factor=16 maps these to 16 different banks → parallel reads OK.
      compute_seq:
      for (int i = 0; i < SEQ_LEN; i++) {
         compute_col:
         for (int j = 0; j < TILE_SIZE; j++) {
            int p_sum_q = 0;
            int p_sum_k = 0;
            int p_sum_v = 0;
            #pragma HLS PIPELINE II=4
            compute_dot:
            for (int k = 0; k < TILE_SIZE; k += 16) {
               int reg_X[16];
               int reg_Wq[16];
               int reg_Wk[16];
               int reg_Wv[16];
               #pragma HLS ARRAY_PARTITION variable=reg_X  type=complete
               #pragma HLS ARRAY_PARTITION variable=reg_Wq type=complete
               #pragma HLS ARRAY_PARTITION variable=reg_Wk type=complete
               #pragma HLS ARRAY_PARTITION variable=reg_Wv type=complete

               for (int r = 0; r < 16; r++) {
                  #pragma HLS UNROLL factor=16
                  reg_X[r]  = BRAM_X[i][k_tile+k+r];
                  reg_Wq[r] = BRAM_Wq[k+r][j];
                  reg_Wk[r] = BRAM_Wk[k+r][j];
                  reg_Wv[r] = BRAM_Wv[k+r][j];
               }

               for (int r = 0; r < 16; r++) {
                  #pragma HLS UNROLL factor=16
                  p_sum_q += reg_X[r] * reg_Wq[r];
                  p_sum_k += reg_X[r] * reg_Wk[r];
                  p_sum_v += reg_X[r] * reg_Wv[r];
               }
            }
            BRAM_Acc_Q[i][j] += p_sum_q;
            BRAM_Acc_K[i][j] += p_sum_k;
            BRAM_Acc_V[i][j] += p_sum_v;
         }
      }
   }
}

void task_stream_out(
    int BRAM_Acc_Q[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_K[SEQ_LEN][TILE_SIZE],
    int BRAM_Acc_V[SEQ_LEN][TILE_SIZE],
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out
){
#pragma HLS INLINE off
   stream_out_seq:
   for (int i = 0; i < SEQ_LEN; i++) {
      stream_out_col:
      for (int j = 0; j < TILE_SIZE; j += 8) {
         #pragma HLS PIPELINE II=1
         uint256_dt out_q, out_k, out_v;
         for (int r = 0; r < 8; r++) {
            #pragma HLS UNROLL factor=8
            out_q.range(32*(r+1)-1, 32*r) = BRAM_Acc_Q[i][j+r];
            out_k.range(32*(r+1)-1, 32*r) = BRAM_Acc_K[i][j+r];
            out_v.range(32*(r+1)-1, 32*r) = BRAM_Acc_V[i][j+r];
         }
         FIFO_Q_out.write(out_q);
         FIFO_K_out.write(out_k);
         FIFO_V_out.write(out_v);
      }
   }
}

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    int Output_Global[NUM_HEADS][SEQ_LEN][HEAD_DIM]
) {
    uint256_dt *Global_256_ptr = (uint256_dt*)Output_Global;
    int total_chunks = (NUM_HEADS * SEQ_LEN * HEAD_DIM) / 8;

    store_loop:
    for (int i = 0; i < total_chunks; i++) {
        #pragma HLS PIPELINE II=1
        Global_256_ptr[i] = FIFO_Out.read();
    }
}