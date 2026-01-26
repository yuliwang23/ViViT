#include "linear_qkv_multihead.h"

// this version is no optimize
void load_X_and_Bias(
   int X[SEQ_LEN][EMBED_DIM],
   int buffer_X[SEQ_LEN][EMBED_DIM],
   int bq[EMBED_DIM],
   int buffer_bq[EMBED_DIM],
   int bk[EMBED_DIM],
   int buffer_bk[EMBED_DIM],
   int bv[EMBED_DIM],
   int buffer_bv[EMBED_DIM])
{

load_X_row:
   for (int i = 0; i < SEQ_LEN; i++){
      uint512_dt *X_512 = (uint512_dt*)X;
   load_X_col:
      for (int j = 0; j < EMBED_DIM/16; j++){
      #pragma HLS PIPELINE II=1
         uint512_dt temp_chunk = X_512[i * (EMBED_DIM / 16) + j];
         loop_unpack:
            for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL 
               int start_bit = k * 32;
               int end_bit = start_bit + 31;
               buffer_X[i][j*16+k] = temp_chunk.range(end_bit, start_bit);
            }
      }
   }

load_Bias_Q_loop:
   for(int i = 0; i < EMBED_DIM; i++) {
   #pragma HLS PIPELINE II=1
      buffer_bq[i] = bq[i];
   }
load_Bias_K_loop:
   for(int i = 0; i < EMBED_DIM; i++) {
   #pragma HLS PIPELINE II=1
      buffer_bk[i] = bk[i];
   }
load_Bias_V_loop:
   for(int i = 0; i < EMBED_DIM; i++) {
   #pragma HLS PIPELINE II=1
      buffer_bv[i] = bv[i];
   }
}

void load_Weight_Q(
   int Wq[EMBED_DIM][EMBED_DIM],
   int buffer_Wq[EMBED_DIM][EMBED_DIM])
{
uint512_dt *Wq_512 = (uint512_dt*)Wq;
load_Weight_Q_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_Q_col:
      for (int i = 0; i < EMBED_DIM / 16; i++) {
      #pragma HLS PIPELINE II=1
      uint512_dt temp_chunk = Wq_512[j * (EMBED_DIM / 16) + i];
      loop_unpack:
        for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wq[i*16+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

void load_Weight_K(
   int Wk[EMBED_DIM][EMBED_DIM],
   int buffer_Wk[EMBED_DIM][EMBED_DIM])
{
uint512_dt *Wk_512 = (uint512_dt*)Wk;
load_Weight_K_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_K_col:
      for (int i = 0; i < EMBED_DIM / 16; i++) {
      #pragma HLS PIPELINE II=1
      uint512_dt temp_chunk = Wk_512[j * (EMBED_DIM / 16) + i];
      loop_unpack:
        for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wk[i*16+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

void load_Weight_V(
   int Wv[EMBED_DIM][EMBED_DIM],
   int buffer_Wv[EMBED_DIM][EMBED_DIM])
{
uint512_dt *Wv_512 = (uint512_dt*)Wv;
load_Weight_V_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_V_col:
      for (int i = 0; i < EMBED_DIM / 16; i++) {
      #pragma HLS PIPELINE II=1
      uint512_dt temp_chunk = Wv_512[j * (EMBED_DIM / 16) + i];
      loop_unpack:
        for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wv[i*16+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

// void load_Weight_K(
//    int Wk[EMBED_DIM][EMBED_DIM],
//    int buffer_Wk[EMBED_DIM][EMBED_DIM])
// {
// load_Weight_K_row:
//    for (int i = 0; i < EMBED_DIM; i++) {
//    load_Weight_K_col:
//       for (int j = 0; j < EMBED_DIM; j++) {
//       #pragma HLS PIPELINE II=1
//          buffer_Wk[i][j] = Wk[i][j];
//       }
//    }
// }

// void load_Weight_V(
//    int Wv[EMBED_DIM][EMBED_DIM],
//    int buffer_Wv[EMBED_DIM][EMBED_DIM])
// {
// load_Weight_V_row:
//    for (int i = 0; i < EMBED_DIM; i++) {
//    load_Weight_V_col:
//       for (int j = 0; j < EMBED_DIM; j++) {
//       #pragma HLS PIPELINE II=1
//          buffer_Wv[i][j] = Wv[i][j];
//       }
//    }
// }

void load_data_linear_qkv_multihead(
   int X[SEQ_LEN][EMBED_DIM],
   int Wq[EMBED_DIM][EMBED_DIM],
   int bq[EMBED_DIM],
   int Wk[EMBED_DIM][EMBED_DIM],
   int bk[EMBED_DIM],
   int Wv[EMBED_DIM][EMBED_DIM],
   int bv[EMBED_DIM],
   int buffer_X[SEQ_LEN][EMBED_DIM],
   int buffer_Wq[EMBED_DIM][EMBED_DIM],
   int buffer_Wk[EMBED_DIM][EMBED_DIM],
   int buffer_Wv[EMBED_DIM][EMBED_DIM],
   int buffer_bq[EMBED_DIM],
   int buffer_bk[EMBED_DIM],
   int buffer_bv[EMBED_DIM])
{
#pragma HLS DATAFLOW

load_X_and_Bias(
   X,
   buffer_X,
   bq,
   buffer_bq,
   bk,
   buffer_bk,
   bv,
   buffer_bv
);

load_Weight_Q(
   Wq,
   buffer_Wq
);
load_Weight_K(
   Wk,
   buffer_Wk
);
load_Weight_V(
   Wv,
   buffer_Wv
);
}

load_X_row:
   for (int i = 0; i < SEQ_LEN; i++){
      uint512_dt *X_512 = (uint512_dt*)X;
   load_X_col:
      for (int j = 0; j < EMBED_DIM/16; j++){
      #pragma HLS PIPELINE II=1
         uint512_dt temp_chunk = X_512[i * (EMBED_DIM / 16) + j];
         loop_unpack:
            for(int k = 0; k < 16; k++) {
            #pragma HLS UNROLL 
               int start_bit = k * 32;
               int end_bit = start_bit + 31;
               buffer_X[i][j*16+k] = temp_chunk.range(end_bit, start_bit);
            }
      }
   }


void store_Q(
   int buffer_Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM]
){
uint512_dt *Q_512 = (uint512_dt*)Q;
store_Q_loop:
   for (int h = 0; h < NUM_HEADS; h++) {
      for (int i = 0; i < SEQ_LEN; i++) {
         for (int d = 0; d < HEAD_DIM/16; d++) {
         #pragma HLS PIPELINE II=1
            uint512_dt temp_chunk = Q_512[h * (SEQ_LEN * HEAD_DIM / 16) + i * (HEAD_DIM / 16) + d];
            loop_pack:
               for(int k = 0; k < 16; k++) {
               #pragma HLS UNROLL
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  Q[h][i][d*16+k] = temp_chunk.range(end_bit, start_bit);
               }
         }
      }
   }
}

void store_K(
   int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int K[NUM_HEADS][SEQ_LEN][HEAD_DIM]
){
uint512_dt *K_512 = (uint512_dt*)K;
store_K_loop:
   for (int h = 0; h < NUM_HEADS; h++) {
      for (int i = 0; i < SEQ_LEN; i++) {
         for (int d = 0; d < HEAD_DIM/16; d++) {
         #pragma HLS PIPELINE II=1
            uint512_dt temp_chunk = K_512[h * (SEQ_LEN * HEAD_DIM / 16) + i * (HEAD_DIM / 16) + d];
            loop_pack:
               for(int k = 0; k < 16; k++) {
               #pragma HLS UNROLL
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  K[h][i][d*16+k] = temp_chunk.range(end_bit, start_bit);
               }
         }
      }
   }
}

void store_V(
   int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int V[NUM_HEADS][SEQ_LEN][HEAD_DIM]
){
uint512_dt *V_512 = (uint512_dt*)V;
store_V_loop:
   for (int h = 0; h < NUM_HEADS; h++) {
      for (int i = 0; i < SEQ_LEN; i++) {
         for (int d = 0; d < HEAD_DIM/16; d++) {
         #pragma HLS PIPELINE II=1
            uint512_dt temp_chunk = V_512[h * (SEQ_LEN * HEAD_DIM / 16) + i * (HEAD_DIM / 16) + d];
            loop_pack:
               for(int k = 0; k < 16; k++) {
               #pragma HLS UNROLL
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  V[h][i][d*16+k] = temp_chunk.range(end_bit, start_bit);
               }
         }
      }
   }
}

// void store_K(
//    int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
//    int K[NUM_HEADS][SEQ_LEN][HEAD_DIM]
// ){
// store_K_loop:
//    for (int h = 0; h < NUM_HEADS; h++) {
//       for (int i = 0; i < SEQ_LEN; i++) {
//          for (int d = 0; d < HEAD_DIM; d++) {
//             #pragma HLS PIPELINE II=1
//             K[h][i][d] = buffer_K[h][i][d];
//          }
//       }
//    }
// }

// void store_V(
//    int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM],
//    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM]
// ){
// store_V_loop:
//    for (int h = 0; h < NUM_HEADS; h++) {
//       for (int i = 0; i < SEQ_LEN; i++) {
//          for (int d = 0; d < HEAD_DIM; d++) {
//             #pragma HLS PIPELINE II=1
//             V[h][i][d] = buffer_V[h][i][d];
//          }
//       }
//    }
// }

void store_data_linear_qkv_multihead(
   int buffer_Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
#pragma HLS DATAFLOW

store_Q(
   buffer_Q,
   Q
);
store_K(
   buffer_K,
   K
);
store_V(
   buffer_V,
   V
);
}

void compute_linear_qkv_multihead(
   int buffer_X[SEQ_LEN][EMBED_DIM],
   int buffer_Wq[EMBED_DIM][EMBED_DIM],
   int buffer_bq[EMBED_DIM],
   int buffer_Wk[EMBED_DIM][EMBED_DIM],
   int buffer_bk[EMBED_DIM],
   int buffer_Wv[EMBED_DIM][EMBED_DIM],
   int buffer_bv[EMBED_DIM],
   int buffer_Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
   Token:
    for (int i = 0; i < SEQ_LEN; i++)
    {
      HEAD:
        for (int h = 0; h < NUM_HEADS; h++)
        {
            DIM:
            #pragma HLS PIPELINE II=1
            for (int d = 0; d < HEAD_DIM; d++)
            {
               int q = 0, k = 0, v = 0;
               int col = h * HEAD_DIM + d;
               EMBED:
               for (int e = 0; e < EMBED_DIM; e++)
               {
                  #pragma HLS UNROLL factor=64
                  q += buffer_X[i][e] * buffer_Wq[e][col];
                  k += buffer_X[i][e] * buffer_Wk[e][col];
                  v += buffer_X[i][e] * buffer_Wv[e][col];
               }
               buffer_Q[h][i][d] = q + buffer_bq[col];
               buffer_K[h][i][d] = k + buffer_bk[col];
               buffer_V[h][i][d] = v + buffer_bv[col];
            }
        }
    }
}

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
static int buffer_X[SEQ_LEN][EMBED_DIM];
static int buffer_Wq[EMBED_DIM][EMBED_DIM];
static int buffer_Wk[EMBED_DIM][EMBED_DIM];
static int buffer_Wv[EMBED_DIM][EMBED_DIM];
static int buffer_bq[EMBED_DIM];
static int buffer_bk[EMBED_DIM];
static int buffer_bv[EMBED_DIM];
// output buffer declaration
static int buffer_Q[NUM_HEADS][SEQ_LEN][HEAD_DIM];
static int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM];
static int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM];

// gmem interface 0: X, bq, bk, bv = 197*768+768+768+768=153984
// gmem interface 1: Wq, Q = 768*768+12*197*64=611584
// gmem interface 2: Wk, K = 768*768+12*197*64=611584
// gmem interface 3: Wv, V = 768*768+12*197*64=611584
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=Wq offset=slave bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=Wk offset=slave bundle=gmem2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=Wv offset=slave bundle=gmem3 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=bq offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=bk offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=bv offset=slave bundle=gmem0 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem1 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=gmem2 max_widen_bitwidth=512
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=gmem3 max_widen_bitwidth=512
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
#pragma HLS ARRAY_PARTITION variable=buffer_X dim=2 type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=buffer_Wq dim=1 type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=buffer_Wk dim=1 type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=buffer_Wv dim=1 type=cyclic factor=64
#pragma HLS ARRAY_PARTITION variable=buffer_Q dim=3 type=cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=buffer_K dim=3 type=cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=buffer_V dim=3 type=cyclic factor=16
// dataflow
#pragma HLS DATAFLOW
// storage binding
#pragma HLS BIND_STORAGE variable=buffer_Wq type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buffer_Wk type=ram_2p impl=uram
#pragma HLS BIND_STORAGE variable=buffer_Wv type=ram_2p impl=uram

load_data_linear_qkv_multihead(
   X, Wq, bq,
   Wk, bk,
   Wv, bv,
   buffer_X,
   buffer_Wq,
   buffer_Wk,
   buffer_Wv,
   buffer_bq,
   buffer_bk,
   buffer_bv
);
compute_linear_qkv_multihead(
   buffer_X,
   buffer_Wq,
   buffer_bq,
   buffer_Wk,
   buffer_bk,
   buffer_Wv,
   buffer_bv,
   buffer_Q,
   buffer_K,
   buffer_V
);
store_data_linear_qkv_multihead(
   buffer_Q,
   buffer_K,
   buffer_V,
   Q,
   K,
   V
);
}
