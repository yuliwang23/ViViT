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
//load bq,bk,bv,X in sequential order
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
load_X_row:
   for (int i = 0; i < SEQ_LEN; i++){
      uint256_dt *X_256 = (uint256_dt*)X;
   load_X_col:
      for (int j = 0; j < EMBED_DIM/8; j++){
      #pragma HLS PIPELINE II=1
         uint256_dt temp_chunk = X_256[i * (EMBED_DIM / 8) + j];
         loop_unpack:
            for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
               int start_bit = k * 32;
               int end_bit = start_bit + 31;
               buffer_X[i][j*8+k] = temp_chunk.range(end_bit, start_bit);
            }
      }
   }
}

void load_Weight_Q(
   int Wq[EMBED_DIM][EMBED_DIM],
   int buffer_Wq[EMBED_DIM][EMBED_DIM])
{
uint256_dt *Wq_256 = (uint256_dt*)Wq;
load_Weight_Q_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_Q_col:
      for (int i = 0; i < EMBED_DIM / 8; i++) {
      #pragma HLS PIPELINE II=1
      uint256_dt temp_chunk = Wq_256[j * (EMBED_DIM / 8) + i];
      loop_unpack:
        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wq[i*8+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

void load_Weight_K(
   int Wk[EMBED_DIM][EMBED_DIM],
   int buffer_Wk[EMBED_DIM][EMBED_DIM])
{
uint256_dt *Wk_256 = (uint256_dt*)Wk;
load_Weight_K_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_K_col:
      for (int i = 0; i < EMBED_DIM / 8; i++) {
      #pragma HLS PIPELINE II=1
      uint256_dt temp_chunk = Wk_256[j * (EMBED_DIM / 8) + i];
      loop_unpack:
        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wk[i*8+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

void load_Weight_V(
   int Wv[EMBED_DIM][EMBED_DIM],
   int buffer_Wv[EMBED_DIM][EMBED_DIM])
{
uint256_dt *Wv_256 = (uint256_dt*)Wv;
load_Weight_V_row:
   for (int j = 0; j < EMBED_DIM; j++) {
   load_Weight_V_col:
      for (int i = 0; i < EMBED_DIM / 8; i++) {
      #pragma HLS PIPELINE II=1
      uint256_dt temp_chunk = Wv_256[j * (EMBED_DIM / 8) + i];
      loop_unpack:
        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            buffer_Wv[i*8+k][j] = temp_chunk.range(end_bit, start_bit);
        }
      }
   }
}

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

void FIFO_gen_X(
   int buffer_X[SEQ_LEN][EMBED_DIM],
   hls::stream<int> stream_Xq[DSP_PER_CYCLE],
   hls::stream<int> stream_Xk[DSP_PER_CYCLE],
   hls::stream<int> stream_Xv[DSP_PER_CYCLE]){
HEAD:
   for (int h = 0; h < NUM_HEADS; h++)
   {
   Token:
      for (int i = 0; i < SEQ_LEN; i++)
      {
         DIM:
         for (int d = 0; d < HEAD_DIM; d++)
         {
            EMBED_PARALLEL:
            for(int e = 0; e < EMBED_DIM; e += DSP_PER_CYCLE) { 
               #pragma HLS PIPELINE II=1
               for (int k = 0; k < DSP_PER_CYCLE; k++) {
                  #pragma HLS UNROLL
                  stream_Xq[k].write(buffer_X[i][e + k]);
                  stream_Xk[k].write(buffer_X[i][e + k]);
                  stream_Xv[k].write(buffer_X[i][e + k]);
               }
            }
         }
      }
   }
}

void FIFO_gen_Weight(
   int buffer_Wq[EMBED_DIM][EMBED_DIM],
   int buffer_Wk[EMBED_DIM][EMBED_DIM],
   int buffer_Wv[EMBED_DIM][EMBED_DIM],
   hls::stream<int> stream_Wq[DSP_PER_CYCLE],
   hls::stream<int> stream_Wk[DSP_PER_CYCLE],
   hls::stream<int> stream_Wv[DSP_PER_CYCLE]){
HEAD:
   for (int h = 0; h < NUM_HEADS; h++)
   { 
   Token:
      for (int i = 0; i < SEQ_LEN; i++)
      {
         DIM:
         for (int d = 0; d < HEAD_DIM; d++)
         {
            int col = h * HEAD_DIM + d;
            EMBED_PARALLEL:
            for(int e = 0; e < EMBED_DIM; e += DSP_PER_CYCLE) { 
               #pragma HLS PIPELINE II=1
               for (int k = 0; k < DSP_PER_CYCLE; k++) {
                  #pragma HLS UNROLL
                  stream_Wq[k].write(buffer_Wq[e + k][col]);
                  stream_Wk[k].write(buffer_Wk[e + k][col]);
                  stream_Wv[k].write(buffer_Wv[e + k][col]);
               }
            }
         }
      }
   }
}

void store_Q(
   int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   hls::stream<int> &stream_Q
    
){
   uint256_dt *Q_256_ptr = (uint256_dt*)Q;
   store_Q_loop_Head:
   for (int h = 0; h < NUM_HEADS; h++) {
      store_Q_loop_Seq:
      for (int i = 0; i < SEQ_LEN; i++) {
         store_Q_loop_Dim:
         for (int d = 0; d < HEAD_DIM/8; d++) {
               uint256_dt temp_chunk;
               loop_pack:
               for(int k = 0; k < 8; k++) {
                  #pragma HLS PIPELINE II=1
                  int val = stream_Q.read();
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  temp_chunk.range(end_bit, start_bit) = val;
               }
               int global_idx = h * (SEQ_LEN * HEAD_DIM / 8) + i * (HEAD_DIM / 8) + d;
               Q_256_ptr[global_idx] = temp_chunk;
         }
      }
   }
}

void store_K(
   int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   hls::stream<int> &stream_K
){
   uint256_dt *K_256_ptr = (uint256_dt*)K;
   store_K_loop_Head:
   for (int h = 0; h < NUM_HEADS; h++) {
      store_K_loop_Seq:
      for (int i = 0; i < SEQ_LEN; i++) {
         store_K_loop_Dim:
         for (int d = 0; d < HEAD_DIM/8; d++) {
               uint256_dt temp_chunk;
               loop_pack:
               for(int k = 0; k < 8; k++) {
                  #pragma HLS PIPELINE II=1
                  int val = stream_K.read();
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  temp_chunk.range(end_bit, start_bit) = val;
               }
               int global_idx = h * (SEQ_LEN * HEAD_DIM / 8) + i * (HEAD_DIM / 8) + d;
               K_256_ptr[global_idx] = temp_chunk;
         }
      }
   }
}

void store_V(
   int V[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   hls::stream<int> &stream_V
){
   uint256_dt *V_256_ptr = (uint256_dt*)V;
   store_V_loop_Head:
   for (int h = 0; h < NUM_HEADS; h++) {
      store_V_loop_Seq:
      for (int i = 0; i < SEQ_LEN; i++) {
         store_V_loop_Dim:
         for (int d = 0; d < HEAD_DIM/8; d++) {
               uint256_dt temp_chunk;
               loop_pack:
               for(int k = 0; k < 8; k++) {
                  #pragma HLS PIPELINE II=1
                  int val = stream_V.read();
                  int start_bit = k * 32;
                  int end_bit = start_bit + 31;
                  temp_chunk.range(end_bit, start_bit) = val;
               }
               int global_idx = h * (SEQ_LEN * HEAD_DIM / 8) + i * (HEAD_DIM / 8) + d;
               V_256_ptr[global_idx] = temp_chunk;
         }
      }
   }
}

void store_data_linear_qkv_multihead(
   hls::stream<int> &FIFO_Q,
   hls::stream<int> &FIFO_K,
   hls::stream<int> &FIFO_V,
   int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
   int V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
#pragma HLS DATAFLOW

store_Q(
   Q,
   FIFO_Q
);
store_K(
   K,
   FIFO_K
);
store_V(
   V,
   FIFO_V
);
}

void compute_linear_qkv_multihead(
   int buffer_bq[EMBED_DIM],
   int buffer_bk[EMBED_DIM],
   int buffer_bv[EMBED_DIM],
   hls::stream<int> stream_Xq[DSP_PER_CYCLE],
   hls::stream<int> stream_Xk[DSP_PER_CYCLE],
   hls::stream<int> stream_Xv[DSP_PER_CYCLE],
   hls::stream<int> stream_Wq[DSP_PER_CYCLE],
   hls::stream<int> stream_Wk[DSP_PER_CYCLE],
   hls::stream<int> stream_Wv[DSP_PER_CYCLE],
   hls::stream<int> &stream_Q,
   hls::stream<int> &stream_K,
   hls::stream<int> &stream_V
   ) 
{
   HEAD:
    for (int h = 0; h < NUM_HEADS; h++)
    {
      Token:
        for (int i = 0; i < SEQ_LEN; i++)
        {
            DIM:
            for (int d = 0; d < HEAD_DIM; d++)
            {
               int q = 0, k = 0, v = 0;
               int col = h * HEAD_DIM + d;
               for (int e = 0; e < EMBED_DIM; e+= DSP_PER_CYCLE)
               {
                  #pragma HLS PIPELINE II=1      
                  for(int w = 0; w < DSP_PER_CYCLE; w++) {
                     #pragma HLS UNROLL
                     q += stream_Xq[w].read() * stream_Wq[w].read();
                     k += stream_Xk[w].read() * stream_Wk[w].read();
                     v += stream_Xv[w].read() * stream_Wv[w].read();
                  }
               }
               stream_Q.write(q + buffer_bq[col]);
               stream_K.write(k + buffer_bk[col]);
               stream_V.write(v + buffer_bv[col]);
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
// static int buffer_Q[NUM_HEADS][SEQ_LEN][HEAD_DIM];
// static int buffer_K[NUM_HEADS][SEQ_LEN][HEAD_DIM];
// static int buffer_V[NUM_HEADS][SEQ_LEN][HEAD_DIM];

// gmem interface 0: X, bq, bk, bv = 197*768+768+768+768=153984
// gmem interface 1: Wq, Q = 768*768+12*197*64=611584
// gmem interface 2: Wk, K = 768*768+12*197*64=611584
// gmem interface 3: Wv, V = 768*768+12*197*64=611584
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wq offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wk offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wv offset=slave bundle=gmem3 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bq offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bk offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bv offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=gmem3 max_widen_bitwidth=256
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
#pragma HLS ARRAY_PARTITION variable=buffer_X dim=2 type=cyclic factor=DSP_PER_CYCLE
#pragma HLS ARRAY_PARTITION variable=buffer_Wq dim=1 type=cyclic factor=DSP_PER_CYCLE
#pragma HLS ARRAY_PARTITION variable=buffer_Wk dim=1 type=cyclic factor=DSP_PER_CYCLE
#pragma HLS ARRAY_PARTITION variable=buffer_Wv dim=1 type=cyclic factor=DSP_PER_CYCLE
// #pragma HLS ARRAY_PARTITION variable=buffer_Q dim=3 type=cyclic factor=DSP_PER_CYCLE
// #pragma HLS ARRAY_PARTITION variable=buffer_K dim=3 type=cyclic factor=DSP_PER_CYCLE
// #pragma HLS ARRAY_PARTITION variable=buffer_V dim=3 type=cyclic factor=DSP_PER_CYCLE
// dataflow
#pragma HLS DATAFLOW
// storage binding
// #pragma HLS BIND_STORAGE variable=buffer_Wq type=ram_2p impl=uram
// #pragma HLS BIND_STORAGE variable=buffer_Wk type=ram_2p impl=uram
// #pragma HLS BIND_STORAGE variable=buffer_Wv type=ram_2p impl=uram
// FIFO buffer
hls::stream<int> FIFO_Xq[DSP_PER_CYCLE];
hls::stream<int> FIFO_Xk[DSP_PER_CYCLE];
hls::stream<int> FIFO_Xv[DSP_PER_CYCLE];
hls::stream<int> FIFO_Wq[DSP_PER_CYCLE];
hls::stream<int> FIFO_Wk[DSP_PER_CYCLE];
hls::stream<int> FIFO_Wv[DSP_PER_CYCLE];
hls::stream<int> FIFO_Q;
hls::stream<int> FIFO_K;
hls::stream<int> FIFO_V;
// FIFO depth
#pragma HLS STREAM variable=FIFO_Xq depth=512
#pragma HLS STREAM variable=FIFO_Xk depth=512
#pragma HLS STREAM variable=FIFO_Xv depth=512
#pragma HLS STREAM variable=FIFO_Wq depth=512
#pragma HLS STREAM variable=FIFO_Wk depth=512
#pragma HLS STREAM variable=FIFO_Wv depth=512
#pragma HLS STREAM variable=FIFO_Q depth=512
#pragma HLS STREAM variable=FIFO_K depth=512
#pragma HLS STREAM variable=FIFO_V depth=512


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
FIFO_gen_X(
   buffer_X, 
   FIFO_Xq, FIFO_Xk, FIFO_Xv
);

FIFO_gen_Weight(
   buffer_Wq, buffer_Wk, buffer_Wv,
   FIFO_Wq, FIFO_Wk, FIFO_Wv
);
compute_linear_qkv_multihead(
   buffer_bq,
   buffer_bk,
   buffer_bv,
   FIFO_Xq,
   FIFO_Xk,
   FIFO_Xv,
   FIFO_Wq,
   FIFO_Wk,
   FIFO_Wv,
   FIFO_Q,
   FIFO_K,
   FIFO_V
);

store_data_linear_qkv_multihead(
   FIFO_Q,
   FIFO_K,
   FIFO_V,
   Q,
   K,
   V
);
}