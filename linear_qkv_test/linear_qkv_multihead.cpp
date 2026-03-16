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
static int BRAM_X[SEQ_LEN][EMBED_DIM];
static int BRAM_bq[EMBED_DIM];
static int BRAM_bk[EMBED_DIM];
static int BRAM_bv[EMBED_DIM];

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

#pragma HLS ARRAY_PARTITION variable=BRAM_X dim=2 type=cyclic factor=8
#pragma HLS DATAFLOW
#pragma HLS BIND_STORAGE variable=BRAM_X type=ram_2p impl=uram

hls::stream<uint256_dt> FIFO_X;
hls::stream<uint256_dt> FIFO_Wq;
hls::stream<uint256_dt> FIFO_Wk;
hls::stream<uint256_dt> FIFO_Wv;
hls::stream<uint256_dt> FIFO_Q;
hls::stream<uint256_dt> FIFO_K;
hls::stream<uint256_dt> FIFO_V;
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

   // Plan B: col-major HBM read → FIFO order = col 0 all rows, col 1 all rows, ...
   //         FIFO2BRAM outer=col(j), inner=row(i) → BRAM[j][i*8+k] = W[i*8+k][j] = W^T
   //         compute_dot: BRAM[j][k+r] = W^T[j][k+r] = W[k+r][j] → correct X@W
   //
   // HBM2FIFO reads col-major: for each col j in tile, read TILE_SIZE rows one by one.
   // Because HBM is row-major, each element W[row][col_offset+j] needs individual addressing:
   //   ptr_index = (row_offset + i) * global_stride + (col_offset + j) / 8
   //   but we must pack 8 consecutive rows into one 256-bit word ourselves.
   // We pack 8 rows at a time: chunk[k] = W[row_offset + i*8 + k][col_offset + j]
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

// Plan B: col-major HBM read.
// For each col j (0..TILE_SIZE-1) in the tile,
// read TILE_SIZE rows in groups of 8, pack into 256-bit chunks.
// Each chunk packs W[row_offset+i*8+0][col_offset+j] .. W[row_offset+i*8+7][col_offset+j].
// Since HBM is row-major, element W[row][col] is at:
//   32-bit word index = row * EMBED_DIM + col
//   256-bit ptr index = (row * EMBED_DIM + col) / 8
//   bit offset within chunk = (col % 8) * 32
// We read each element individually and pack 8 rows into one 256-bit word.
// row_offset = k_tile, col_offset = j_tile
void load_Weight_Q_tile_HBM2FIFO(
    int Wq[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    int row_offset,  // k_tile
    int col_offset   // j_tile
) {
    int global_stride = EMBED_DIM / 8;  // 96

    // Outer: col j in tile (0..TILE_SIZE-1)
    // Inner: row groups of 8 (i = 0..TILE_SIZE/8-1)
    // Each iteration packs 8 consecutive rows of the same col into one 256-bit chunk
    load_tile_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_tile_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt chunk = 0;
            pack_rows:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                // W[row_offset + i*8 + k][col_offset + j]
                int row = row_offset + i*8 + k;
                int col = col_offset + j;
                // 256-bit ptr index and bit position for this element
                int ptr_idx  = (row * EMBED_DIM + col) / 8;
                int bit_pos  = (col % 8) * 32;
                // Cast to 256-bit ptr to read the chunk containing this element
                uint256_dt *Wq_256_ptr = (uint256_dt*)Wq;
                int val = Wq_256_ptr[ptr_idx].range(bit_pos + 31, bit_pos);
                chunk.range(k*32+31, k*32) = val;
            }
            FIFO_Wq.write(chunk);
        }
    }
}

// FIFO order (from col-major HBM2FIFO):
//   j=0: chunk[0..TILE_SIZE/8-1] → W[row0..rowTILE-1][col0], packed 8 rows per chunk
//   j=1: chunk[TILE_SIZE/8..2*TILE_SIZE/8-1] → W[row0..rowTILE-1][col1]
//   ...
// Outer=col(j), inner=row(i): perfectly matches FIFO order.
// BRAM[j][i*8+k] = W[i*8+k][j] = W^T  ✓
// dim=1 factor=8:  FIFO2BRAM writes to same row j, 8 consecutive cols → dim=2 factor=8
// dim=2 factor=16: compute_dot reads same row j, 16 consecutive cols  → dim=2 factor=16
// Use dim=2 factor=16 to cover both (factor=16 ≥ factor=8).
void load_Weight_Q_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wq,
    int BRAM_Wq[TILE_SIZE][TILE_SIZE]
) {
    load_local_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_local_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wq.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
               #pragma HLS UNROLL
               // BRAM[j][i*8+k] = W[i*8+k][j] = W^T[j][i*8+k]
               BRAM_Wq[j][i*8+k] = temp_chunk.range(k*32+31, k*32);
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
    int global_stride = EMBED_DIM / 8;

    load_tile_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_tile_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt chunk = 0;
            pack_rows:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                int row = row_offset + i*8 + k;
                int col = col_offset + j;
                int ptr_idx = (row * EMBED_DIM + col) / 8;
                int bit_pos = (col % 8) * 32;
                uint256_dt *Wk_256_ptr = (uint256_dt*)Wk;
                int val = Wk_256_ptr[ptr_idx].range(bit_pos + 31, bit_pos);
                chunk.range(k*32+31, k*32) = val;
            }
            FIFO_Wk.write(chunk);
        }
    }
}

void load_Weight_K_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wk,
    int BRAM_Wk[TILE_SIZE][TILE_SIZE]
) {
    load_local_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_local_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wk.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                BRAM_Wk[j][i*8+k] = temp_chunk.range(k*32+31, k*32);
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
    int global_stride = EMBED_DIM / 8;

    load_tile_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_tile_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt chunk = 0;
            pack_rows:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                int row = row_offset + i*8 + k;
                int col = col_offset + j;
                int ptr_idx = (row * EMBED_DIM + col) / 8;
                int bit_pos = (col % 8) * 32;
                uint256_dt *Wv_256_ptr = (uint256_dt*)Wv;
                int val = Wv_256_ptr[ptr_idx].range(bit_pos + 31, bit_pos);
                chunk.range(k*32+31, k*32) = val;
            }
            FIFO_Wv.write(chunk);
        }
    }
}

void load_Weight_V_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Wv,
    int BRAM_Wv[TILE_SIZE][TILE_SIZE]
) {
    load_local_col:
    for (int j = 0; j < TILE_SIZE; j++) {
        load_local_row:
        for (int i = 0; i < TILE_SIZE / 8; i++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_Wv.read();
            unpack_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                BRAM_Wv[j][i*8+k] = temp_chunk.range(k*32+31, k*32);
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
   // BRAM stores W^T: BRAM[j][k] = W[k][j]
   // FIFO2BRAM: outer=col(j), writes 8 elems to BRAM[j][i*8+0..7] → same row j, consecutive cols
   //            → dim=2 factor=8 needed
   // compute_dot: reads 16 elems from BRAM[j][k+0..15]  → same row j, consecutive cols
   //            → dim=2 factor=16 needed
   // Use dim=2 factor=16 (covers both). Add dim=1 factor=8 for structural balance.
   // Fully partitioned (8*16=128 banks) → registers, no BIND_STORAGE.
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=1 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=2 type=cyclic factor=16
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=1 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=2 type=cyclic factor=16
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=1 type=cyclic factor=8
   #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=2 type=cyclic factor=16

   loop_out_tile:
   for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE) {
      #pragma HLS DATAFLOW

      task_compute_tile(
         BRAM_X,
         BRAM_Wq, BRAM_Wk, BRAM_Wv,
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
                  // BRAM[j][k+r] = W^T[j][k+r] = W[k+r][j] → correct for X@W
                  reg_Wq[r] = BRAM_Wq[j][k+r];
                  reg_Wk[r] = BRAM_Wk[j][k+r];
                  reg_Wv[r] = BRAM_Wv[j][k+r];
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