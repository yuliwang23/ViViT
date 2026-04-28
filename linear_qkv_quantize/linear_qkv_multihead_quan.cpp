#include "linear_qkv_multihead_quan.h"

extern "C" {
void linear_qkv_multihead(
    int8_t X[SEQ_LEN][EMBED_DIM],
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    float bq[EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    float bk[EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    float bv[EMBED_DIM],
    float scale_input[SEQ_LEN], // per-token, from previos LayerNorm kernel
    float scale_wq[EMBED_DIM], // per-output-channel
    float scale_wk[EMBED_DIM],
    float scale_wv[EMBED_DIM],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
// gmem interface 0: X, bq, bk, bv, 
// gmem interface 1: Wq, Q = 768*768+12*197*64=611584
// gmem interface 2: Wk, K = 768*768+12*197*64=611584
// gmem interface 3: Wv, V = 768*768+12*197*64=611584

// suppose data in HBM is int32
#pragma HLS INTERFACE m_axi port=X offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=scale_input offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bq offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bk offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bv offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=scale_wq offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=scale_wk offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=scale_wv offset=slave bundle=gmem0 max_widen_bitwidth=256

#pragma HLS INTERFACE m_axi port=Wq offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wk offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Wv offset=slave bundle=gmem3 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=gmem3 max_widen_bitwidth=256

// control interface
#pragma HLS INTERFACE s_axilite port=X bundle=control
#pragma HLS INTERFACE s_axilite port=Wq bundle=control
#pragma HLS INTERFACE s_axilite port=bq bundle=control
#pragma HLS INTERFACE s_axilite port=Wk bundle=control
#pragma HLS INTERFACE s_axilite port=bk bundle=control
#pragma HLS INTERFACE s_axilite port=Wv bundle=control
#pragma HLS INTERFACE s_axilite port=bv bundle=control
#pragma HLS INTERFACE s_axilite port=scale_input bundle=control
#pragma HLS INTERFACE s_axilite port=scale_wq bundle=control
#pragma HLS INTERFACE s_axilite port=scale_wk bundle=control
#pragma HLS INTERFACE s_axilite port=scale_wv bundle=control
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=V bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

// input buffer declaration
static int8_t BRAM_X[SEQ_LEN][EMBED_DIM];
static float BRAM_scale_input[SEQ_LEN];
static float BRAM_bq[EMBED_DIM];
static float BRAM_bk[EMBED_DIM];
static float BRAM_bv[EMBED_DIM];
static float BRAM_scale_wq[EMBED_DIM];
static float BRAM_scale_wk[EMBED_DIM];
static float BRAM_scale_wv[EMBED_DIM];

// partitioning
#pragma HLS ARRAY_PARTITION variable=BRAM_X dim=2 type=cyclic factor=32
#pragma HLS ARRAY_PARTITION variable=BRAM_scale_input dim=1 type=complete
// storage binding
#pragma HLS BIND_STORAGE variable=BRAM_X type=ram_2p impl=uram

#pragma HLS ARRAY_PARTITION variable=BRAM_bq dim=1 type=cyclic factor=TILE_SIZE
#pragma HLS ARRAY_PARTITION variable=BRAM_bk dim=1 type=cyclic factor=TILE_SIZE
#pragma HLS ARRAY_PARTITION variable=BRAM_bv dim=1 type=cyclic factor=TILE_SIZE
#pragma HLS ARRAY_PARTITION variable=BRAM_scale_wq dim=1 type=cyclic factor=TILE_SIZE
#pragma HLS ARRAY_PARTITION variable=BRAM_scale_wk dim=1 type=cyclic factor=TILE_SIZE
#pragma HLS ARRAY_PARTITION variable=BRAM_scale_wv dim=1 type=cyclic factor=TILE_SIZE
// FIFO buffer
hls::stream<uint256_dt> FIFO_X;
hls::stream<uint256_dt> FIFO_Wq, FIFO_Wk, FIFO_Wv;
hls::stream<uint256_dt> FIFO_Q,  FIFO_K,  FIFO_V;
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
    }
    load_Scale_Input_HBM2BRAM(scale_input, BRAM_scale_input);
    load_Vec_HBM2BRAM(bq, BRAM_bq);
    load_Vec_HBM2BRAM(bk, BRAM_bk);
    load_Vec_HBM2BRAM(bv, BRAM_bv);
    load_Vec_HBM2BRAM(scale_wq, BRAM_scale_wq);
    load_Vec_HBM2BRAM(scale_wk, BRAM_scale_wk);
    load_Vec_HBM2BRAM(scale_wv, BRAM_scale_wv);

    {
        #pragma HLS DATAFLOW
        weight_streamer(Wq, Wk, Wv, FIFO_Wq, FIFO_Wk, FIFO_Wv);

        compute(BRAM_X,
                FIFO_Wq, FIFO_Wk, FIFO_Wv,
                BRAM_bq, BRAM_bk, BRAM_bv,
                BRAM_scale_input,
                BRAM_scale_wq, BRAM_scale_wk, BRAM_scale_wv,
                FIFO_Q, FIFO_K, FIFO_V);

        store_result_FIFO2HBM(FIFO_Q, Q);
        store_result_FIFO2HBM(FIFO_K, K);
        store_result_FIFO2HBM(FIFO_V, V);
    }
}
} 

void weight_streamer(
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv)
{
    for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE) {
        for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE) {
            load_Weight_tile_HBM2FIFO(Wq, FIFO_Wq, j_tile, k_tile);
            load_Weight_tile_HBM2FIFO(Wk, FIFO_Wk, j_tile, k_tile);
            load_Weight_tile_HBM2FIFO(Wv, FIFO_Wv, j_tile, k_tile);
        }
    }
}

void load_X_HBM2FIFO(
    int8_t X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X)
{
    const int TOTAL_CHUNKS = (SEQ_LEN * EMBED_DIM) / 32; // 32 bytes = 256 bits
    uint256_dt *X_ptr = reinterpret_cast<uint256_dt*>(X); // reinterpret_cast 把二維陣列視為一維的 256-bit 陣列
    load_X_loop:
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        FIFO_X.write(X_ptr[i]);
    }
}

void load_X_FIFO2BRAM( // 從 FIFO 接收 256-bit chunks，unpack 成個別的 int8 並存進 BRAM_X
    hls::stream<uint256_dt> &FIFO_X,
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM])
{
    load_row:
    for (int i = 0; i < SEQ_LEN; i++) {
        load_col:
        for (int j = 0; j < EMBED_DIM / 32; j++) { // 每個 chunk 有 32 個 int8
        #pragma HLS PIPELINE II=1
            uint256_dt chunk = FIFO_X.read();
            unpack_x:
            for (int k = 0; k < 32; k++) {
                #pragma HLS UNROLL
                BRAM_X[i][j * 32 + k] = (int8_t)(chunk.range(8 * k + 7, 8 * k).to_uint()); // 每 8 bits 取一個 byte
            }
        }
    }
}

void load_Weight_tile_HBM2FIFO(
    int8_t W[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset, //tile start row
    int col_offset) //tile start col
{
    uint256_dt *W_ptr = reinterpret_cast<uint256_dt*>(W);
    const int row_chunks  = EMBED_DIM / 32;
    const int tile_chunks = TILE_SIZE / 32;
    const int base = row_offset * row_chunks + (col_offset / 32); //tile start index

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        load_tile_col:
        for (int j = 0; j < tile_chunks; j++) {
            #pragma HLS PIPELINE II=1
            FIFO_W.write(W_ptr[base + i * row_chunks + j]);
        }
    }
}

void load_Weight_tile_FIFO2BRAM( // 從 FIFO 接收 weight tile 的 chunks，unpack 並以 BRAM_W[k][j] 的格式存入 BRAM
    hls::stream<uint256_dt> &FIFO_W,
    int8_t BRAM_W[TILE_SIZE][TILE_SIZE])
{
    fifo2bram_jrow:
    for (int j_local = 0; j_local < TILE_SIZE; j_local++) {
        fifo2bram_kchunk:
        for (int kc = 0; kc < TILE_SIZE / 32; kc++) {
            #pragma HLS PIPELINE II=1
            uint256_dt chunk = FIFO_W.read();
            unpack_w:
            for (int r = 0; r < 32; r++) {
                #pragma HLS UNROLL
                BRAM_W[kc * 32 + r][j_local] = (int8_t)(chunk.range(8 * r + 7, 8 * r).to_uint());
            }
        }
    }
}

void load_Scale_Input_HBM2BRAM(float s_hbm[SEQ_LEN], float BRAM_s[SEQ_LEN])
{
    load_scale_input:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1
        BRAM_s[i] = s_hbm[i];
    }
}

void load_Vec_HBM2BRAM(float v_hbm[EMBED_DIM], float BRAM_v[EMBED_DIM])
{
    load_vec:
    for (int j = 0; j < EMBED_DIM; j++) {
        #pragma HLS PIPELINE II=1
        BRAM_v[j] = v_hbm[j];
    }
}

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
    hls::stream<uint256_dt> &FIFO_V_out)
{
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE];
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE];
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_Q dim=2 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_K dim=2 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_V dim=2 type=cyclic factor=8
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE];
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE];
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE];
    #pragma HLS ARRAY_PARTITION variable=BRAM_Wq dim=1 type=cyclic factor=32
    #pragma HLS ARRAY_PARTITION variable=BRAM_Wk dim=1 type=cyclic factor=32
    #pragma HLS ARRAY_PARTITION variable=BRAM_Wv dim=1 type=cyclic factor=32
    #pragma HLS BIND_STORAGE variable=BRAM_Wq type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=BRAM_Wk type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=BRAM_Wv type=ram_2p impl=bram

    loop_out_tile:
    for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE) {
    #pragma HLS DATAFLOW //wrong?
        // INT8 MAC (Multiply and ACcumulate)，結果存進 BRAM_Acc
        task_compute_tile(
            BRAM_X,
            BRAM_Wq, BRAM_Wk, BRAM_Wv,
            BRAM_Acc_Q, BRAM_Acc_K, BRAM_Acc_V,
            FIFO_Wq, FIFO_Wk, FIFO_Wv);

        // dequant + FP32 bias，結果推進 output FIFO
        task_stream_out(
            BRAM_Acc_Q, BRAM_Acc_K, BRAM_Acc_V,
            BRAM_scale_input,
            BRAM_scale_wq, BRAM_scale_wk, BRAM_scale_wv,
            BRAM_bq, BRAM_bk, BRAM_bv,
            j_tile,
            FIFO_Q_out, FIFO_K_out, FIFO_V_out);
    }
}

void load_All_Weights_Parallel( // 把 Wq, Wk, Wv 三個 weight tile 從 FIFO 載入 BRAM
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE],
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE])
{
    #pragma HLS DATAFLOW
    load_Weight_tile_FIFO2BRAM(FIFO_Wq, BRAM_Wq);
    load_Weight_tile_FIFO2BRAM(FIFO_Wk, BRAM_Wk);
    load_Weight_tile_FIFO2BRAM(FIFO_Wv, BRAM_Wv);
}

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
    hls::stream<uint256_dt> &FIFO_Wv)
{
    #pragma HLS INLINE off

    init_acc_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < TILE_SIZE; j += 8) {
            #pragma HLS PIPELINE II=1
            for (int r = 0; r < 8; r++) {
                #pragma HLS UNROLL
                BRAM_Acc_Q[i][j + r] = 0;
                BRAM_Acc_K[i][j + r] = 0;
                BRAM_Acc_V[i][j + r] = 0;
            }
        }
    }

    loop_k_tile:
    for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE) {
        load_All_Weights_Parallel(
            FIFO_Wq, FIFO_Wk, FIFO_Wv,
            BRAM_Wq, BRAM_Wk, BRAM_Wv);

        compute_seq:
        for (int i = 0; i < SEQ_LEN; i++) {
            compute_col:
            for (int j = 0; j < TILE_SIZE; j++) {
                // Partial Sum Initialization
                int32_t p_sum_q = 0;
                int32_t p_sum_k = 0;
                int32_t p_sum_v = 0;
                #pragma HLS PIPELINE II=4

                compute_dot:
                for (int k = 0; k < TILE_SIZE; k += 16) {
                    int16_t reg_X [16];
                    int16_t reg_Wq[16];
                    int16_t reg_Wk[16];
                    int16_t reg_Wv[16];
                    #pragma HLS ARRAY_PARTITION variable=reg_X  type=complete
                    #pragma HLS ARRAY_PARTITION variable=reg_Wq type=complete
                    #pragma HLS ARRAY_PARTITION variable=reg_Wk type=complete
                    #pragma HLS ARRAY_PARTITION variable=reg_Wv type=complete

                    // 因為 int8 的範圍是 -128 ~ +127，兩個 int8 相乘的最壞情況是 (-128) * (-128) = 16384
                    // 但 int8 只能裝到 127，放不下 16384，所以先升到 int16 防止 overflow (int16 範圍是 -32768 ~ +32767)
                    for (int r = 0; r < 16; r++) {
                    #pragma HLS UNROLL
                        reg_X [r] = (int16_t)BRAM_X [i][k_tile + k + r];
                        reg_Wq[r] = (int16_t)BRAM_Wq[k + r][j];
                        reg_Wk[r] = (int16_t)BRAM_Wk[k + r][j];
                        reg_Wv[r] = (int16_t)BRAM_Wv[k + r][j];
                    }
                    for (int r = 0; r < 16; r++) {
                        p_sum_q += (int32_t)reg_X[r] * (int32_t)reg_Wq[r]; // int16 * int16 的最壞情況是 (-32768) * (-32768) = 1,073,741,824，int32 範圍是 +-2,147,483,648，可以裝下
                        p_sum_k += (int32_t)reg_X[r] * (int32_t)reg_Wk[r];
                        p_sum_v += (int32_t)reg_X[r] * (int32_t)reg_Wv[r];
                    }
                }
                BRAM_Acc_Q[i][j] += p_sum_q;
                BRAM_Acc_K[i][j] += p_sum_k;
                BRAM_Acc_V[i][j] += p_sum_v;
            }
        }
    }
}

void task_stream_out( // 把累積結果 (int32) dequantize 成 float32 並加上 bias，打包成 256-bit 推進 output FIFO。
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
    hls::stream<uint256_dt> &FIFO_V_out)
{
    #pragma HLS INLINE off
    stream_out_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        const float si = BRAM_scale_input[i]; // per-token scale
        stream_out_col:
        for (int j = 0; j < TILE_SIZE; j += 8) { // 一次打包 8 個 float (256 bits)
            #pragma HLS PIPELINE II=1

            uint256_dt out_q, out_k, out_v;

            pack_loop:
            for (int r = 0; r < 8; r++) {
                #pragma HLS UNROLL
                const int idx = j_tile + j + r;
                const float cq = si * BRAM_scale_wq[idx]; // scale_input * scale_q
                const float ck = si * BRAM_scale_wk[idx]; // scale_input * scale_k
                const float cv = si * BRAM_scale_wv[idx]; // scale_input * scale_v

                float fq = (float)BRAM_Acc_Q[i][j + r] * cq + BRAM_bq[idx]; // result (float) = acc (int32) * (scale_input*scale_w) + bias
                float fk = (float)BRAM_Acc_K[i][j + r] * ck + BRAM_bk[idx];
                float fv = (float)BRAM_Acc_V[i][j + r] * cv + BRAM_bv[idx];

                union { float f; uint32_t u; } uq, uk, uv;
                uq.f = fq;  uk.f = fk;  uv.f = fv; // 把 float 值存進去
                out_q.range(32 * (r + 1) - 1, 32 * r) = uq.u; // uq.u 現在是同一塊記憶體，以 uint32 的角度讀出來。
                // uint32_t wrong = (uint32_t)fq; 這是數值轉換，例如 3.7 -> 3，會改變 bit pattern
                out_k.range(32 * (r + 1) - 1, 32 * r) = uk.u;
                out_v.range(32 * (r + 1) - 1, 32 * r) = uv.u;
            }
            FIFO_Q_out.write(out_q);
            FIFO_K_out.write(out_k);
            FIFO_V_out.write(out_v);
        }
    }
}

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    float Output_Global[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    const int TOTAL_CHUNKS = (NUM_HEADS * SEQ_LEN * HEAD_DIM) / 8;
    uint256_dt *Out_ptr = reinterpret_cast<uint256_dt*>(Output_Global);

    store_loop:
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        Out_ptr[i] = FIFO_Out.read();
    }
}