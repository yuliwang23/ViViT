#include "linear_qkv_multihead_quan.h"

// ============================================================
// FIFO depth constants
//   Weight FIFO : one j_tile ?? all k_tiles
//     = (EMBED_DIM/TILE_SIZE) ?? TILE_SIZE ?? (TILE_SIZE/32)
//     = 12 ?? 64 ?? 2 = 1536 chunks per matrix ??? ??3 shared stream ??? 1536
//   Output FIFO : one j_tile ?? SEQ_LEN rows
//     = SEQ_LEN ?? (TILE_SIZE/8) = 197 ?? 8 = 1576
// ============================================================
#define FIFO_W_DEPTH 1536
#define FIFO_OUT_DEPTH 1576

// ============================================================
// Top-level kernel
// ============================================================
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
        float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
    {
#pragma HLS INTERFACE m_axi port = X offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = scale_input offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = bq offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = bk offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = bv offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = scale_wq offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = scale_wk offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = scale_wv offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = Wq offset = slave bundle = gmem1 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = Q offset = slave bundle = gmem1 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = Wk offset = slave bundle = gmem2 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = K offset = slave bundle = gmem2 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = Wv offset = slave bundle = gmem3 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = V offset = slave bundle = gmem3 max_widen_bitwidth = 256

#pragma HLS INTERFACE s_axilite port = X bundle = control
#pragma HLS INTERFACE s_axilite port = Wq bundle = control
#pragma HLS INTERFACE s_axilite port = bq bundle = control
#pragma HLS INTERFACE s_axilite port = Wk bundle = control
#pragma HLS INTERFACE s_axilite port = bk bundle = control
#pragma HLS INTERFACE s_axilite port = Wv bundle = control
#pragma HLS INTERFACE s_axilite port = bv bundle = control
#pragma HLS INTERFACE s_axilite port = scale_input bundle = control
#pragma HLS INTERFACE s_axilite port = scale_wq bundle = control
#pragma HLS INTERFACE s_axilite port = scale_wk bundle = control
#pragma HLS INTERFACE s_axilite port = scale_wv bundle = control
#pragma HLS INTERFACE s_axilite port = Q bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = V bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        // ----------------------------------------------------------
        // On-chip buffers
        // ----------------------------------------------------------
        static int8_t BRAM_X[SEQ_LEN][EMBED_DIM];
        static float BRAM_scale_input[SEQ_LEN];
        static float BRAM_bq[EMBED_DIM];
        static float BRAM_bk[EMBED_DIM];
        static float BRAM_bv[EMBED_DIM];
        static float BRAM_scale_wq[EMBED_DIM];
        static float BRAM_scale_wk[EMBED_DIM];
        static float BRAM_scale_wv[EMBED_DIM];

        // BRAM_X: partition dim=2 by 32 ??? 32 banks, each 197??(768/32)=24 words
        // Use URAM to save BRAM budget
#pragma HLS ARRAY_PARTITION variable = BRAM_X dim = 2 type = cyclic factor = 32
#pragma HLS BIND_STORAGE variable = BRAM_X type = ram_2p impl = uram

        // scale/bias vectors: fully partitioned along TILE_SIZE chunks
        // so task_stream_out can access 8 elements per cycle without MUX
#pragma HLS ARRAY_PARTITION variable = BRAM_scale_input dim = 1 type = complete
#pragma HLS ARRAY_PARTITION variable = BRAM_bq dim = 1 type = cyclic factor = TILE_SIZE
#pragma HLS ARRAY_PARTITION variable = BRAM_bk dim = 1 type = cyclic factor = TILE_SIZE
#pragma HLS ARRAY_PARTITION variable = BRAM_bv dim = 1 type = cyclic factor = TILE_SIZE
#pragma HLS ARRAY_PARTITION variable = BRAM_scale_wq dim = 1 type = cyclic factor = TILE_SIZE
#pragma HLS ARRAY_PARTITION variable = BRAM_scale_wk dim = 1 type = cyclic factor = TILE_SIZE
#pragma HLS ARRAY_PARTITION variable = BRAM_scale_wv dim = 1 type = cyclic factor = TILE_SIZE

        // ----------------------------------------------------------
        // FIFOs between the three pipeline stages
        // Depth must cover one full j_tile worth of data so the
        // stages can run independently without deadlock.
        // ----------------------------------------------------------
        hls::stream<uint256_dt> FIFO_X;
        hls::stream<uint256_dt> FIFO_Wq, FIFO_Wk, FIFO_Wv;
        hls::stream<uint256_dt> FIFO_Q, FIFO_K, FIFO_V;

#pragma HLS STREAM variable = FIFO_X depth = 256
#pragma HLS STREAM variable = FIFO_Wq depth = FIFO_W_DEPTH
#pragma HLS STREAM variable = FIFO_Wk depth = FIFO_W_DEPTH
#pragma HLS STREAM variable = FIFO_Wv depth = FIFO_W_DEPTH
#pragma HLS STREAM variable = FIFO_Q depth = FIFO_OUT_DEPTH
#pragma HLS STREAM variable = FIFO_K depth = FIFO_OUT_DEPTH
#pragma HLS STREAM variable = FIFO_V depth = FIFO_OUT_DEPTH

        // ----------------------------------------------------------
        // Phase 1: Load X (two-stage dataflow: HBM???FIFO???BRAM)
        // ----------------------------------------------------------
        {
#pragma HLS DATAFLOW
            load_X_HBM2FIFO(X, FIFO_X);
            load_X_FIFO2BRAM(FIFO_X, BRAM_X);
        }

        // ----------------------------------------------------------
        // Phase 2: Load scalar constants (sequential, small volume)
        // ----------------------------------------------------------
        load_Scale_Input_HBM2BRAM(scale_input, BRAM_scale_input);
        load_Vec_HBM2BRAM(bq, BRAM_bq);
        load_Vec_HBM2BRAM(bk, BRAM_bk);
        load_Vec_HBM2BRAM(bv, BRAM_bv);
        load_Vec_HBM2BRAM(scale_wq, BRAM_scale_wq);
        load_Vec_HBM2BRAM(scale_wk, BRAM_scale_wk);
        load_Vec_HBM2BRAM(scale_wv, BRAM_scale_wv);

        // ----------------------------------------------------------
        // Phase 3: Main compute ??? ONE top-level DATAFLOW
        //
        //   stage_weight_stream  ??????FIFO_Wq/k/v?????????  stage_compute
        //                                               ???
        //                                          FIFO_Q/K/V
        //                                               ???
        //                                        stage_store ????????? HBM
        //
        // Each stage owns the full j_tile loop internally.
        // No loop-level DATAFLOW anywhere.
        // ----------------------------------------------------------
        {
#pragma HLS DATAFLOW
            stage_weight_stream(Wq, Wk, Wv, FIFO_Wq, FIFO_Wk, FIFO_Wv);
            stage_compute(BRAM_X,
                          BRAM_bq, BRAM_bk, BRAM_bv,
                          BRAM_scale_input,
                          BRAM_scale_wq, BRAM_scale_wk, BRAM_scale_wv,
                          FIFO_Wq, FIFO_Wk, FIFO_Wv,
                          FIFO_Q, FIFO_K, FIFO_V);
            stage_store(FIFO_Q, FIFO_K, FIFO_V, Q, K, V);
        }
    }
} // extern "C"

// ============================================================
// stage_weight_stream
//   Streams ALL j_tile ?? k_tile weight chunks into FIFOs.
//   Loop order: j_tile (outer) ??? k_tile (inner) ??? row ??? col
//   Total per FIFO = 12 j_tiles ?? 12 k_tiles ?? 64 rows ?? 2 chunks
//                  = 12 ?? 1536 = 18432 chunks
//   FIFO depth = 1536 (one j_tile), so stage_compute must drain
//   each j_tile before this function moves to the next ??? which is
//   guaranteed by the consumer loop ordering.
// ============================================================
void stage_weight_stream(
    int8_t Wq[EMBED_DIM][EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv)
{
#pragma HLS INLINE off
    for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE)
    {
        for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE)
        {
            load_Weight_tile_HBM2FIFO(Wq, FIFO_Wq, j_tile, k_tile);
            load_Weight_tile_HBM2FIFO(Wk, FIFO_Wk, j_tile, k_tile);
            load_Weight_tile_HBM2FIFO(Wv, FIFO_Wv, j_tile, k_tile);
        }
    }
}

// ============================================================
// stage_compute
//   For each j_tile: load weights ??? MAC ??? dequant ??? output FIFO
//   NO inner DATAFLOW. load_All_Weights is now sequential.
// ============================================================
void stage_compute(
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM],
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    float BRAM_scale_input[SEQ_LEN],
    float BRAM_scale_wq[EMBED_DIM],
    float BRAM_scale_wk[EMBED_DIM],
    float BRAM_scale_wv[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out)
{
#pragma HLS INLINE off

    // Weight tile buffers (local, synthesized as BRAM)
    int8_t BRAM_Wq[TILE_SIZE][TILE_SIZE];
    int8_t BRAM_Wk[TILE_SIZE][TILE_SIZE];
    int8_t BRAM_Wv[TILE_SIZE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = BRAM_Wq dim = 1 type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = BRAM_Wk dim = 1 type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = BRAM_Wv dim = 1 type = cyclic factor = 32
#pragma HLS BIND_STORAGE variable = BRAM_Wq type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = BRAM_Wk type = ram_2p impl = bram
#pragma HLS BIND_STORAGE variable = BRAM_Wv type = ram_2p impl = bram

    // Accumulator buffers (one j_tile column block, reused across j_tiles)
    int32_t BRAM_Acc_Q[SEQ_LEN][TILE_SIZE];
    int32_t BRAM_Acc_K[SEQ_LEN][TILE_SIZE];
    int32_t BRAM_Acc_V[SEQ_LEN][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = BRAM_Acc_Q dim = 2 type = cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = BRAM_Acc_K dim = 2 type = cyclic factor = 8
#pragma HLS ARRAY_PARTITION variable = BRAM_Acc_V dim = 2 type = cyclic factor = 8

// ---- j_tile loop: no DATAFLOW here ----
loop_j_tile:
    for (int j_tile = 0; j_tile < EMBED_DIM; j_tile += TILE_SIZE)
    {

    // ---- Zero accumulators ----
    init_acc:
        for (int i = 0; i < SEQ_LEN; i++)
        {
            for (int j = 0; j < TILE_SIZE; j += 8)
            {
#pragma HLS PIPELINE II = 1
                for (int r = 0; r < 8; r++)
                {
#pragma HLS UNROLL
                    BRAM_Acc_Q[i][j + r] = 0;
                    BRAM_Acc_K[i][j + r] = 0;
                    BRAM_Acc_V[i][j + r] = 0;
                }
            }
        }

    // ---- k_tile reduction: load weight tile then MAC ----
    loop_k_tile:
        for (int k_tile = 0; k_tile < EMBED_DIM; k_tile += TILE_SIZE)
        {

            // FIX: Serial load (no DATAFLOW inside loop)
            load_Weight_tile_FIFO2BRAM(FIFO_Wq, BRAM_Wq);
            load_Weight_tile_FIFO2BRAM(FIFO_Wk, BRAM_Wk);
            load_Weight_tile_FIFO2BRAM(FIFO_Wv, BRAM_Wv);

        // INT8 MAC
        compute_seq:
            for (int i = 0; i < SEQ_LEN; i++)
            {
            compute_col:
                for (int j = 0; j < TILE_SIZE; j++)
                {
                    int32_t p_sum_q = 0;
                    int32_t p_sum_k = 0;
                    int32_t p_sum_v = 0;
#pragma HLS PIPELINE II = 4

                compute_dot:
                    for (int k = 0; k < TILE_SIZE; k += 16)
                    {
                        int16_t reg_X[16];
                        int16_t reg_Wq[16];
                        int16_t reg_Wk[16];
                        int16_t reg_Wv[16];
#pragma HLS ARRAY_PARTITION variable = reg_X type = complete
#pragma HLS ARRAY_PARTITION variable = reg_Wq type = complete
#pragma HLS ARRAY_PARTITION variable = reg_Wk type = complete
#pragma HLS ARRAY_PARTITION variable = reg_Wv type = complete

                        for (int r = 0; r < 16; r++)
                        {
#pragma HLS UNROLL
                            reg_X[r] = (int16_t)BRAM_X[i][k_tile + k + r];
                            reg_Wq[r] = (int16_t)BRAM_Wq[k + r][j];
                            reg_Wk[r] = (int16_t)BRAM_Wk[k + r][j];
                            reg_Wv[r] = (int16_t)BRAM_Wv[k + r][j];
                        }
                        for (int r = 0; r < 16; r++)
                        {
#pragma HLS UNROLL
                            p_sum_q += (int32_t)reg_X[r] * (int32_t)reg_Wq[r];
                            p_sum_k += (int32_t)reg_X[r] * (int32_t)reg_Wk[r];
                            p_sum_v += (int32_t)reg_X[r] * (int32_t)reg_Wv[r];
                        }
                    }
                    BRAM_Acc_Q[i][j] += p_sum_q;
                    BRAM_Acc_K[i][j] += p_sum_k;
                    BRAM_Acc_V[i][j] += p_sum_v;
                }
            }
        } // end k_tile

        // ---- FIX: Copy tile's scale/bias to local arrays ----
        // Eliminates dynamic index (j_tile + j + r) into EMBED_DIM-wide arrays,
        // which caused large MUX trees and prevented II=1 in stream_out.
        float local_scale_wq[TILE_SIZE];
        float local_scale_wk[TILE_SIZE];
        float local_scale_wv[TILE_SIZE];
        float local_bq[TILE_SIZE];
        float local_bk[TILE_SIZE];
        float local_bv[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable = local_scale_wq type = complete
#pragma HLS ARRAY_PARTITION variable = local_scale_wk type = complete
#pragma HLS ARRAY_PARTITION variable = local_scale_wv type = complete
#pragma HLS ARRAY_PARTITION variable = local_bq type = complete
#pragma HLS ARRAY_PARTITION variable = local_bk type = complete
#pragma HLS ARRAY_PARTITION variable = local_bv type = complete

    copy_scale:
        for (int r = 0; r < TILE_SIZE; r++)
        {
#pragma HLS PIPELINE II = 1
            local_scale_wq[r] = BRAM_scale_wq[j_tile + r];
            local_scale_wk[r] = BRAM_scale_wk[j_tile + r];
            local_scale_wv[r] = BRAM_scale_wv[j_tile + r];
            local_bq[r] = BRAM_bq[j_tile + r];
            local_bk[r] = BRAM_bk[j_tile + r];
            local_bv[r] = BRAM_bv[j_tile + r];
        }

    // ---- Dequant + pack ??? output FIFO ----
    stream_out_seq:
        for (int i = 0; i < SEQ_LEN; i++)
        {
            const float si = BRAM_scale_input[i];

        stream_out_col:
            for (int j = 0; j < TILE_SIZE; j += 8)
            {
#pragma HLS PIPELINE II = 1
                uint256_dt out_q, out_k, out_v;

            pack_loop:
                for (int r = 0; r < 8; r++)
                {
#pragma HLS UNROLL
                    // FIX: use local array with static offset [j + r]
                    const float cq = si * local_scale_wq[j + r];
                    const float ck = si * local_scale_wk[j + r];
                    const float cv = si * local_scale_wv[j + r];

                    float fq = (float)BRAM_Acc_Q[i][j + r] * cq + local_bq[j + r];
                    float fk = (float)BRAM_Acc_K[i][j + r] * ck + local_bk[j + r];
                    float fv = (float)BRAM_Acc_V[i][j + r] * cv + local_bv[j + r];

                    union
                    {
                        float f;
                        uint32_t u;
                    } uq, uk, uv;
                    uq.f = fq;
                    uk.f = fk;
                    uv.f = fv;
                    out_q.range(32 * (r + 1) - 1, 32 * r) = uq.u;
                    out_k.range(32 * (r + 1) - 1, 32 * r) = uk.u;
                    out_v.range(32 * (r + 1) - 1, 32 * r) = uv.u;
                }
                FIFO_Q_out.write(out_q);
                FIFO_K_out.write(out_k);
                FIFO_V_out.write(out_v);
            }
        }
    } // end j_tile
}

// ============================================================
// stage_store
//   Reads all output chunks from FIFOs and writes to HBM.
//   Flat layout: [NUM_HEADS][SEQ_LEN][HEAD_DIM]
//   One j_tile = one head (HEAD_DIM == TILE_SIZE == 64).
// ============================================================
void stage_store(
    hls::stream<uint256_dt> &FIFO_Q,
    hls::stream<uint256_dt> &FIFO_K,
    hls::stream<uint256_dt> &FIFO_V,
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
#pragma HLS INLINE off
    // Total chunks = NUM_HEADS ?? SEQ_LEN ?? (HEAD_DIM/8)
    //              = 12 ?? 197 ?? 8 = 18912
    const int TOTAL = NUM_HEADS * SEQ_LEN * (HEAD_DIM / 8);

    uint256_dt *Q_ptr = reinterpret_cast<uint256_dt *>(Q);
    uint256_dt *K_ptr = reinterpret_cast<uint256_dt *>(K);
    uint256_dt *V_ptr = reinterpret_cast<uint256_dt *>(V);

store_loop:
    for (int i = 0; i < TOTAL; i++)
    {
#pragma HLS PIPELINE II = 1
        Q_ptr[i] = FIFO_Q.read();
        K_ptr[i] = FIFO_K.read();
        V_ptr[i] = FIFO_V.read();
    }
}

// ============================================================
// load_X_HBM2FIFO
// ============================================================
void load_X_HBM2FIFO(
    int8_t X[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X)
{
    const int TOTAL_CHUNKS = (SEQ_LEN * EMBED_DIM) / 32;
    uint256_dt *X_ptr = reinterpret_cast<uint256_dt *>(X);
load_X_loop:
    for (int i = 0; i < TOTAL_CHUNKS; i++)
    {
#pragma HLS PIPELINE II = 1
        FIFO_X.write(X_ptr[i]);
    }
}

// ============================================================
// load_X_FIFO2BRAM
// ============================================================
void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    int8_t BRAM_X[SEQ_LEN][EMBED_DIM])
{
load_row:
    for (int i = 0; i < SEQ_LEN; i++)
    {
    load_col:
        for (int j = 0; j < EMBED_DIM / 32; j++)
        {
#pragma HLS PIPELINE II = 1
            uint256_dt chunk = FIFO_X.read();
        unpack_x:
            for (int k = 0; k < 32; k++)
            {
#pragma HLS UNROLL
                BRAM_X[i][j * 32 + k] =
                    (int8_t)(chunk.range(8 * k + 7, 8 * k).to_uint());
            }
        }
    }
}

// ============================================================
// load_Weight_tile_HBM2FIFO
// ============================================================
void load_Weight_tile_HBM2FIFO(
    int8_t W[EMBED_DIM][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset,
    int col_offset)
{
    uint256_dt *W_ptr = reinterpret_cast<uint256_dt *>(W);
    const int row_chunks = EMBED_DIM / 32;  // 24
    const int tile_chunks = TILE_SIZE / 32; // 2
    const int base = row_offset * row_chunks + (col_offset / 32);

load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++)
    {
    load_tile_col:
        for (int j = 0; j < tile_chunks; j++)
        {
#pragma HLS PIPELINE II = 1
            FIFO_W.write(W_ptr[base + i * row_chunks + j]);
        }
    }
}

// ============================================================
// load_Weight_tile_FIFO2BRAM
//   Reads one TILE_SIZE??TILE_SIZE block from FIFO into BRAM.
//   Stores in transposed layout: BRAM_W[k][j] so that the
//   MAC loop can access a full column (all k) with dim=1 partition.
// ============================================================
void load_Weight_tile_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_W,
    int8_t BRAM_W[TILE_SIZE][TILE_SIZE])
{
fifo2bram_row:
    for (int j_local = 0; j_local < TILE_SIZE; j_local++)
    {
    fifo2bram_col:
        for (int kc = 0; kc < TILE_SIZE / 32; kc++)
        {
#pragma HLS PIPELINE II = 1
            uint256_dt chunk = FIFO_W.read();
        unpack_w:
            for (int r = 0; r < 32; r++)
            {
#pragma HLS UNROLL
                BRAM_W[kc * 32 + r][j_local] =
                    (int8_t)(chunk.range(8 * r + 7, 8 * r).to_uint());
            }
        }
    }
}

// ============================================================
// load_Scale_Input_HBM2BRAM
// ============================================================
void load_Scale_Input_HBM2BRAM(float s_hbm[SEQ_LEN], float BRAM_s[SEQ_LEN])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
#pragma HLS PIPELINE II = 1
        BRAM_s[i] = s_hbm[i];
    }
}

// ============================================================
// load_Vec_HBM2BRAM
// ============================================================
void load_Vec_HBM2BRAM(float v_hbm[EMBED_DIM], float BRAM_v[EMBED_DIM])
{
    for (int j = 0; j < EMBED_DIM; j++)
    {
#pragma HLS PIPELINE II = 1
        BRAM_v[j] = v_hbm[j];
    }
}