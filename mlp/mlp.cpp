#include "mlp.h"
#include "gelu_lut.h"

#ifndef __SYNTHESIS__
#define HLS_CSIM_STATIC static
#else
#define HLS_CSIM_STATIC
#endif

static void dense0_dataflow_tile(
    uint256_dt *W1,
    int64_dt    X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    float       scaler_X [SEQ_LEN_PADDED],
    float       SRAM_b1  [HIDDEN_DIM],
    float       SRAM_sw1 [HIDDEN_DIM],
    half_dt     H_full_fp[SEQ_LEN_PADDED][HIDDEN_DIM],
    float       scaler_H [SEQ_LEN_PADDED],
    int         iter);

static void dense1_dataflow_tile(
    uint256_dt *W2,
    uint256_dt *output,
    int64_dt    H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float       scaler_H [SEQ_LEN_PADDED],
    float       SRAM_b2  [EMBED_DIM],
    float       SRAM_sw2 [EMBED_DIM],
    int         iter,
    int         token_offset,
    int         valid_seq_len);

extern "C" {
void mlp_q(
    uint256_dt *input,
    float      *input_scaler,
    uint256_dt *W1,
    uint256_dt *W2,
    float      *b1,
    float      *b2,
    float      *scale_w1,
    float      *scale_w2,
    uint256_dt *output,
    int         token_offset,
    int         valid_seq_len)
{
#pragma HLS BIND_STORAGE variable=GELU_LUT type=rom_1p impl=lutram

#pragma HLS INTERFACE m_axi port=input    offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=4728
#pragma HLS INTERFACE m_axi port=input_scaler offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=197
#pragma HLS INTERFACE m_axi port=output   offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=18912
#pragma HLS INTERFACE m_axi port=W1       offset=slave bundle=gmem1 max_widen_bitwidth=256 depth=73728
#pragma HLS INTERFACE m_axi port=W2       offset=slave bundle=gmem2 max_widen_bitwidth=256 depth=73728
#pragma HLS INTERFACE m_axi port=b1       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=3072
#pragma HLS INTERFACE m_axi port=b2       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=768
#pragma HLS INTERFACE m_axi port=scale_w1 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=3072
#pragma HLS INTERFACE m_axi port=scale_w2 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=768

#pragma HLS INTERFACE s_axilite port=input    bundle=control
#pragma HLS INTERFACE s_axilite port=input_scaler bundle=control
#pragma HLS INTERFACE s_axilite port=output   bundle=control
#pragma HLS INTERFACE s_axilite port=W1       bundle=control
#pragma HLS INTERFACE s_axilite port=W2       bundle=control
#pragma HLS INTERFACE s_axilite port=b1       bundle=control
#pragma HLS INTERFACE s_axilite port=b2       bundle=control
#pragma HLS INTERFACE s_axilite port=scale_w1 bundle=control
#pragma HLS INTERFACE s_axilite port=scale_w2 bundle=control
#pragma HLS INTERFACE s_axilite port=token_offset bundle=control
#pragma HLS INTERFACE s_axilite port=valid_seq_len bundle=control
#pragma HLS INTERFACE s_axilite port=return   bundle=control

    HLS_CSIM_STATIC int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package];
    HLS_CSIM_STATIC float    scaler_X [SEQ_LEN_PADDED];
#pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=X_package type=ram_2p impl=bram

    HLS_CSIM_STATIC int64_dt H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package];
    HLS_CSIM_STATIC float    scaler_H [SEQ_LEN_PADDED];
#pragma HLS ARRAY_PARTITION variable=H_package type=cyclic factor=4 dim=1
#pragma HLS BIND_STORAGE    variable=H_package type=ram_2p impl=uram

    HLS_CSIM_STATIC half_dt H_full_fp[SEQ_LEN_PADDED][HIDDEN_DIM];
#pragma HLS BIND_STORAGE    variable=H_full_fp type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=H_full_fp dim=2 type=cyclic factor=8

    HLS_CSIM_STATIC float SRAM_b1 [HIDDEN_DIM];
    HLS_CSIM_STATIC float SRAM_b2 [EMBED_DIM];
    HLS_CSIM_STATIC float SRAM_sw1[HIDDEN_DIM];
    HLS_CSIM_STATIC float SRAM_sw2[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=SRAM_b1  dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_b2  dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_sw1 dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_sw2 dim=1 type=cyclic factor=8

    load_Bias_HBM2BRAM(b1,       SRAM_b1,  HIDDEN_DIM);
    load_Bias_HBM2BRAM(b2,       SRAM_b2,  EMBED_DIM );
    load_Bias_HBM2BRAM(scale_w1, SRAM_sw1, HIDDEN_DIM);
    load_Bias_HBM2BRAM(scale_w2, SRAM_sw2, EMBED_DIM );

    const int CHUNKS_PER_ROW = EMBED_DIM / 32;   // = 24

    scaler_load_loop:
    for (int i = 0; i < SEQ_LEN_PADDED; i++) {
#pragma HLS PIPELINE II=1
        int  source_token = token_offset + i;
        bool in_range     = (i < SEQ_LEN);
        bool valid_token  = in_range && (source_token < valid_seq_len);

        if (valid_token) {
            scaler_X[i] = input_scaler[source_token];
        } else {
            scaler_X[i] = 1.0f;
        }
    }

    seq_load_loop:
    for (int idx = 0; idx < SEQ_LEN_PADDED * CHUNKS_PER_ROW; idx++) {
#pragma HLS PIPELINE II=1
        int  i            = idx / CHUNKS_PER_ROW;
        int  j            = idx % CHUNKS_PER_ROW;
        int  source_token = token_offset + i;
        bool in_range     = (i < SEQ_LEN);                   // not a pad row
        bool valid_token  = in_range && (source_token < valid_seq_len);

        uint256_dt chunk;
        if (valid_token) {
            chunk = input[source_token * CHUNKS_PER_ROW + j];
        } else {
            chunk = (uint256_dt)0;
        }
        for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
            int s = k * 64, e = s + 63;
            X_package[i][j * 4 + k] = chunk.range(e, s);
        }
    }

    // Stage 2:  Dense_0 + GELU + (running-max + stash)
    dense0_tile_loop:
    for (int iter = 0; iter < HIDDEN_DIM / TILE_SIZE; iter++) {
        dense0_dataflow_tile(
            W1,
            X_package, scaler_X,
            SRAM_b1, SRAM_sw1,
            H_full_fp,
            scaler_H,
            iter);
    }

    // Stage 2b:  per-row quantize hidden activation.
    //   Iterates SEQ_LEN_PADDED rows; rows >= SEQ_LEN are zeros so
    //   max_val will hit EPSILON floor and the resulting H_package
    //   rows will be all zero (consistent with valid_token masking
    //   in store_Output_FIFO2HBM).
    quant_hidden_loop:
    for (int i = 0; i < SEQ_LEN_PADDED; i++) {
#pragma HLS LOOP_FLATTEN off
        float max_val = scaler_H[i];
        if (max_val < EPSILON) max_val = EPSILON;
        float scale   = 127.5f / max_val;
        float inv_sc  = max_val / 127.5f;

        pack_hidden:
        for (int j = 0; j < HIDDEN_DIM; j += 8) {
#pragma HLS PIPELINE II=1
            int64_dt pkg = 0;
            for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                float    f  = (float)H_full_fp[i][j + k] * scale;
                int      qi = (int)((f >= 0.0f) ? (f + 0.5f) : (f - 0.5f));
                if (qi >  127) qi =  127;
                if (qi < -128) qi = -128;
                int8_dt qv = (int8_dt)qi;
                int s = k * 8;
                pkg.range(s + 7, s) = qv;
            }
            H_package[i][j >> 3] = pkg;
        }
        scaler_H[i] = inv_sc;
    }

    // Stage 3:  Dense_1 + dequant + bias
    dense1_tile_loop:
    for (int iter = 0; iter < EMBED_DIM / TILE_SIZE; iter++) {
        dense1_dataflow_tile(
            W2, output,
            H_package, scaler_H,
            SRAM_b2, SRAM_sw2,
            iter,
            token_offset,
            valid_seq_len);
    }
}
} // extern "C"

static void dense0_dataflow_tile(
    uint256_dt *W1,
    int64_dt    X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    float       scaler_X [SEQ_LEN_PADDED],
    float       SRAM_b1  [HIDDEN_DIM],
    float       SRAM_sw1 [HIDDEN_DIM],
    half_dt     H_full_fp[SEQ_LEN_PADDED][HIDDEN_DIM],
    float       scaler_H [SEQ_LEN_PADDED],
    int         iter)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<uint256_dt> FIFO_W1("FIFO_W1");
#pragma HLS STREAM variable=FIFO_W1 depth=64

    load_Weight_tile_HBM2FIFO(W1, FIFO_W1, iter * TILE_SIZE, EMBED_DIM);

    dense0_one_tile(
        X_package, scaler_X,
        FIFO_W1,
        SRAM_b1, SRAM_sw1,
        H_full_fp,
        scaler_H,
        iter);
}

static void dense1_dataflow_tile(
    uint256_dt *W2,
    uint256_dt *output,
    int64_dt    H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float       scaler_H [SEQ_LEN_PADDED],
    float       SRAM_b2  [EMBED_DIM],
    float       SRAM_sw2 [EMBED_DIM],
    int         iter,
    int         token_offset,
    int         valid_seq_len)
{
#pragma HLS INLINE off
#pragma HLS DATAFLOW
    hls::stream<uint256_dt> FIFO_W2 ("FIFO_W2");
    hls::stream<uint256_dt> FIFO_OUT("FIFO_OUT");
#pragma HLS STREAM variable=FIFO_W2  depth=32
#pragma HLS STREAM variable=FIFO_OUT depth=32

    load_Weight_tile_HBM2FIFO(W2, FIFO_W2, iter * TILE_SIZE, HIDDEN_DIM);

    dense1_one_tile(
        H_package, scaler_H,
        FIFO_W2,
        SRAM_b2, SRAM_sw2,
        FIFO_OUT,
        iter);

    store_Output_FIFO2HBM(FIFO_OUT, output, iter, token_offset, valid_seq_len);
}

void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X)
{
    load_X_loop:
    for (int i = 0; i < (SEQ_LEN * EMBED_DIM) / 32; i++) {
#pragma HLS PIPELINE II=1
        FIFO_X.write(HBM_X[i]);
    }
}

void load_Bias_HBM2BRAM(
    float *b,
    float  BRAM_b[],
    int N)
{
    load_b_loop:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
        BRAM_b[i] = b[i];
    }
}

void load_Weight_tile_HBM2FIFO(
    uint256_dt *W,
    hls::stream<uint256_dt> &FIFO_W,
    int row_offset,
    int K_dim)
{
    int chunks_per_row = K_dim / 32;
    int base_addr_idx  = row_offset * chunks_per_row;
    int total_chunks   = TILE_SIZE * chunks_per_row;

    load_W_flat:
    for (int k = 0; k < total_chunks; k++) {
#pragma HLS PIPELINE II=1
        FIFO_W.write(W[base_addr_idx + k]);
    }
}

void quantize_row_dynamic(
    float    row_fp[],
    int64_dt row_pkg[],
    int       length,
    float    *out_scaler)
{
#pragma HLS INLINE off

    float pmax[8] = {0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=pmax complete

    find_max:
    for (int j = 0; j < length; j += 8) {
#pragma HLS PIPELINE II=2
        float a[8];
#pragma HLS ARRAY_PARTITION variable=a complete
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            float v = row_fp[j + k];
            a[k] = (v < 0.0f) ? -v : v;
        }
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            if (a[k] > pmax[k]) pmax[k] = a[k];
        }
    }

    float m01   = (pmax[0] > pmax[1]) ? pmax[0] : pmax[1];
    float m23   = (pmax[2] > pmax[3]) ? pmax[2] : pmax[3];
    float m45   = (pmax[4] > pmax[5]) ? pmax[4] : pmax[5];
    float m67   = (pmax[6] > pmax[7]) ? pmax[6] : pmax[7];
    float m0123 = (m01 > m23) ? m01 : m23;
    float m4567 = (m45 > m67) ? m45 : m67;
    float max_val = (m0123 > m4567) ? m0123 : m4567;

    if (max_val < EPSILON) max_val = EPSILON;
    float scale = 127.5f / max_val;

    pack:
    for (int j = 0; j < length; j += 8) {
#pragma HLS PIPELINE II=1
        int64_dt pkg = 0;
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            float f = row_fp[j + k] * scale;
            int qi  = (int)((f >= 0.0f) ? (f + 0.5f) : (f - 0.5f));
            if (qi >  127) qi =  127;
            if (qi < -128) qi = -128;
            pkg.range(k*8+7, k*8) = (int8_dt)qi;
        }
        row_pkg[j >> 3] = pkg;
    }
    *out_scaler = 1.0f / scale;
}

void dense0_one_tile(
    int64_dt  X_package [SEQ_LEN_PADDED][EMBED_DIM_package],
    float     scaler_X  [SEQ_LEN_PADDED],
    hls::stream<uint256_dt> &FIFO_W1,
    float     SRAM_b1   [HIDDEN_DIM],
    float     SRAM_sw1  [HIDDEN_DIM],
    half_dt   H_full_fp [SEQ_LEN_PADDED][HIDDEN_DIM],
    float     scaler_H  [SEQ_LEN_PADDED],
    int       tile_idx)
{
#pragma HLS INLINE off
    HLS_CSIM_STATIC int64_dt URAM_W[TILE_SIZE][EMBED_DIM_package];
#pragma HLS ARRAY_PARTITION variable=URAM_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=URAM_W cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=URAM_W type=ram_2p impl=bram

    int total_chunks = TILE_SIZE * (EMBED_DIM / 32);
    load_W_flat:
    for (int k = 0; k < total_chunks; k++) {
#pragma HLS PIPELINE II=1
        uint256_dt chunk = FIFO_W1.read();
        int i  = k / (EMBED_DIM / 32);
        int jb = (k % (EMBED_DIM / 32)) * 4;
        for (int m = 0; m < 4; m++) {
#pragma HLS UNROLL
            URAM_W[i][jb + m] = chunk.range(m*64+63, m*64);
        }
    }

    compute_seq:
    for (int i_base = 0; i_base < SEQ_LEN_PADDED; i_base += TOKEN_TILE) {
#pragma HLS LOOP_FLATTEN off

        int26_dt acc_reg[TOKEN_TILE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc_reg complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_reg complete dim=2

        init_acc:
        for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                acc_reg[it][t] = 0;
            }
        }

        compute_dot:
        for (int k = 0; k < EMBED_DIM_package; k++) {
#pragma HLS PIPELINE II=1

            int64_dt xp_tile[TOKEN_TILE];
#pragma HLS ARRAY_PARTITION variable=xp_tile complete
            for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
                xp_tile[it] = X_package[i_base + it][k];
            }

            mac_outputs:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                int64_dt wp = URAM_W[t][k];

                for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
                    int64_dt xp = xp_tile[it];
                    int26_dt s  = acc_reg[it][t];
                    for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                        int sb = r * 8, eb = sb + 7;
                        int8_dt xv = (int8_dt)xp.range(eb, sb);
                        int8_dt wv = (int8_dt)wp.range(eb, sb);
                        s += xv * wv;
                    }
                    acc_reg[it][t] = s;
                }
            }
        }

        post_token_loop:
        for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS LOOP_FLATTEN off
            int   i      = i_base + it;
            float s_row  = scaler_X[i];
            float h_tile[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=h_tile complete

            gelu_loop:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                int   col            = tile_idx * TILE_SIZE + t;
                float combined_scale = s_row * SRAM_sw1[col];
                float dequant        = (float)acc_reg[it][t] * combined_scale;
                float biased         = dequant + SRAM_b1[col];
                float activated      = gelu_lut_lookup(biased);
                H_full_fp[i][col]    = activated;
                h_tile[t]            = activated;
            }

            float pmax[8] = {0,0,0,0,0,0,0,0};
#pragma HLS ARRAY_PARTITION variable=pmax complete

            max_lanes:
            for (int t = 0; t < TILE_SIZE; t += 8) {
#pragma HLS PIPELINE II=2
                float a[8];
#pragma HLS ARRAY_PARTITION variable=a complete
                for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                    float v = h_tile[t + k];
                    a[k] = (v < 0.0f) ? -v : v;
                }
                for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                    if (a[k] > pmax[k]) pmax[k] = a[k];
                }
            }
            float m01   = (pmax[0] > pmax[1]) ? pmax[0] : pmax[1];
            float m23   = (pmax[2] > pmax[3]) ? pmax[2] : pmax[3];
            float m45   = (pmax[4] > pmax[5]) ? pmax[4] : pmax[5];
            float m67   = (pmax[6] > pmax[7]) ? pmax[6] : pmax[7];
            float m0123 = (m01 > m23) ? m01 : m23;
            float m4567 = (m45 > m67) ? m45 : m67;
            float tile_max = (m0123 > m4567) ? m0123 : m4567;

            float prev_max = (tile_idx == 0) ? 0.0f : scaler_H[i];
            scaler_H[i] = (tile_max > prev_max) ? tile_max : prev_max;
        }
    }
}

void dense1_one_tile(
    int64_dt  H_package [SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float     scaler_H  [SEQ_LEN_PADDED],
    hls::stream<uint256_dt> &FIFO_W2,
    float     SRAM_b2   [EMBED_DIM],
    float     SRAM_sw2  [EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_OUT,
    int       tile_idx)
{
#pragma HLS INLINE off

    HLS_CSIM_STATIC int64_dt URAM_W[TILE_SIZE][HIDDEN_DIM_package];
#pragma HLS ARRAY_PARTITION variable=URAM_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=URAM_W cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=URAM_W type=ram_2p impl=uram

    int total_chunks = TILE_SIZE * (HIDDEN_DIM / 32);
    load_W_flat:
    for (int k = 0; k < total_chunks; k++) {
#pragma HLS PIPELINE II=1
        uint256_dt chunk = FIFO_W2.read();
        int i  = k / (HIDDEN_DIM / 32);
        int jb = (k % (HIDDEN_DIM / 32)) * 4;
        for (int m = 0; m < 4; m++) {
#pragma HLS UNROLL
            URAM_W[i][jb + m] = chunk.range(m*64+63, m*64);
        }
    }

    compute_seq:
    for (int i_base = 0; i_base < SEQ_LEN_PADDED; i_base += TOKEN_TILE) {
#pragma HLS LOOP_FLATTEN off

        int26_dt acc_reg[TOKEN_TILE][TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc_reg complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_reg complete dim=2

        init_acc:
        for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                acc_reg[it][t] = 0;
            }
        }
        compute_dot:
        for (int k = 0; k < HIDDEN_DIM_package; k++) {
#pragma HLS PIPELINE II=1
            int64_dt hp_tile[TOKEN_TILE];
#pragma HLS ARRAY_PARTITION variable=hp_tile complete
            for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
                hp_tile[it] = H_package[i_base + it][k];
            }

            mac_outputs:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                int64_dt wp = URAM_W[t][k];

                for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS UNROLL
                    int64_dt hp = hp_tile[it];
                    int26_dt s  = acc_reg[it][t];
                    for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                        int sb = r * 8, eb = sb + 7;
                        int8_dt hv = (int8_dt)hp.range(eb, sb);
                        int8_dt wv = (int8_dt)wp.range(eb, sb);
                        s += hv * wv;
                    }
                    acc_reg[it][t] = s;
                }
            }
        }

        post_token_loop:
        for (int it = 0; it < TOKEN_TILE; it++) {
#pragma HLS LOOP_FLATTEN off
            int   i     = i_base + it;
            float s_row = scaler_H[i];

            write_out_loop:
            for (int j = 0; j < TILE_SIZE; j += 8) {
#pragma HLS PIPELINE II=1
                uint256_dt chunk = 0;
                for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                    int col = tile_idx * TILE_SIZE + j + r;
                    float combined_scale = s_row * SRAM_sw2[col];
                    float f = (float)acc_reg[it][j + r] * combined_scale + SRAM_b2[col];
                    int   sb = r * 32, eb = sb + 31;
                    union { float f; uint32_t i; } cv;
                    cv.f = f;
                    chunk.range(eb, sb) = cv.i;
                }
                FIFO_OUT.write(chunk);
            }
        }
    }
}

void store_Output_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_OUT,
    uint256_dt *output,
    int tile_idx,
    int token_offset,
    int valid_seq_len)
{
    int chunks_per_tile = TILE_SIZE / 8;
    int tile_offset     = tile_idx * chunks_per_tile;
    int chunks_per_row  = EMBED_DIM / 8;

    store_seq:
    for (int i = 0; i < SEQ_LEN_PADDED; i++) {
#pragma HLS LOOP_FLATTEN off
        int  source_token = token_offset + i;
        bool in_range     = (i < SEQ_LEN);
        bool valid_token  = in_range && (source_token < valid_seq_len);

        store_tile:
        for (int j = 0; j < chunks_per_tile; j++) {
#pragma HLS PIPELINE II=1
            uint256_dt chunk = FIFO_OUT.read();   // always drain
            if (valid_token) {
                int idx = source_token * chunks_per_row + tile_offset + j;
                output[idx] = chunk;
            }
        }
    }
}

float gelu_lut_lookup(float x)
{
#pragma HLS INLINE
    if (x <= GELU_X_MIN) return 0.0f;
    if (x >= GELU_X_MAX) return x;

    float pos  = (x - GELU_X_MIN) * GELU_INV_STEP;
    int   idx  = (int)pos;
    float frac = pos - (float)idx;

    if (idx < 0)               idx = 0;
    if (idx > GELU_LUT_SIZE-1) idx = GELU_LUT_SIZE - 1;

    float y0 = GELU_LUT[idx];
    float y1 = GELU_LUT[idx + 1];
    return y0 + frac * (y1 - y0);
}
