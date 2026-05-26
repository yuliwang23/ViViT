#include "mlp.h"
#include "gelu_lut.h"

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
// m_axi depths sized for the production GPT-2 small MLP:
//   input:        197 * EMBED_DIM / 32            = 197 * 768 / 32 = 4728
//   output:       197 * EMBED_DIM / 8             = 197 * 768 / 8  = 18912
//   W1:           HIDDEN_DIM * EMBED_DIM  / 32    = 3072 * 768/ 32 = 73728
//   W2:           EMBED_DIM  * HIDDEN_DIM / 32    = 768  * 3072/32 = 73728
//   b1,sw1:       HIDDEN_DIM                       = 3072
//   b2,sw2:       EMBED_DIM                        = 768
// These depths are needed for cosim to allocate the correct AXI
// memory model size, otherwise the testbench writes past the end
// and SIGSEGVs inside CALL_C_DUT.
#pragma HLS INTERFACE m_axi port=input    offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_EMBED_INT8_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=input_scaler offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_SEQ_LEN
#pragma HLS INTERFACE m_axi port=output   offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_EMBED_FP_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=W1       offset=slave bundle=gmem1 max_widen_bitwidth=256 depth=VIVIT_MLP_WEIGHT_INT8_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=W2       offset=slave bundle=gmem2 max_widen_bitwidth=256 depth=VIVIT_MLP_WEIGHT_INT8_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=b1       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_MLP_DIM
#pragma HLS INTERFACE m_axi port=b2       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_EMBED_DIM
#pragma HLS INTERFACE m_axi port=scale_w1 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_MLP_DIM
#pragma HLS INTERFACE m_axi port=scale_w2 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_EMBED_DIM

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

    int64_dt X_package[SEQ_LEN][EMBED_DIM_package];
    float    scaler_X [SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE    variable=X_package type=ram_2p impl=uram

    int64_dt H_package[SEQ_LEN][HIDDEN_DIM_package];
    float    scaler_H [SEQ_LEN];
#pragma HLS ARRAY_PARTITION variable=H_package type=cyclic factor=2 dim=2
#pragma HLS BIND_STORAGE    variable=H_package type=ram_2p impl=uram

    float H_full_fp[SEQ_LEN][HIDDEN_DIM];
#pragma HLS BIND_STORAGE    variable=H_full_fp type=ram_2p impl=uram
#pragma HLS ARRAY_PARTITION variable=H_full_fp dim=2 type=cyclic factor=8

    float SRAM_b1 [HIDDEN_DIM];
    float SRAM_b2 [EMBED_DIM];
    float SRAM_sw1[HIDDEN_DIM];
    float SRAM_sw2[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=SRAM_b1  dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_b2  dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_sw1 dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_sw2 dim=1 type=cyclic factor=8

    load_Bias_HBM2BRAM(b1,       SRAM_b1,  HIDDEN_DIM);
    load_Bias_HBM2BRAM(b2,       SRAM_b2,  EMBED_DIM );
    load_Bias_HBM2BRAM(scale_w1, SRAM_sw1, HIDDEN_DIM);
    load_Bias_HBM2BRAM(scale_w2, SRAM_sw2, EMBED_DIM );

    // Stage 1: load pre-quantized LayerNorm output directly.
    //   (No DATAFLOW / FIFO — read HBM directly into row buffer.
    //    The previous DATAFLOW version deadlocked on hardware because
    //    the producer wrote 6144 chunks into a depth-32 FIFO while the
    //    consumer was a plain loop, not a separate dataflow process.)
    seq_load_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int source_token = token_offset + i;
        bool valid_token = (source_token < valid_seq_len);
        scaler_X[i] = valid_token ? input_scaler[source_token] : 1.0f;

        unpack_row:
        for (int j = 0; j < EMBED_DIM / 32; j++) {
#pragma HLS PIPELINE II=1
            uint256_dt chunk = valid_token ? input[source_token * (EMBED_DIM / 32) + j] : (uint256_dt)0;
            for (int k = 0; k < 4; k++) {
#pragma HLS UNROLL
                int s = k * 64, e = s + 63;
                X_package[i][j * 4 + k] = chunk.range(e, s);
            }
        }
    }

    // Stage 2:  Dense_0 + GELU + (running-max + stash)
    init_max:
    for (int i = 0; i < SEQ_LEN; i++) {
#pragma HLS PIPELINE II=1
        scaler_H[i] = 0.0f;
    }

    dense0_tile_loop:
    for (int iter = 0; iter < HIDDEN_DIM / TILE_SIZE; iter++) {
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

    // Stage 2b:  per-row quantize hidden activation
    quant_hidden_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
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
                float    f  = H_full_fp[i][j + k] * scale;
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
}
} // extern "C"

void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X)
{
    load_X_loop:
    for (int i = 0; i < (SEQ_LEN * EMBED_DIM) / 8; i++) {
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

// Flattened weight loader — single outer PIPELINE, no nested pipeline
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

// Per-row dynamic quantization — reduction-tree max (no timing violation)
// find_max design: 8 parallel partial-max lanes, each lane updates
// its own pmax[k] every II=2 cycles (gives the comparator + writeback
// 2 clocks to close timing at 500 MHz).  Then a combinational tree
// reduces the 8 partials.  This loop runs only 96 times per row,
// so II=2 vs II=1 costs ~96 cycles — negligible.
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
        // Stage 1: register the absolute values
        float a[8];
#pragma HLS ARRAY_PARTITION variable=a complete
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            float v = row_fp[j + k];
            a[k] = (v < 0.0f) ? -v : v;
        }
        // Stage 2: compare and update.  Each lane only touches pmax[k].
        for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
            if (a[k] > pmax[k]) pmax[k] = a[k];
        }
    }

    // Combinational tree-reduce of 8 partials
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
    int64_dt  X_package [SEQ_LEN][EMBED_DIM_package],
    float     scaler_X  [SEQ_LEN],
    hls::stream<uint256_dt> &FIFO_W1,
    float     SRAM_b1   [HIDDEN_DIM],
    float     SRAM_sw1  [HIDDEN_DIM],
    float     H_full_fp [SEQ_LEN][HIDDEN_DIM],
    float     scaler_H  [SEQ_LEN],
    int       tile_idx)
{
#pragma HLS INLINE off

    int64_dt URAM_W[TILE_SIZE][EMBED_DIM_package];
#pragma HLS ARRAY_PARTITION variable=URAM_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=URAM_W cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=URAM_W type=ram_2p impl=uram

    // flattened weight load
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
    for (int i = 0; i < SEQ_LEN; i++) {

        int26_dt acc_reg[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc_reg complete

        init_acc:
        for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
            acc_reg[t] = 0;
        }

        // single pipelined K-loop, all TILE_SIZE outputs in parallel
        compute_dot:
        for (int k = 0; k < EMBED_DIM_package; k++) {
#pragma HLS PIPELINE II=1
            int64_dt xp = X_package[i][k];

            mac_outputs:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                int64_dt wp = URAM_W[t][k];
                int26_dt s  = acc_reg[t];
                for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                    int sb = r * 8, eb = sb + 7;
                    int8_dt xv = (int8_dt)xp.range(eb, sb);
                    int8_dt wv = (int8_dt)wp.range(eb, sb);
                    s += xv * wv;
                }
                acc_reg[t] = s;
            }
        }

        // ---- post-processing ----
        // Path: int->FP cast + 2 muls + add + GELU(2 ROM reads + sub + mul + add)
        //       + abs + scalar compare/write was 1 long combinational path.
        // Split into two short pipelined loops + tree-reduce the max.
        float s_row   = scaler_X[i];
        float h_tile[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=h_tile complete

        // Loop 1: dequant + bias + GELU + stash, all II=1, shorter path.
        gelu_loop:
        for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
            int   col = tile_idx * TILE_SIZE + t;
            float combined_scale = s_row * SRAM_sw1[col];
            float dequant   = (float)acc_reg[t] * combined_scale;
            float biased    = dequant + SRAM_b1[col];
            float activated = gelu_lut_lookup(biased);
            H_full_fp[i][col] = activated;
            h_tile[t] = activated;
        }

        // Loop 2: abs values into 8 partial-max lanes (II=2 OK; runs
        // TILE_SIZE/8 = 4 times only, cost is negligible).
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
        // Combinational 3-level max-tree of 8 partials
        float m01   = (pmax[0] > pmax[1]) ? pmax[0] : pmax[1];
        float m23   = (pmax[2] > pmax[3]) ? pmax[2] : pmax[3];
        float m45   = (pmax[4] > pmax[5]) ? pmax[4] : pmax[5];
        float m67   = (pmax[6] > pmax[7]) ? pmax[6] : pmax[7];
        float m0123 = (m01 > m23) ? m01 : m23;
        float m4567 = (m45 > m67) ? m45 : m67;
        float tile_max = (m0123 > m4567) ? m0123 : m4567;

        // Update running max across tiles
        float prev_max = scaler_H[i];
        scaler_H[i] = (tile_max > prev_max) ? tile_max : prev_max;
    }
}

void dense1_one_tile(
    int64_dt  H_package [SEQ_LEN][HIDDEN_DIM_package],
    float     scaler_H  [SEQ_LEN],
    hls::stream<uint256_dt> &FIFO_W2,
    float     SRAM_b2   [EMBED_DIM],
    float     SRAM_sw2  [EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_OUT,
    int       tile_idx)
{
#pragma HLS INLINE off

    int64_dt URAM_W[TILE_SIZE][HIDDEN_DIM_package];
#pragma HLS ARRAY_PARTITION variable=URAM_W complete dim=1
#pragma HLS ARRAY_PARTITION variable=URAM_W cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=URAM_W type=ram_2p impl=uram

    // flattened weight load
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
    for (int i = 0; i < SEQ_LEN; i++) {

        int26_dt acc_reg[TILE_SIZE];
#pragma HLS ARRAY_PARTITION variable=acc_reg complete

        init_acc:
        for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
            acc_reg[t] = 0;
        }

        compute_dot:
        for (int k = 0; k < HIDDEN_DIM_package; k++) {
#pragma HLS PIPELINE II=1
            int64_dt hp = H_package[i][k];

            mac_outputs:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS UNROLL
                int64_dt wp = URAM_W[t][k];
                int26_dt s  = acc_reg[t];
                for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                    int sb = r * 8, eb = sb + 7;
                    int8_dt hv = (int8_dt)hp.range(eb, sb);
                    int8_dt wv = (int8_dt)wp.range(eb, sb);
                    s += hv * wv;
                }
                acc_reg[t] = s;
            }
        }

        // dequant + bias + pack into 256-bit chunks
        float s_row = scaler_H[i];

        write_out_loop:
        for (int j = 0; j < TILE_SIZE; j += 8) {
#pragma HLS PIPELINE II=1
            uint256_dt chunk = 0;
            for (int r = 0; r < 8; r++) {
#pragma HLS UNROLL
                int col = tile_idx * TILE_SIZE + j + r;
                float combined_scale = s_row * SRAM_sw2[col];
                float f = (float)acc_reg[j + r] * combined_scale + SRAM_b2[col];
                int   sb = r * 32, eb = sb + 31;
                union { float f; uint32_t i; } cv;
                cv.f = f;
                chunk.range(eb, sb) = cv.i;
            }
            FIFO_OUT.write(chunk);
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
    for (int i = 0; i < SEQ_LEN; i++) {
        int source_token = token_offset + i;
        bool valid_token = (source_token < valid_seq_len);
        store_tile:
        for (int j = 0; j < chunks_per_tile; j++) {
#pragma HLS PIPELINE II=1
            uint256_dt chunk = FIFO_OUT.read();
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
