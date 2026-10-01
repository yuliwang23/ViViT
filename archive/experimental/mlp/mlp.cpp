#include "mlp.h"
#include "fast_math.h"

#ifndef __SYNTHESIS__
#define HLS_CSIM_STATIC static
#else
#define HLS_CSIM_STATIC
#endif

extern "C" {
void mlp_q(
    uint256_dt *input,
    uint256_dt *gamma,
    uint256_dt *beta,
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

#pragma HLS INTERFACE m_axi port=input    offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=18912
#pragma HLS INTERFACE m_axi port=gamma    offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=96 
#pragma HLS INTERFACE m_axi port=beta     offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=96 
#pragma HLS INTERFACE m_axi port=output   offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=18912
#pragma HLS INTERFACE m_axi port=W1       offset=slave bundle=gmem1 max_widen_bitwidth=256 depth=73728
#pragma HLS INTERFACE m_axi port=W2       offset=slave bundle=gmem2 max_widen_bitwidth=256 depth=73728
#pragma HLS INTERFACE m_axi port=b1       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=3072
#pragma HLS INTERFACE m_axi port=b2       offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=768
#pragma HLS INTERFACE m_axi port=scale_w1 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=3072
#pragma HLS INTERFACE m_axi port=scale_w2 offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=768

#pragma HLS INTERFACE s_axilite port=input    bundle=control
#pragma HLS INTERFACE s_axilite port=gamma    bundle=control
#pragma HLS INTERFACE s_axilite port=beta     bundle=control
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

    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8

    HLS_CSIM_STATIC int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package];
    HLS_CSIM_STATIC float    scaler_X [SEQ_LEN_PADDED];
#pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=X_package type=ram_2p impl=bram

    HLS_CSIM_STATIC int64_dt H_package[SEQ_LEN_PADDED][HIDDEN_DIM_package];
    HLS_CSIM_STATIC float    scaler_H [SEQ_LEN_PADDED];
#pragma HLS ARRAY_PARTITION variable=H_package type=cyclic factor=4 dim=1
#pragma HLS BIND_STORAGE    variable=H_package type=ram_2p impl=uram

    HLS_CSIM_STATIC uint128_dt H_full_pack[SEQ_LEN_PADDED][HIDDEN_DIM_package];
#pragma HLS BIND_STORAGE    variable=H_full_pack type=ram_1p impl=uram

    HLS_CSIM_STATIC float SRAM_b1 [HIDDEN_DIM];
    HLS_CSIM_STATIC float SRAM_b2 [EMBED_DIM];
    HLS_CSIM_STATIC float SRAM_sw1[HIDDEN_DIM];
    HLS_CSIM_STATIC float SRAM_sw2[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable=SRAM_b2  dim=1 type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_sw2 dim=1 type=cyclic factor=8

    load_beta_gamma_HBM2SRAM(gamma, beta, SRAM_gamma, SRAM_beta);

    load_Bias_HBM2BRAM(b1,       SRAM_b1,  HIDDEN_DIM);
    load_Bias_HBM2BRAM(b2,       SRAM_b2,  EMBED_DIM );
    load_Bias_HBM2BRAM(scale_w1, SRAM_sw1, HIDDEN_DIM);
    load_Bias_HBM2BRAM(scale_w2, SRAM_sw2, EMBED_DIM );

    do_layernorm_dataflow(input, SRAM_gamma, SRAM_beta, scaler_X, X_package, token_offset, valid_seq_len);

    pad_clear_loop:
    for (int i = SEQ_LEN; i < SEQ_LEN_PADDED; i++) {
#pragma HLS LOOP_FLATTEN off
        scaler_X[i] = 1.0f;

        clear_x_package:
        for (int j = 0; j < EMBED_DIM_package; j++) {
#pragma HLS PIPELINE II=1
            X_package[i][j] = 0;
        }
    }

    dense0_tile_loop:
    for (int iter = 0; iter < HIDDEN_DIM / TILE_SIZE; iter++) {
        dense0_dataflow_tile(
            W1,
            X_package, scaler_X,
            SRAM_b1, SRAM_sw1,
            H_full_pack,
            scaler_H,
            iter);
    }

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
            uint128_dt h_pack = H_full_pack[i][j >> 3];
            for (int k = 0; k < 8; k++) {
#pragma HLS UNROLL
                ap_uint<16> h_bits = h_pack.range(k * 16 + 15, k * 16);
                half_dt h_val = 0;
                h_val.range(15, 0) = h_bits;
                float    f  = (float)h_val * scale;
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
    uint128_dt  H_full_pack[SEQ_LEN_PADDED][HIDDEN_DIM_package],
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
        H_full_pack,
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
    hls::stream<uint256_dt> &FIFO_X,
    int token_offset,
    int valid_seq_len)
{
    const int chunks_per_row = EMBED_DIM / 8;

    load_X_loop:
    for (int idx = 0; idx < SEQ_LEN * chunks_per_row; idx++) {
#pragma HLS PIPELINE II=1
        int i = idx / chunks_per_row;
        int j = idx % chunks_per_row;
        int source_token = token_offset + i;
        bool valid_token = (source_token < valid_seq_len);

        uint256_dt temp_chunk = valid_token ? HBM_X[source_token * chunks_per_row + j] : (uint256_dt)0;
        FIFO_X.write(temp_chunk);
    }
}

void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM])
{
   load_X_col:
   for (int j = 0; j < EMBED_DIM/8; j++){
       #pragma HLS PIPELINE II=1
       
       uint256_dt temp_chunk = FIFO_X.read();
       
       unpack_write_loop:
       for(int k = 0; k < 8; k++) {
           #pragma HLS UNROLL 
           int start_bit = k * 32;
           int end_bit = start_bit + 31;
           
           uint32_t raw_bits = temp_chunk.range(end_bit, start_bit);
           
           union {
               uint32_t i_val;
               float f_val;
           } converter;
           
           converter.i_val = raw_bits;
           BRAM_X[j*8+k] = converter.f_val; 
       }
   }
}

void load_beta_gamma_HBM2SRAM(
    uint256_dt HBM_gamma[EMBED_DIM / 8],
    uint256_dt HBM_beta[EMBED_DIM / 8],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
) {
    #pragma HLS INLINE off
    load_weights_loop:
    for (int i = 0; i < EMBED_DIM / 8; i++) {
        #pragma HLS PIPELINE II=2
        
        uint256_dt temp_gamma_chunk = HBM_gamma[i];
        uint256_dt temp_beta_chunk  = HBM_beta[i];
        
        unpack_write_loop:
        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            
            uint32_t raw_gamma_bits = temp_gamma_chunk.range(end_bit, start_bit);
            union { uint32_t i_val; float f_val; } conv_gamma;
            conv_gamma.i_val = raw_gamma_bits;
            SRAM_gamma[i*8 + k] = conv_gamma.f_val;
            
            uint32_t raw_beta_bits = temp_beta_chunk.range(end_bit, start_bit);
            union { uint32_t i_val; float f_val; } conv_beta;
            conv_beta.i_val = raw_beta_bits;
            SRAM_beta[i*8 + k] = conv_beta.f_val;
        }
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

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN_PADDED],
    int64_dt X_package_array[SEQ_LEN_PADDED][EMBED_DIM_package],
    int seq_idx
){
    float global_mean = 0.0f; 
    float M2AB = 0.0f;
    float norm_buffer[EMBED_DIM];
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_X type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=norm_buffer type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=gamma type=cyclic factor= 8
    #pragma HLS ARRAY_PARTITION variable=beta  type=cyclic factor= 8 

    compute_top_loop:
    for(int j = 0, i = 0; j < EMBED_DIM; j+=16, i++){     
        #pragma HLS PIPELINE II = 6
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        compute_local_sum:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            float val = BRAM_X[j+k];
            local_sum += val;
            local_sum_sq += val * val;
        }
        float X_b_mean = local_sum * 0.0625f; 
        float mean_diff = X_b_mean - global_mean;
        float M_2_B = local_sum_sq - (local_sum * X_b_mean);
        float mean_diff_sq = mean_diff * mean_diff;
        float mean_diff_mul_table_val = MEAN_WT[i] * mean_diff;
        float mean_diff_sq_mul_table_val = M2_WT[i] * mean_diff_sq;
        global_mean = mean_diff_mul_table_val + global_mean;
        M2AB = mean_diff_sq_mul_table_val + M_2_B + M2AB;
    }
    
    float var = M2AB * EMBED_DIM_INV; 
    float inv_std_out = 1.0f / std::sqrt(var + EPSILON); 
    float max_val = 0.0f;
    
    compute_norm_and_find_max_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE II=1
        float local_max = 0.0f;
        
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            int idx = j + k;
            float norm_val = (BRAM_X[idx] - global_mean) * inv_std_out * gamma[idx] + beta[idx];
            
            norm_buffer[idx] = norm_val;
            
            float abs_val = std::abs(norm_val);
            local_max = std::max(local_max, abs_val);
        }
        max_val = std::max(max_val, local_max);
    }
    
    max_val = std::max(max_val, EPSILON);
    float scale = 127.5f / max_val;
    
    quant_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE II=1
        
        int64_dt chunk0 = 0; 
        int64_dt chunk1 = 0;

        pack_loop:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            int idx = j + k;
            float scaled_val = norm_buffer[idx] * scale; 
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            int8_dt q_val = static_cast<int8_dt>(std::round(scaled_val)); 
            
            int bit_offset = (k % 8) * 8; 
            
            if (k < 8) {
                chunk0.range(bit_offset + 7, bit_offset) = q_val;
            } else {
                chunk1.range(bit_offset + 7, bit_offset) = q_val;
            }
        }
        
        int pkg_idx = j >> 3; 
        X_package_array[seq_idx][pkg_idx]     = chunk0;
        X_package_array[seq_idx][pkg_idx + 1] = chunk1;
    }
    scaler_array[seq_idx] = 1.0f / scale;
}

void layernorm_compute(
    hls::stream<uint256_dt> &FIFO_X,
    float scaler[SEQ_LEN_PADDED],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    int token_offset,
    int valid_seq_len)
{
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=2 dim=2
    
    #pragma HLS BIND_STORAGE variable=SRAM_gamma type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SRAM_beta type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=X_package type=ram_2p impl=bram

    seq_loop:
    for(int i = 0; i < SEQ_LEN; i++){
        #pragma HLS stable variable=SRAM_gamma
        #pragma HLS stable variable=SRAM_beta
        #pragma HLS stable variable=scaler
        #pragma HLS stable variable=X_package

        float SRAM_X[EMBED_DIM];
        #pragma HLS ARRAY_PARTITION variable=SRAM_X type=cyclic factor=16
        #pragma HLS BIND_STORAGE variable=SRAM_X type=ram_2p impl=bram
        load_X_FIFO2BRAM(FIFO_X, SRAM_X);

        int source_token = token_offset + i;
        bool valid_token = (source_token < valid_seq_len);
        if (valid_token) {
            compute_mean_variance(SRAM_X, SRAM_gamma, SRAM_beta, scaler, X_package, i);
        } else {
            scaler[i] = 1.0f;
            clear_invalid_x_package:
            for (int j = 0; j < EMBED_DIM_package; j++) {
#pragma HLS PIPELINE II=1
                X_package[i][j] = 0;
            }
        }
    }
}

void do_layernorm_dataflow(
    uint256_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN_PADDED],
    int64_dt X_package[SEQ_LEN_PADDED][EMBED_DIM_package],
    int token_offset,
    int valid_seq_len
) {
    #pragma HLS DATAFLOW
    hls::stream<uint256_dt> FIFO_X("FIFO_X");
#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_X depth=32
#endif
    #pragma HLS stable variable=SRAM_gamma
    #pragma HLS stable variable=SRAM_beta
    #pragma HLS stable variable=scaler_X
    #pragma HLS stable variable=X_package
    load_X_HBM2FIFO(x, FIFO_X, token_offset, valid_seq_len);
    layernorm_compute(FIFO_X, scaler_X, SRAM_gamma, SRAM_beta, X_package, token_offset, valid_seq_len);
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
    uint128_dt H_full_pack[SEQ_LEN_PADDED][HIDDEN_DIM_package],
    float     scaler_H  [SEQ_LEN_PADDED],
    int       tile_idx)
{
#pragma HLS INLINE off
    HLS_CSIM_STATIC int64_dt W_tile[TILE_SIZE][EMBED_DIM_package];
#pragma HLS ARRAY_PARTITION variable=W_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=W_tile cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=W_tile type=ram_1p impl=lutram

    int total_chunks = TILE_SIZE * (EMBED_DIM / 32);
    load_W_flat:
    for (int k = 0; k < total_chunks; k++) {
#pragma HLS PIPELINE II=1
        uint256_dt chunk = FIFO_W1.read();
        int i  = k / (EMBED_DIM / 32);
        int jb = (k % (EMBED_DIM / 32)) * 4;
        for (int m = 0; m < 4; m++) {
#pragma HLS UNROLL
            W_tile[i][jb + m] = chunk.range(m*64+63, m*64);
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
                int64_dt wp = W_tile[t][k];

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

            uint128_dt h_pack = 0;

            gelu_loop:
            for (int t = 0; t < TILE_SIZE; t++) {
#pragma HLS PIPELINE II=1
                int   col            = tile_idx * TILE_SIZE + t;
                float combined_scale = s_row * SRAM_sw1[col];
                float dequant        = (float)acc_reg[it][t] * combined_scale;
                float biased         = dequant + SRAM_b1[col];
                float activated      = gelu_approx_top(biased);  // ??? ?��??��??
                half_dt activated_half = (half_dt)activated;
                int lane = t & 7;
                ap_uint<16> h_bits = activated_half.range(15, 0);
                h_pack.range(lane * 16 + 15, lane * 16) = h_bits;
                if (lane == 7) {
                    H_full_pack[i][col >> 3] = h_pack;
                    h_pack = 0;
                }
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

    HLS_CSIM_STATIC int64_dt W_tile[TILE_SIZE][HIDDEN_DIM_package];
#pragma HLS ARRAY_PARTITION variable=W_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=W_tile cyclic factor=4 dim=2
#pragma HLS BIND_STORAGE    variable=W_tile type=ram_1p impl=lutram

    int total_chunks = TILE_SIZE * (HIDDEN_DIM / 32);
    load_W_flat:
    for (int k = 0; k < total_chunks; k++) {
#pragma HLS PIPELINE II=1
        uint256_dt chunk = FIFO_W2.read();
        int i  = k / (HIDDEN_DIM / 32);
        int jb = (k % (HIDDEN_DIM / 32)) * 4;
        for (int m = 0; m < 4; m++) {
#pragma HLS UNROLL
            W_tile[i][jb + m] = chunk.range(m*64+63, m*64);
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
                int64_dt wp = W_tile[t][k];

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
            uint256_dt chunk = FIFO_OUT.read();
            if (valid_token) {
                int idx = source_token * chunks_per_row + tile_offset + j;
                output[idx] = chunk;
            }
        }
    }
}