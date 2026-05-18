#include "linear_qkv_multihead.h"
extern "C"{
void linear_qkv_multihead(
    uint256_dt *x,
    uint256_dt *gamma,
    uint256_dt *beta,
    uint256_dt *Wk,
    uint256_dt *Wv,
    uint256_dt *Wq,
    float *bq,
    float *bk,
    float *bv,
    float *scale_wq,
    float *scale_wk,
    float *scale_wv,
    uint256_dt *Q,
    uint256_dt *K,
    uint256_dt *V)
{
    // Group 1: Native 256-bit Input Feature Map & LayerNorm Weights
    #pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem_x     depth=18912 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=gamma    offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=beta     offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    
    // Group 2: Native 32-bit Floating-Point Vectors (Separated to avoid wrapc struct alignment crash)
    #pragma HLS INTERFACE m_axi port=bq       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=bk       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=bv       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wq offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wk offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wv offset=slave bundle=gmem_bias  depth=768

    // Group 3: Q-Stream Weights & Outputs (Matched 256-bit width)
    #pragma HLS INTERFACE m_axi port=Wq       offset=slave bundle=gmem_q     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=Q        offset=slave bundle=gmem_q     depth=18912 max_widen_bitwidth=256
    
    // Group 4: K-Stream Weights & Outputs (Matched 256-bit width)
    #pragma HLS INTERFACE m_axi port=Wk       offset=slave bundle=gmem_k     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=K        offset=slave bundle=gmem_k     depth=18912 max_widen_bitwidth=256
    
    // Group 5: V-Stream Weights & Outputs (Matched 256-bit width)
    #pragma HLS INTERFACE m_axi port=Wv       offset=slave bundle=gmem_v     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=V        offset=slave bundle=gmem_v     depth=18912 max_widen_bitwidth=256

    // S_AXILite Control Register Interfaces
    #pragma HLS INTERFACE s_axilite port=x        bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma    bundle=control
    #pragma HLS INTERFACE s_axilite port=beta     bundle=control
    #pragma HLS INTERFACE s_axilite port=Wk       bundle=control
    #pragma HLS INTERFACE s_axilite port=V        bundle=control
    #pragma HLS INTERFACE s_axilite port=Wq       bundle=control
    #pragma HLS INTERFACE s_axilite port=Wv       bundle=control
    #pragma HLS INTERFACE s_axilite port=bq       bundle=control
    #pragma HLS INTERFACE s_axilite port=bk       bundle=control
    #pragma HLS INTERFACE s_axilite port=bv       bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wq bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wk bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wv bundle=control
    #pragma HLS INTERFACE s_axilite port=Q        bundle=control
    #pragma HLS INTERFACE s_axilite port=K        bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8
    
    float SRAM_bq[EMBED_DIM]; float SRAM_bk[EMBED_DIM]; float SRAM_bv[EMBED_DIM];
    float SRAM_swq[EMBED_DIM]; float SRAM_swk[EMBED_DIM]; float SRAM_swv[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=SRAM_bq dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_bk dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_bv dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_swq dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_swk dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_swv dim=1 type=cyclic factor=8

    static int64_dt X_package[SEQ_LEN][EMBED_DIM_package];
    float scaler_X[SEQ_LEN];
    #pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=2 dim=2
    #pragma HLS BIND_STORAGE variable=X_package type=ram_2p impl=uram

    load_beta_gamma_HBM2SRAM(gamma, beta, SRAM_gamma, SRAM_beta);
    
    load_Bias_HBM2BRAM(bq, SRAM_bq);
    load_Bias_HBM2BRAM(bk, SRAM_bk);
    load_Bias_HBM2BRAM(bv, SRAM_bv);
    
    load_Bias_HBM2BRAM(scale_wq, SRAM_swq);
    load_Bias_HBM2BRAM(scale_wk, SRAM_swk);
    load_Bias_HBM2BRAM(scale_wv, SRAM_swv);

    do_layernorm_dataflow(x, SRAM_gamma, SRAM_beta, scaler_X, X_package);

    tile_loop:
    for (int iter = 0; iter < EMBED_DIM / TILE_SIZE; iter++) {
        process_single_tile(
            X_package,
            Wq, Wk, Wv,
            Q, K, V,
            SRAM_bq, SRAM_bk, SRAM_bv,
            SRAM_swq, SRAM_swk, SRAM_swv,
            scaler_X,
            iter
        );
    }
}
}

void load_X_HBM2FIFO(
    uint256_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_X)
{
    load_X_loop:
    for (int i = 0; i < (SEQ_LEN * EMBED_DIM) / 8; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = HBM_X[i];
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

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package],
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
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package])
{
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=2 dim=2
    
    #pragma HLS BIND_STORAGE variable=SRAM_gamma type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SRAM_beta type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=X_package type=ram_2p impl=uram

    seq_loop:
    for(int i = 0; i < SEQ_LEN; i++){
        // #pragma HLS DATAFLOW 
        #pragma HLS stable variable=SRAM_gamma
        #pragma HLS stable variable=SRAM_beta
        #pragma HLS stable variable=scaler
        #pragma HLS stable variable=X_package

        float SRAM_X[EMBED_DIM];
        #pragma HLS ARRAY_PARTITION variable=SRAM_X type=cyclic factor=16
        #pragma HLS BIND_STORAGE variable=SRAM_X type=ram_2p impl=bram
        load_X_FIFO2BRAM(FIFO_X, SRAM_X);
        compute_mean_variance(SRAM_X, SRAM_gamma, SRAM_beta, scaler, X_package, i);
    }
}

void load_Weight_tile_HBM2FIFO(
    uint256_dt *W,
    hls::stream<uint256_dt> &FIFO_W,
    int iter
) {  
    int row_offset = iter * TILE_SIZE; 
    int chunks_per_row = EMBED_DIM / 32; 
    int base_addr_idx = row_offset * chunks_per_row; 

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start = base_addr_idx + (i * chunks_per_row);

        load_tile_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = W[current_row_start + j];
            FIFO_W.write(temp_chunk);
        }
    }
}

void load_Bias_HBM2BRAM(
    float b[EMBED_DIM],
    float BRAM_b[EMBED_DIM])
{
    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
    #pragma HLS PIPELINE II=1
        BRAM_b[i] = b[i];
    }
}

// void load_Weight_tile_FIFO2URAM(
//     hls::stream<uint256_dt> &FIFO_W,
//     int64_dt URAM_W[TILE_SIZE][EMBED_DIM_package]
// ) {
//     #pragma HLS ARRAY_PARTITION variable=URAM_W type=cyclic factor=2 dim=1
//     #pragma HLS BIND_STORAGE variable=URAM_W type=ram_2p impl=uram

//     load_local_row:
//     for (int i = 0; i < TILE_SIZE; i++) { 
//         load_local_col:
//         for (int j = 0; j < 24; j++) {
//             #pragma HLS PIPELINE II=4
//             uint256_dt temp_chunk = FIFO_W.read();
//             unpack_loop:
//             for (int k = 0; k < 4; k++) {
//                 #pragma HLS UNROLL
//                 int start_bit = k * 64;
//                 int end_bit = start_bit + 63;
//                 URAM_W[i][j * 4 + k] = temp_chunk.range(end_bit, start_bit);
//             }
//         }
//     }
// }

void load_All_Weights_Parallel(
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    int64_dt URAM_Wq[TILE_SIZE][EMBED_DIM_package], 
    int64_dt URAM_Wk[TILE_SIZE][EMBED_DIM_package], 
    int64_dt URAM_Wv[TILE_SIZE][EMBED_DIM_package]
) {
    #pragma HLS INLINE off
    load_local_row:
    for (int i = 0; i < TILE_SIZE; i++) { 
        load_local_col:
        for (int j = 0; j < 24; j++) {
            #pragma HLS PIPELINE II=4
            
            uint256_dt q_chunk = FIFO_Wq.read();
            uint256_dt k_chunk = FIFO_Wk.read();
            uint256_dt v_chunk = FIFO_Wv.read();
            
            unpack_loop:
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 64;
                int end_bit = start_bit + 63;
                URAM_Wq[i][j * 4 + k] = q_chunk.range(end_bit, start_bit);
                URAM_Wk[i][j * 4 + k] = k_chunk.range(end_bit, start_bit);
                URAM_Wv[i][j * 4 + k] = v_chunk.range(end_bit, start_bit);
            }
        }
    }
}

void compute_linear_qkv_one_head(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out,
    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],
    float scale_wq[EMBED_DIM],
    float scale_wk[EMBED_DIM],
    float scale_wv[EMBED_DIM],
    float scale_X [SEQ_LEN],
    int iter
){
    #pragma HLS INLINE off

    #ifdef __SYNTHESIS__
    // Hardware Synthesis: Local array allows HLS scheduler to correctly map to physical URAM blocks
    int64_dt URAM_Wq[TILE_SIZE][EMBED_DIM_package];
    int64_dt URAM_Wk[TILE_SIZE][EMBED_DIM_package];
    int64_dt URAM_Wv[TILE_SIZE][EMBED_DIM_package];
    #else
        // C-Simulation: Static array avoids Stack Overflow by allocation on Heap instead of Stack
        static int64_dt URAM_Wq[TILE_SIZE][EMBED_DIM_package];
        static int64_dt URAM_Wk[TILE_SIZE][EMBED_DIM_package];
        static int64_dt URAM_Wv[TILE_SIZE][EMBED_DIM_package];
    #endif

    int26_dt BRAM_Acc_Q[TILE_SIZE];
    int26_dt BRAM_Acc_K[TILE_SIZE];
    int26_dt BRAM_Acc_V[TILE_SIZE];

    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_Q dim=1 type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_K dim=1 type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=BRAM_Acc_V dim=1 type=cyclic factor=16

    #pragma HLS ARRAY_PARTITION variable=BRAM_bq dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_bk dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_bv dim=1 type=cyclic factor=8
    
    #pragma HLS ARRAY_PARTITION variable=scale_wq dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=scale_wk dim=1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=scale_wv dim=1 type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=URAM_Wq type=cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=URAM_Wk type=cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=URAM_Wv type=cyclic factor=2 dim=1

    #pragma HLS BIND_STORAGE variable=BRAM_Acc_Q type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=BRAM_Acc_K type=ram_2p impl=lutram
    #pragma HLS BIND_STORAGE variable=BRAM_Acc_V type=ram_2p impl=lutram

    load_All_Weights_Parallel(
        FIFO_Wq, FIFO_Wk, FIFO_Wv,
        URAM_Wq, URAM_Wk, URAM_Wv
    );

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off 

        compute_outer:
        for (int j = 0; j < TILE_SIZE; j += 4) { 
            int26_dt sum_q0, sum_q1, sum_q2, sum_q3;
            int26_dt sum_k0, sum_k1, sum_k2, sum_k3;
            int26_dt sum_v0, sum_v1, sum_v2, sum_v3;

            compute_dot:
            for (int k_pkg = 0; k_pkg < EMBED_DIM_package; k_pkg++) {
                #pragma HLS PIPELINE II=1
                if (k_pkg == 0) {
                    sum_q0 = 0; sum_q1 = 0; sum_q2 = 0; sum_q3 = 0;
                    sum_k0 = 0; sum_k1 = 0; sum_k2 = 0; sum_k3 = 0;
                    sum_v0 = 0; sum_v1 = 0; sum_v2 = 0; sum_v3 = 0;
                }

                int64_dt x_pkg = X_package[i][k_pkg];

                int64_dt wq_pkg_0 = URAM_Wq[j][k_pkg];   int64_dt wq_pkg_1 = URAM_Wq[j+1][k_pkg];
                int64_dt wq_pkg_2 = URAM_Wq[j+2][k_pkg]; int64_dt wq_pkg_3 = URAM_Wq[j+3][k_pkg];
                
                int64_dt wk_pkg_0 = URAM_Wk[j][k_pkg];   int64_dt wk_pkg_1 = URAM_Wk[j+1][k_pkg];
                int64_dt wk_pkg_2 = URAM_Wk[j+2][k_pkg]; int64_dt wk_pkg_3 = URAM_Wk[j+3][k_pkg];
                
                int64_dt wv_pkg_0 = URAM_Wv[j][k_pkg];   int64_dt wv_pkg_1 = URAM_Wv[j+1][k_pkg];
                int64_dt wv_pkg_2 = URAM_Wv[j+2][k_pkg]; int64_dt wv_pkg_3 = URAM_Wv[j+3][k_pkg];

                MAC_unroll:
                for (int r = 0; r < 8; r++) {
                    #pragma HLS UNROLL
                    int start_bit = r * 8;
                    int end_bit = start_bit + 7;
                    int8_dt x_val = (int8_dt)x_pkg.range(end_bit, start_bit);

                    sum_q0 += x_val * (int8_dt)wq_pkg_0.range(end_bit, start_bit);
                    sum_q1 += x_val * (int8_dt)wq_pkg_1.range(end_bit, start_bit);
                    sum_q2 += x_val * (int8_dt)wq_pkg_2.range(end_bit, start_bit);
                    sum_q3 += x_val * (int8_dt)wq_pkg_3.range(end_bit, start_bit);
                    
                    sum_k0 += x_val * (int8_dt)wk_pkg_0.range(end_bit, start_bit);
                    sum_k1 += x_val * (int8_dt)wk_pkg_1.range(end_bit, start_bit);
                    sum_k2 += x_val * (int8_dt)wk_pkg_2.range(end_bit, start_bit);
                    sum_k3 += x_val * (int8_dt)wk_pkg_3.range(end_bit, start_bit);

                    sum_v0 += x_val * (int8_dt)wv_pkg_0.range(end_bit, start_bit);
                    sum_v1 += x_val * (int8_dt)wv_pkg_1.range(end_bit, start_bit);
                    sum_v2 += x_val * (int8_dt)wv_pkg_2.range(end_bit, start_bit);
                    sum_v3 += x_val * (int8_dt)wv_pkg_3.range(end_bit, start_bit);
                }

                if (k_pkg == EMBED_DIM_package - 1) {
                    BRAM_Acc_Q[j]   = sum_q0; BRAM_Acc_Q[j+1] = sum_q1; BRAM_Acc_Q[j+2] = sum_q2; BRAM_Acc_Q[j+3] = sum_q3;
                    BRAM_Acc_K[j]   = sum_k0; BRAM_Acc_K[j+1] = sum_k1; BRAM_Acc_K[j+2] = sum_k2; BRAM_Acc_K[j+3] = sum_k3;
                    BRAM_Acc_V[j]   = sum_v0; BRAM_Acc_V[j+1] = sum_v1; BRAM_Acc_V[j+2] = sum_v2; BRAM_Acc_V[j+3] = sum_v3;
                }
            }
        }

        write_out_loop:
        for (int j = 0; j < TILE_SIZE; j += 8) {
            #pragma HLS PIPELINE II=1
            uint256_dt out_q = 0; 
            uint256_dt out_k = 0; 
            uint256_dt out_v = 0; 
            
            pack_loop:
            for (int r = 0; r < 8; r++) {
                #pragma HLS UNROLL
                int start_bit = r * 32;
                int end_bit = start_bit + 31;
                int weight_idx = iter * TILE_SIZE + j + r; 
                
                float f_q = (float)BRAM_Acc_Q[j+r];
                float f_k = (float)BRAM_Acc_K[j+r];
                float f_v = (float)BRAM_Acc_V[j+r];
                
                f_q = f_q * scale_X[i];
                f_k = f_k * scale_X[i];
                f_v = f_v * scale_X[i];

                f_q = f_q * scale_wq[weight_idx];
                f_k = f_k * scale_wk[weight_idx];
                f_v = f_v * scale_wv[weight_idx];

                f_q = f_q + BRAM_bq[weight_idx];
                f_k = f_k + BRAM_bk[weight_idx];
                f_v = f_v + BRAM_bv[weight_idx];
                union { float f; uint32_t i; } conv_q, conv_k, conv_v;
                conv_q.f = f_q;
                conv_k.f = f_k;
                conv_v.f = f_v;
                
                out_q.range(end_bit, start_bit) = conv_q.i;
                out_k.range(end_bit, start_bit) = conv_k.i;
                out_v.range(end_bit, start_bit) = conv_v.i;
            }
            FIFO_Q_out.write(out_q);
            FIFO_K_out.write(out_k);
            FIFO_V_out.write(out_v);
        }
    }
}

void store_QKV_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Q_out,
    hls::stream<uint256_dt> &FIFO_K_out,
    hls::stream<uint256_dt> &FIFO_V_out,
    uint256_dt *Q,
    uint256_dt *K,
    uint256_dt *V,
    int iter
) {
    int chunks_per_tile = TILE_SIZE / 8;
    int tile_offset = iter * chunks_per_tile;
    int chunks_per_row = EMBED_DIM / 8; 

    store_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        store_tile_loop:
        for (int j = 0; j < chunks_per_tile; j++) {
            #pragma HLS PIPELINE II=1
            int target_idx = i * chunks_per_row + tile_offset + j;
            
            Q[target_idx] = FIFO_Q_out.read();
            K[target_idx] = FIFO_K_out.read();
            V[target_idx] = FIFO_V_out.read();
        }
    }
}

void do_layernorm_dataflow(
    uint256_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]
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
    load_X_HBM2FIFO(x, FIFO_X);
    layernorm_compute(FIFO_X, scaler_X, SRAM_gamma, SRAM_beta, X_package);
}


void process_single_tile(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *Wq, uint256_dt *Wk, uint256_dt *Wv,
    uint256_dt *Q,  uint256_dt *K,  uint256_dt *V,
    float SRAM_bq[EMBED_DIM], float SRAM_bk[EMBED_DIM], float SRAM_bv[EMBED_DIM],
    float SRAM_swq[EMBED_DIM], float SRAM_swk[EMBED_DIM], float SRAM_swv[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int iter
) {
    
    #pragma HLS DATAFLOW
    
    hls::stream<uint256_dt> FIFO_Wq("FIFO_Wq");
    hls::stream<uint256_dt> FIFO_Wk("FIFO_Wk");
    hls::stream<uint256_dt> FIFO_Wv("FIFO_Wv");
    hls::stream<uint256_dt> FIFO_Q("FIFO_Q");
    hls::stream<uint256_dt> FIFO_K("FIFO_K");
    hls::stream<uint256_dt> FIFO_V("FIFO_V");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wq depth=32
    #pragma HLS STREAM variable=FIFO_Wk depth=32
    #pragma HLS STREAM variable=FIFO_Wv depth=32
    #pragma HLS STREAM variable=FIFO_Q  depth=32
    #pragma HLS STREAM variable=FIFO_K  depth=32
    #pragma HLS STREAM variable=FIFO_V  depth=32
#endif
    #pragma HLS stable variable=X_package
    #pragma HLS stable variable=SRAM_bq
    #pragma HLS stable variable=SRAM_bk
    #pragma HLS stable variable=SRAM_bv
    #pragma HLS stable variable=SRAM_swq
    #pragma HLS stable variable=SRAM_swk
    #pragma HLS stable variable=SRAM_swv
    #pragma HLS stable variable=scaler_X
    load_Weight_tile_HBM2FIFO(Wq, FIFO_Wq, iter);
    load_Weight_tile_HBM2FIFO(Wk, FIFO_Wk, iter);
    load_Weight_tile_HBM2FIFO(Wv, FIFO_Wv, iter);

    compute_linear_qkv_one_head(
        X_package,
        FIFO_Wq, FIFO_Wk, FIFO_Wv,
        FIFO_Q, FIFO_K, FIFO_V,
        SRAM_bq, SRAM_bk, SRAM_bv,
        SRAM_swq, SRAM_swk, SRAM_swv,
        scaler_X, iter
    );

    store_QKV_FIFO2HBM(FIFO_Q, FIFO_K, FIFO_V, Q, K, V, iter);
}
