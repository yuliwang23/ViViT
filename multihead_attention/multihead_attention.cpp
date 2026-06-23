#include "multihead_attention.h"

extern "C"{
void multihead_attention(
    // Input Feature Map (Used for both LayerNorm and Residual)
    uint256_dt *x,
    
    // LayerNorm Weights
    uint256_dt *gamma,
    uint256_dt *beta,
    
    // QKV & Output Projection Weights
    uint256_dt *Wq,
    uint256_dt *Wk,
    uint256_dt *Wv,
    uint256_dt *Wo,
    
    // Biases
    float *bq,
    float *bk,
    float *bv,
    float *bo,
    
    // Scales
    float *scale_wq,
    float *scale_wk,
    float *scale_wv,
    float *scale_wo,
    
    // Final Output
    uint256_dt *out)
{
    // ========================================================================
    // AXI Master Interfaces
    // ========================================================================
    // Group 1: Native 256-bit Input Feature Map & LayerNorm Weights
    #pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem_x     depth=18912 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=gamma    offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=beta     offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    
    // Group 2: Native 32-bit Floating-Point Vectors (Biases & Scales)
    #pragma HLS INTERFACE m_axi port=bq       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=bk       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=bv       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=bo       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wq offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wk offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wv offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_wo offset=slave bundle=gmem_bias  depth=768

    // Group 3: Weights & Output (Matched 256-bit width)
    #pragma HLS INTERFACE m_axi port=Wq       offset=slave bundle=gmem_q     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=Wk       offset=slave bundle=gmem_k     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=Wv       offset=slave bundle=gmem_v     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=Wo       offset=slave bundle=gmem_o     depth=18432 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=out      offset=slave bundle=gmem_o     depth=18912 max_widen_bitwidth=256

    // ========================================================================
    // AXI-Lite Control Register Interfaces
    // ========================================================================
    #pragma HLS INTERFACE s_axilite port=x        bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma    bundle=control
    #pragma HLS INTERFACE s_axilite port=beta     bundle=control
    #pragma HLS INTERFACE s_axilite port=Wq       bundle=control
    #pragma HLS INTERFACE s_axilite port=Wk       bundle=control
    #pragma HLS INTERFACE s_axilite port=Wv       bundle=control
    #pragma HLS INTERFACE s_axilite port=Wo       bundle=control
    #pragma HLS INTERFACE s_axilite port=bq       bundle=control
    #pragma HLS INTERFACE s_axilite port=bk       bundle=control
    #pragma HLS INTERFACE s_axilite port=bv       bundle=control
    #pragma HLS INTERFACE s_axilite port=bo       bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wq bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wk bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wv bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wo bundle=control
    #pragma HLS INTERFACE s_axilite port=out      bundle=control
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // ========================================================================
    // On-Chip Memory Allocation (SRAM / BRAM / URAM)
    // ========================================================================
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

    // Internal Buffers for Inter-Module Communication
    static int64_dt X_package[SEQ_LEN][EMBED_DIM_package];
    float scaler_X[SEQ_LEN];
    #pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=2 dim=2
    #pragma HLS BIND_STORAGE variable=X_package type=ram_2p impl=uram

    static int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK];
    float global_max[SEQ_LEN];
    static int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK];
    float out_scaler[SEQ_LEN];
    
    #pragma HLS ARRAY_PARTITION variable=URAM_Out dim=2 type=cyclic factor=4
    #pragma HLS BIND_STORAGE variable=URAM_Out type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=global_max type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=URAM_Quant_Out type=ram_2p impl=uram

    // ========================================================================
    // Execution Flow
    // ========================================================================
    
    // Step 1: Load LayerNorm Weights
    load_beta_gamma_HBM2SRAM(gamma, beta, SRAM_gamma, SRAM_beta);
    
    // Step 2: Load QKV Biases & Scales to BRAM
    load_Bias_HBM2BRAM(bq, SRAM_bq);
    load_Bias_HBM2BRAM(bk, SRAM_bk);
    load_Bias_HBM2BRAM(bv, SRAM_bv);
    load_Bias_HBM2BRAM(scale_wq, SRAM_swq);
    load_Bias_HBM2BRAM(scale_wk, SRAM_swk);
    load_Bias_HBM2BRAM(scale_wv, SRAM_swv);

    // Step 3: Compute LayerNorm and pack feature map X
    do_layernorm_dataflow(x, SRAM_gamma, SRAM_beta, scaler_X, X_package);

    // Step 4: Multi-Head Attention Computation
    head_loop:
    for (int head_idx = 0; head_idx < NUM_HEADS; head_idx++) {
        process_attention_one_head_dataflow(
            X_package,
            Wq, Wk, Wv,
            SRAM_bq, SRAM_bk, SRAM_bv,
            SRAM_swq, SRAM_swk, SRAM_swv,
            scaler_X,
            URAM_Out,
            global_max,
            head_idx
        );
    }

    // Step 5: Quantize the full FP32 Attention Output to INT8
    quantize_attention_output(
        URAM_Out,
        global_max,
        URAM_Quant_Out,
        out_scaler
    );

    // Step 6: Final Output Projection & Residual Connection (Reusing input 'x')
    run_full_output_projection(
        URAM_Quant_Out,
        Wo,
        x, // Pass 'x' directly as the residual component
        out,
        out_scaler,
        bo,
        scale_wo
    );
}
}

//  =======================================================================
//     load X
//  =======================================================================

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

//  =======================================================================
//     laod X
//  =======================================================================

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

//  =======================================================================
//     laod beta gamma
//  =======================================================================

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

//  =======================================================================
//     mean and variance
//  =======================================================================

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array[SEQ_LEN][EMBED_DIM_package],
    int seq_idx
){
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_X type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=gamma  type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=beta   type=cyclic factor=8 
    
    // Local buffer to store normalized fp32 results before quantization
    float norm_buffer[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=norm_buffer type=cyclic factor=16

    // ========================================================================
    // Pass 1: Compute Global Mean (Summation)
    // ========================================================================
    float sum = 0.0f;
    compute_mean_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){      
        // Relaxed II for stable floating-point accumulation
        #pragma HLS PIPELINE II=4 
        float local_sum = 0.0f;
        
        compute_local_sum:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            local_sum += BRAM_X[j+k];
        }
        sum += local_sum;
    }
    float global_mean = sum * EMBED_DIM_INV;

    // ========================================================================
    // Pass 2: Compute Variance (Sum of Squared Differences)
    // ========================================================================
    float sq_diff_sum = 0.0f;
    compute_var_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        // Relaxed II for stable floating-point accumulation
        #pragma HLS PIPELINE II=4 
        float local_sq_diff_sum = 0.0f;
        
        compute_local_sq_diff:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            float diff = BRAM_X[j+k] - global_mean;
            local_sq_diff_sum += diff * diff;
        }
        sq_diff_sum += local_sq_diff_sum;
    }
    
    // Variance and Inverse Standard Deviation
    float var = sq_diff_sum * EMBED_DIM_INV; 
    // Extra safety guard against extreme precision loss giving negative variance
    float safe_var = std::max(0.0f, var); 
    float inv_std_out = 1.0f / std::sqrt(safe_var + EPSILON); 

    // ========================================================================
    // Pass 3: Normalize and Find Max Absolute Value
    // ========================================================================
    float max_val = 0.0f;
    compute_norm_and_find_max_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        // Set II=2 because gamma and beta are only partitioned by 8, 
        // but we are trying to read 16 elements per iteration.
        #pragma HLS PIPELINE II=2 
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
    // Use 127.0f instead of 127.5f to prevent integer clipping at the maximum value bounds
    float scale = 127.5f / max_val;
    
    // ========================================================================
    // Pass 4: Quantize and Pack
    // ========================================================================
    quant_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE II=1
        
        int64_dt chunk0 = 0; 
        int64_dt chunk1 = 0;

        pack_loop:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            int idx = j + k;

            // Safe Quantization Logic
            float scaled_val = norm_buffer[idx] * scale;
            int32_t q_int = static_cast<int32_t>(scaled_val >= 0 ? scaled_val + 0.5f : scaled_val - 0.5f);
            int8_dt q_val = static_cast<int8_dt>(std::max(-128, std::min(127, q_int)));
            
            int bit_offset = (k % 8) * 8; 
            
            if (k < 8) {
                chunk0.range(bit_offset + 7, bit_offset) = q_val;
            } else {
                chunk1.range(bit_offset + 7, bit_offset) = q_val;
            }
        }
        
        // Write out 2 packed chunks (16 int8s total) to URAM
        int pkg_idx = j >> 3; 
        X_package_array[seq_idx][pkg_idx]     = chunk0;
        X_package_array[seq_idx][pkg_idx + 1] = chunk1;
    }
    scaler_array[seq_idx] = 1.0f / scale;
}

//  =======================================================================
//     layernorm
//  =======================================================================

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
        #pragma HLS LOOP_FLATTEN off
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

//  =======================================================================
//     weight FIFO -> URAM
//  =======================================================================

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

//  =======================================================================
//     bias FIFO -> URAM
//  =======================================================================

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

//  =======================================================================
//     weight FIFO -> URAM
//  =======================================================================

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

//  =======================================================================
//     linear QKV top module
//  =======================================================================

void compute_linear_qkv_one_head(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    hls::stream<uint256_dt> &FIFO_Wq,
    hls::stream<uint256_dt> &FIFO_Wk,
    hls::stream<uint256_dt> &FIFO_Wv,
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out, 
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
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            int weight_idx = iter * TILE_SIZE + j; 
            
            float f_q = (float)BRAM_Acc_Q[j] * scale_X[i] * scale_wq[weight_idx] + BRAM_bq[weight_idx];
            float f_k = (float)BRAM_Acc_K[j] * scale_X[i] * scale_wk[weight_idx] + BRAM_bk[weight_idx];
            float f_v = (float)BRAM_Acc_V[j] * scale_X[i] * scale_wv[weight_idx] + BRAM_bv[weight_idx];
            
            FIFO_Q_out.write(f_q);
            FIFO_K_out.write(f_k);
            FIFO_V_out.write(f_v);
        }
    }
}

//  =======================================================================
//     layernorm top module
//  =======================================================================

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

//  =======================================================================
//     self attention top module
//  =======================================================================

void process_attention_one_head_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *Wq, uint256_dt *Wk, uint256_dt *Wv,
    float SRAM_bq[EMBED_DIM], float SRAM_bk[EMBED_DIM], float SRAM_bv[EMBED_DIM],
    float SRAM_swq[EMBED_DIM], float SRAM_swk[EMBED_DIM], float SRAM_swv[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],
    float global_max[SEQ_LEN],
    int head_idx
) {
    #pragma HLS DATAFLOW

    hls::stream<uint256_dt> FIFO_Wq("FIFO_Wq");
    hls::stream<uint256_dt> FIFO_Wk("FIFO_Wk");
    hls::stream<uint256_dt> FIFO_Wv("FIFO_Wv");

    hls::stream<float> FIFO_Q_out("FIFO_Q_out");
    hls::stream<float> FIFO_K_out("FIFO_K_out");
    hls::stream<float> FIFO_V_out("FIFO_V_out");

    int64_dt URAM_Q[SEQ_LEN][HEAD_DIM / INT8_PACK];
    int64_dt URAM_K[SEQ_LEN][HEAD_DIM / INT8_PACK];
    int64_dt URAM_V_T[HEAD_DIM][SEQ_LEN_PADDED / INT8_PACK];
    float scale_Q[SEQ_LEN];
    float scale_K[SEQ_LEN];
    float scale_V[HEAD_DIM];

    #pragma HLS BIND_STORAGE variable=URAM_Q type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=URAM_K type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=URAM_V_T type=ram_2p impl=uram

    hls::stream<int8_dt> FIFO_Softmax_out("FIFO_Softmax_out");
    hls::stream<float> FIFO_Softmax_scaler_out("FIFO_Softmax_scaler");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wq depth=32
    #pragma HLS STREAM variable=FIFO_Wk depth=32
    #pragma HLS STREAM variable=FIFO_Wv depth=32
    #pragma HLS STREAM variable=FIFO_Q_out depth=32
    #pragma HLS STREAM variable=FIFO_K_out depth=32
    #pragma HLS STREAM variable=FIFO_V_out depth=32
    #pragma HLS STREAM variable=FIFO_Softmax_out depth=40000
    #pragma HLS STREAM variable=FIFO_Softmax_scaler_out depth=2
#endif
    load_Weight_tile_HBM2FIFO(Wq, FIFO_Wq, head_idx);
    load_Weight_tile_HBM2FIFO(Wk, FIFO_Wk, head_idx);
    load_Weight_tile_HBM2FIFO(Wv, FIFO_Wv, head_idx);

    compute_linear_qkv_one_head(
        X_package, FIFO_Wq, FIFO_Wk, FIFO_Wv,
        FIFO_Q_out, FIFO_K_out, FIFO_V_out,
        SRAM_bq, SRAM_bk, SRAM_bv,
        SRAM_swq, SRAM_swk, SRAM_swv,
        scaler_X, head_idx
    );

    qkv_sram_and_quantize(
        FIFO_Q_out, FIFO_K_out, FIFO_V_out,
        URAM_Q, URAM_K, URAM_V_T,
        scale_Q, scale_K, scale_V
    );

    compute_score_and_softmax(
        URAM_Q, URAM_K, scale_Q, scale_K,
        FIFO_Softmax_out, FIFO_Softmax_scaler_out
    );

    compute_attention_output_one_head(
        FIFO_Softmax_out, FIFO_Softmax_scaler_out,
        URAM_V_T, scale_V,
        URAM_Out, global_max, head_idx
    );
}

//  =======================================================================
//     quantize QKV
//  =======================================================================

void qkv_sram_and_quantize(
    hls::stream<float> &FIFO_Q_out, 
    hls::stream<float> &FIFO_K_out, 
    hls::stream<float> &FIFO_V_out, 
    int64_dt URAM_Q[SEQ_LEN][HEAD_DIM / INT8_PACK],         // [197][8]
    int64_dt URAM_K[SEQ_LEN][HEAD_DIM / INT8_PACK],         // [197][8]
    int64_dt URAM_V_T[HEAD_DIM][SEQ_LEN_PADDED / INT8_PACK], // [64][25]
    float scaler_Q[SEQ_LEN],            
    float scaler_K[SEQ_LEN],            
    float scaler_V[HEAD_DIM]            
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off
    float Buffer_V[SEQ_LEN][HEAD_DIM];
    #pragma HLS ARRAY_PARTITION variable=Buffer_V type=cyclic factor=8 dim=2
    #pragma HLS BIND_STORAGE variable=Buffer_V type=ram_2p impl=bram

    // === Step 1: Process Q, K quantization and Buffer V ===
    process_qk_and_buffer_v_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off 
        
        float local_q[HEAD_DIM];
        float local_k[HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=local_q type=complete
        #pragma HLS ARRAY_PARTITION variable=local_k type=complete
        
        float max_q = 0.0f;
        float max_k = 0.0f;

        read_and_find_max:
        for (int j = 0; j < HEAD_DIM; j++) {
            #pragma HLS PIPELINE II=1
            float q_val = FIFO_Q_out.read();
            float k_val = FIFO_K_out.read();
            float v_val = FIFO_V_out.read();

            local_q[j] = q_val;
            local_k[j] = k_val;
            Buffer_V[i][j] = v_val;

            max_q = std::max(max_q, std::abs(q_val));
            max_k = std::max(max_k, std::abs(k_val));
        }

        max_q = std::max(max_q, EPSILON);
        max_k = std::max(max_k, EPSILON);
        float scale_q_val = 127.5f / max_q;
        float scale_k_val = 127.5f / max_k;

        pack_and_write_qk_loop:
        for (int j = 0; j < HEAD_DIM / INT8_PACK; j++) {
            #pragma HLS PIPELINE II=1
            int64_dt pack_q = 0;
            int64_dt pack_k = 0;

            for (int k = 0; k < INT8_PACK; k++) {
                #pragma HLS UNROLL
                int idx = j * INT8_PACK + k;
                
                // Safe Quantization Logic
                float scaled_q = local_q[idx] * scale_q_val;
                float scaled_k = local_k[idx] * scale_k_val;
                int32_t q_int = static_cast<int32_t>(scaled_q >= 0 ? scaled_q + 0.5f : scaled_q - 0.5f);
                int32_t k_int = static_cast<int32_t>(scaled_k >= 0 ? scaled_k + 0.5f : scaled_k - 0.5f);
                
                int8_dt q_val = static_cast<int8_dt>(std::max(-128, std::min(127, q_int)));
                int8_dt k_val = static_cast<int8_dt>(std::max(-128, std::min(127, k_int)));
                
                int start_bit = k * 8;
                int end_bit = start_bit + 7;
                
                pack_q.range(end_bit, start_bit) = q_val;
                pack_k.range(end_bit, start_bit) = k_val;
            }
            URAM_Q[i][j] = pack_q;
            URAM_K[i][j] = pack_k;
        }
        scaler_Q[i] = 1.0f / scale_q_val;
        scaler_K[i] = 1.0f / scale_k_val;
    }

    // === Step 2: V transpose, padding, and packing to URAM ===
    transpose_pad_and_pack_v_loop:
    for (int j = 0; j < HEAD_DIM; j++) {
        #pragma HLS LOOP_FLATTEN off
        
        float max_v = 0.0f;
        float local_v_col[SEQ_LEN];
        
        find_v_col_max_loop:
        for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1
            float val = Buffer_V[i][j];
            local_v_col[i] = val;
            max_v = std::max(max_v, std::abs(val));
        }

        max_v = std::max(max_v, EPSILON);
        float scale_v_val = 127.5f / max_v;
        
        write_v_transposed_packed_loop:
        for (int i = 0; i < SEQ_LEN_PADDED / INT8_PACK; i++) {
            #pragma HLS PIPELINE II=1
            int64_dt pack_v = 0;

            for (int k = 0; k < INT8_PACK; k++) {
                #pragma HLS UNROLL
                int orig_idx = i * INT8_PACK + k;
                int8_dt v_val = 0;
                if (orig_idx < SEQ_LEN) {
                    // Safe Quantization Logic
                    float scaled_v = local_v_col[orig_idx] * scale_v_val;
                    int32_t v_int = static_cast<int32_t>(scaled_v >= 0 ? scaled_v + 0.5f : scaled_v - 0.5f);
                    v_val = static_cast<int8_dt>(std::max(-128, std::min(127, v_int)));
                }

                int start_bit = k * 8;
                int end_bit = start_bit + 7;
                pack_v.range(end_bit, start_bit) = v_val;
            }
            URAM_V_T[j][i] = pack_v;
        }
        scaler_V[j] = 1.0f / scale_v_val;
    }
}
//  =======================================================================
//     score and softmax
//  =======================================================================

void compute_score_and_softmax(
    int64_dt URAM_Q[SEQ_LEN][HEAD_DIM / INT8_PACK],         // [197][8]
    int64_dt URAM_K[SEQ_LEN][HEAD_DIM / INT8_PACK],         // [197][8]
    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],
    hls::stream<int8_dt> &FIFO_Softmax_out,
    hls::stream<float> &FIFO_Softmax_scaler_out
) {
    #pragma HLS ARRAY_PARTITION variable=URAM_Q dim=2 type=cyclic factor=4
    #pragma HLS ARRAY_PARTITION variable=URAM_K dim=2 type=cyclic factor=4
    #pragma HLS INLINE off
    
    row_score_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        
        float sq_i = scale_Q[i];
        
        float score_buffer[SEQ_LEN_PADDED];
        float exp_buffer[SEQ_LEN_PADDED];
        #pragma HLS BIND_STORAGE variable=score_buffer type=ram_2p impl=lutram
        #pragma HLS BIND_STORAGE variable=exp_buffer type=ram_2p impl=lutram
        
        float row_max = NEG_INF;
        pass1_dot_max:
        for (int k = 0; k < SEQ_LEN_PADDED; k++) {
            #pragma HLS PIPELINE II=1
            float raw_score = NEG_INF;

            if (k < SEQ_LEN) {
                int32_dt acc = 0;
                
                dot_product_loop:
                for (int c = 0; c < HEAD_DIM / INT8_PACK; c += 4) {
                    #pragma HLS UNROLL
                    int32_dt cycle_acc = 0;

                    read_unpack_loop:
                    for (int p = 0; p < 4; p++) {
                        #pragma HLS UNROLL
                        int idx = c + p;
                        int64_dt q_pack = URAM_Q[i][idx];
                        int64_dt k_pack = URAM_K[k][idx];

                        mac_unroll:
                        for (int r = 0; r < INT8_PACK; r++) {
                            #pragma HLS UNROLL
                            int start_bit = r * 8;
                            int end_bit = start_bit + 7;
                            int8_dt q_val = (int8_dt)q_pack.range(end_bit, start_bit);
                            int8_dt k_val = (int8_dt)k_pack.range(end_bit, start_bit);
                            cycle_acc += q_val * k_val;
                        }
                    }
                    acc += cycle_acc;
                }
                raw_score = static_cast<float>(acc) * 0.125f * sq_i * scale_K[k];
            }

            score_buffer[k] = raw_score;
            if (raw_score > row_max) {
                row_max = raw_score;
            }
        }

        float exp_sum = 0.0f;
        pass2_exp_sum:
        for (int k = 0; k < SEQ_LEN_PADDED; k++) {
            #pragma HLS PIPELINE II=1 
            
            float score = score_buffer[k];
            float exp_val = 0.0f;
            
            if (k < SEQ_LEN) {
                exp_val = hls::exp(score - row_max);
            }
            
            exp_buffer[k] = exp_val;
            exp_sum += exp_val;
        }

        float safe_exp_sum = std::max(exp_sum, 1e-12f);
        float inv_sum = 1.0f / safe_exp_sum;
        float row_scaler = inv_sum / 127.5f; 
        FIFO_Softmax_scaler_out.write(row_scaler);
        
        pass3_quantize_write:
        for (int k = 0; k < SEQ_LEN_PADDED; k++) {
            #pragma HLS PIPELINE II=1
            
            float exp_val = exp_buffer[k];
            int8_dt q_softmax = 0;
            
            if (k < SEQ_LEN) {
                // Safe Quantization Logic for Softmax (positive values)
                int32_t s_int = static_cast<int32_t>(exp_val * 127.5f + 0.5f);
                q_softmax = static_cast<int8_dt>(std::max(0, std::min(127, s_int)));
            }
            
            FIFO_Softmax_out.write(q_softmax);
        }
    }
}

//  =======================================================================
//     self attention
//  =======================================================================

void compute_attention_output_one_head(
    hls::stream<int8_dt> &FIFO_Softmax_out,
    hls::stream<float> &FIFO_Softmax_scaler_out,
    int64_dt URAM_V_T[HEAD_DIM][SEQ_LEN_PADDED / INT8_PACK], // [64][25]
    float scale_V[HEAD_DIM],
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],
    float global_max[SEQ_LEN],
    int head_idx
) {
    #pragma HLS ARRAY_PARTITION variable=URAM_V_T dim=2 type=cyclic factor=4
    #pragma HLS BIND_STORAGE variable=global_max type=ram_2p impl=bram
    #pragma HLS INLINE off

    row_compute_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        float softmax_scaler = FIFO_Softmax_scaler_out.read();

        int8_dt line_buffer[SEQ_LEN_PADDED];
        #pragma HLS ARRAY_PARTITION variable=line_buffer type=cyclic factor=32
        
        read_softmax_loop:
        for (int k = 0; k < SEQ_LEN_PADDED; k++) {
            #pragma HLS PIPELINE II=1
            line_buffer[k] = FIFO_Softmax_out.read();
        }

        float local_max = NEG_INF;
        float out_row_buffer[HEAD_DIM];
        #pragma HLS ARRAY_PARTITION variable=out_row_buffer type=cyclic factor=2

        compute_v_loop:
        for (int c = 0; c < HEAD_DIM; c++) {
            #pragma HLS LOOP_FLATTEN off
            int32_dt acc = 0;

            dot_product_step:
            for (int step = 0; step < 7; step++) {
                #pragma HLS PIPELINE II=1
                int32_dt cycle_acc = 0;
                int pack_count = (step == 6) ? 1 : 4;

                mac_pack_loop:
                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    if (p < pack_count) {
                        int pack_idx = step * 4 + p;
                        int64_dt v_pack = URAM_V_T[c][pack_idx];

                        mac_element_loop:
                        for (int r = 0; r < INT8_PACK; r++) {
                            #pragma HLS UNROLL
                            int k_idx = pack_idx * INT8_PACK + r;
                            int8_dt s_val = line_buffer[k_idx];
                            int8_dt v_val = (int8_dt)v_pack.range(r * 8 + 7, r * 8);
                            cycle_acc += s_val * v_val;
                        }
                    }
                }
                acc += cycle_acc;
            }

            float out_val = (float)acc * softmax_scaler * scale_V[c];
            out_row_buffer[c] = out_val;
            local_max = std::max(local_max, std::abs(out_val));
        }

        if (head_idx == 0) {
            global_max[i] = local_max;
        } else {
            global_max[i] = std::max(global_max[i], local_max);
        }

        write_out_loop:
        for (int c = 0; c < HEAD_DIM / FP32_PACK; c++) {
            #pragma HLS PIPELINE II=1
            
            union FloatInt { float f; uint32_t i; } conv0, conv1;
            
            conv0.f = out_row_buffer[c * 2];
            conv1.f = out_row_buffer[c * 2 + 1];
            
            int64_dt pack_out = 0;
            pack_out.range(31, 0)  = conv0.i;
            pack_out.range(63, 32) = conv1.i;
            
            int target_c = head_idx * (HEAD_DIM / FP32_PACK) + c;
            URAM_Out[i][target_c] = pack_out;
        }
    }
}

//  =======================================================================
//     quantize_attention_output
//  =======================================================================

void quantize_attention_output(
    int64_dt URAM_Out[SEQ_LEN][EMBED_DIM / FP32_PACK],        
    float global_max[SEQ_LEN],                               
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK], 
    float out_scaler[SEQ_LEN]                                
) {
    #pragma HLS ARRAY_PARTITION variable=URAM_Out dim=2 type=cyclic factor=4
    #pragma HLS BIND_STORAGE variable=URAM_Quant_Out type=ram_2p impl=uram
    #pragma HLS INLINE off

    row_quant_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        float scale;

        col_quant_loop:
        for (int j = 0; j < EMBED_DIM / INT8_PACK; j++) {
            #pragma HLS PIPELINE II=1
            
            if (j == 0) {
                float max_val = std::max(global_max[i], EPSILON);
                scale = 127.5f / max_val;
                out_scaler[i] = max_val / 127.5f; 
            }
            
            int64_dt packed_int8 = 0;

            unpack_and_quantize_loop:
            for (int p = 0; p < 4; p++) {
                #pragma HLS UNROLL
                
                int read_idx = j * 4 + p;
                int64_dt fp32_pack = URAM_Out[i][read_idx];

                union { float f; uint32_t i; } conv0, conv1;
                conv0.i = fp32_pack.range(31, 0);
                conv1.i = fp32_pack.range(63, 32);

                // Safe Quantization Logic
                float scaled_out0 = conv0.f * scale;
                float scaled_out1 = conv1.f * scale;
                
                int32_t out0_int = static_cast<int32_t>(scaled_out0 >= 0 ? scaled_out0 + 0.5f : scaled_out0 - 0.5f);
                int32_t out1_int = static_cast<int32_t>(scaled_out1 >= 0 ? scaled_out1 + 0.5f : scaled_out1 - 0.5f);
                
                int8_dt q0 = static_cast<int8_dt>(std::max(-128, std::min(127, out0_int)));
                int8_dt q1 = static_cast<int8_dt>(std::max(-128, std::min(127, out1_int)));

                int start_bit_0 = (p * 2) * 8;
                int start_bit_1 = (p * 2 + 1) * 8;
                
                packed_int8.range(start_bit_0 + 7, start_bit_0) = q0;
                packed_int8.range(start_bit_1 + 7, start_bit_1) = q1;
            }
            
            URAM_Quant_Out[i][j] = packed_int8;
        }
    }
}

//  =======================================================================
//      Load residual (X in FP32)
//  =======================================================================

void load_Residual_Tile_HBM2FIFO(
    uint256_dt *HBM_X,         
    hls::stream<uint256_dt> &FIFO_Res,
    int iter                 
) {
    int chunks_per_row = EMBED_DIM / 8; // 256-bit contains 8 floats
    int chunks_per_tile = TILE_SIZE / 8;
    int tile_col_offset = iter * chunks_per_tile;

    load_res_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start_idx = i * chunks_per_row;
        
        load_res_tile_loop:
        for (int j = 0; j < chunks_per_tile; j++) {
            #pragma HLS PIPELINE II=1
            int target_idx = row_start_idx + tile_col_offset + j;
            FIFO_Res.write(HBM_X[target_idx]);
        }
    }
}

//  =======================================================================
//      output projection result
//  =======================================================================

void store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint256_dt *HBM_Out,
    int iter
) {
    int chunks_per_row = EMBED_DIM / 8; 
    int chunks_per_tile = TILE_SIZE / 8; 
    int tile_col_offset = iter * chunks_per_tile;

    store_res_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start_idx = i * chunks_per_row;
        
        store_res_tile_loop:
        for (int j = 0; j < chunks_per_tile; j++) {
            #pragma HLS PIPELINE II=1
            int target_idx = row_start_idx + tile_col_offset + j;
            HBM_Out[target_idx] = FIFO_Final_Out.read();
        }
    }
}

//  =======================================================================
//      output projection and residual compute
//  =======================================================================

void compute_output_proj_and_residual(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK], 
    hls::stream<uint256_dt> &FIFO_Wo,                        
    hls::stream<uint256_dt> &FIFO_Res,                       
    hls::stream<uint256_dt> &FIFO_Final_Out,                 
    float out_scaler[SEQ_LEN],                               
    float BRAM_bo[EMBED_DIM],                                
    float scale_wo[EMBED_DIM],                               
    int iter
) {
    #pragma HLS INLINE off

    #ifdef __SYNTHESIS__
    int64_dt URAM_Wo[TILE_SIZE][EMBED_DIM / INT8_PACK];
    #pragma HLS BIND_STORAGE variable=URAM_Wo type=ram_2p impl=uram
    #else
    static int64_dt URAM_Wo[TILE_SIZE][EMBED_DIM / INT8_PACK];
    #endif

    #pragma HLS ARRAY_PARTITION variable=URAM_Wo type=cyclic factor=4 dim=2

    // load weight
    load_wo_row:
    for (int i = 0; i < TILE_SIZE; i++) { 
        load_wo_col:
        for (int j = 0; j < EMBED_DIM / 32; j++) { 
            #pragma HLS PIPELINE II=4
            uint256_dt wo_chunk = FIFO_Wo.read();
            
            unpack_wo_loop:
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 64;
                int end_bit = start_bit + 63;
                URAM_Wo[i][j * 4 + k] = wo_chunk.range(end_bit, start_bit);
            }
        }
    }

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        int32_dt acc_buffer[TILE_SIZE];
        #pragma HLS BIND_STORAGE variable=acc_buffer type=ram_2p impl=lutram

        // ====================================================================
        // MAC
        // ====================================================================
        int32_dt acc = 0;
        
        compute_tile:
        for (int j = 0; j < TILE_SIZE; j++) { 
            dot_product_step:
            for (int step = 0; step < EMBED_DIM / 32; step++) {
                #pragma HLS PIPELINE II=1
                int32_dt cycle_acc = 0;

                mac_pack_loop:
                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL
                    int pack_idx = step * 4 + p;
                    int64_dt in_pack = URAM_Quant_Out[i][pack_idx];
                    int64_dt wo_pack = URAM_Wo[j][pack_idx];

                    mac_element_loop:
                    for (int r = 0; r < INT8_PACK; r++) {
                        #pragma HLS UNROLL
                        int start_bit = r * 8;
                        int end_bit = start_bit + 7;
                        int8_dt in_val = (int8_dt)in_pack.range(end_bit, start_bit);
                        int8_dt wo_val = (int8_dt)wo_pack.range(end_bit, start_bit);
                        cycle_acc += in_val * wo_val;
                    }
                }

                if (step == 0) acc = cycle_acc;
                else acc += cycle_acc;

                if (step == (EMBED_DIM / 32) - 1) {
                    acc_buffer[j] = acc;
                }
            }
        }

        // ====================================================================
        // Dequantize, Residual Addition, and Pack
        // ====================================================================
        float token_scaler = out_scaler[i];
        uint256_dt res_chunk = 0;
        
        float pack_array[8]; // Pack into 8 floats (256-bit)
        #pragma HLS ARRAY_PARTITION variable=pack_array type=complete

        post_process_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            
            // Read new residual 256-bit chunk every 8 elements
            if (j % 8 == 0) {
                res_chunk = FIFO_Res.read();
            }

            int32_dt final_acc = acc_buffer[j];
            int weight_idx = iter * TILE_SIZE + j;
            
            float w_scale = scale_wo[weight_idx];
            float w_bias  = BRAM_bo[weight_idx];
            float combined_scale = token_scaler * w_scale; 
            
            float proj_val = (float)final_acc * combined_scale + w_bias;

            // Extract the matching FP32 value from the residual chunk
            int res_start_bit = (j % 8) * 32;
            int res_end_bit = res_start_bit + 31;
            union { float f; uint32_t i; } res_conv;
            res_conv.i = res_chunk.range(res_end_bit, res_start_bit);
            
            // Add residual
            pack_array[j % 8] = proj_val + res_conv.f;

            // Write out 256-bit chunk every 8 elements
            if ((j % 8 == 7) || (j == TILE_SIZE - 1)) {
                uint256_dt out_chunk = 0;
                pack_write_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL
                    union { float f; uint32_t i; } out_conv;
                    out_conv.f = pack_array[k];
                    int out_start = k * 32;
                    int out_end = out_start + 31;
                    out_chunk.range(out_end, out_start) = out_conv.i;
                }
                FIFO_Final_Out.write(out_chunk);
            }
        }
    }
}

//  =======================================================================
//     output projection single tile TOP module
//  =======================================================================

void process_o_proj_single_tile_dataflow(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK], 
    uint256_dt *HBM_Wo,
    uint256_dt *HBM_X_Res,       
    uint256_dt *HBM_Out,         
    float out_scaler[SEQ_LEN],   
    float BRAM_bo[EMBED_DIM],    
    float scale_wo[EMBED_DIM],   
    int iter                     
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint256_dt> FIFO_Wo("FIFO_Wo");
    hls::stream<uint256_dt> FIFO_Res("FIFO_Res");
    hls::stream<uint256_dt> FIFO_Final_Out("FIFO_Final_Out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wo depth=32
    #pragma HLS STREAM variable=FIFO_Res depth=32
    #pragma HLS STREAM variable=FIFO_Final_Out depth=32
#endif
    
    load_Weight_tile_HBM2FIFO(HBM_Wo, FIFO_Wo, iter);

    load_Residual_Tile_HBM2FIFO(HBM_X_Res, FIFO_Res, iter);

    compute_output_proj_and_residual(
        URAM_Quant_Out, 
        FIFO_Wo, 
        FIFO_Res, 
        FIFO_Final_Out, 
        out_scaler, 
        BRAM_bo, 
        scale_wo, 
        iter
    );

    store_Final_Tile_FIFO2HBM(FIFO_Final_Out, HBM_Out, iter);
}

//  =======================================================================
//     output projection TOP module
//  =======================================================================

void run_full_output_projection(
    int64_dt URAM_Quant_Out[SEQ_LEN][EMBED_DIM / INT8_PACK], 
    uint256_dt *HBM_Wo,          
    uint256_dt *HBM_X_Res,      
    uint256_dt *HBM_Out,         
    float out_scaler[SEQ_LEN],   
    float *HBM_bo,
    float *HBM_scale_wo
) {
    float SRAM_bo[EMBED_DIM];
    float SRAM_scale_wo[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=SRAM_bo type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_wo type=cyclic factor=8
    #pragma HLS BIND_STORAGE variable=SRAM_bo type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SRAM_scale_wo type=ram_2p impl=bram

    load_Bias_HBM2BRAM(HBM_bo, SRAM_bo);
    load_Bias_HBM2BRAM(HBM_scale_wo, SRAM_scale_wo);

    const int NUM_TILES = EMBED_DIM / TILE_SIZE; // 768 / 64 = 12

    full_proj_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        process_o_proj_single_tile_dataflow(
            URAM_Quant_Out,
            HBM_Wo,
            HBM_X_Res,
            HBM_Out,
            out_scaler,
            SRAM_bo,
            SRAM_scale_wo,
            iter
        );
    }
}