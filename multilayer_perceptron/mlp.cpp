#include "mlp.h"
extern "C" {
void mlp(
    // Input Feature Map (197 * 768)
    uint256_dt *x,
    
    // LayerNorm Weights
    uint256_dt *gamma,
    uint256_dt *beta,
    
    // MLP Layer 1 & 2 Weights
    uint256_dt *W1,
    uint256_dt *W2,
    
    // Biases
    float *b1,
    float *b2,
    
    // Scales
    float *scale_w1,
    float *scale_w2,
    
    // Final Output (197 * 768)
    uint256_dt *out,
    uint256_dt *HBM_FP32_Intermediate
) {
    // ========================================================================
    // AXI Master Interfaces
    // ========================================================================
    #pragma HLS INTERFACE m_axi port=x        offset=slave bundle=gmem_x     depth=18912 max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=gamma    offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    #pragma HLS INTERFACE m_axi port=beta     offset=slave bundle=gmem_x     depth=96    max_widen_bitwidth=256
    
    // b1/scale_w1 length = 3072, b2/scale_w2 length = 768
    #pragma HLS INTERFACE m_axi port=b1       offset=slave bundle=gmem_bias  depth=3072
    #pragma HLS INTERFACE m_axi port=b2       offset=slave bundle=gmem_bias  depth=768
    #pragma HLS INTERFACE m_axi port=scale_w1 offset=slave bundle=gmem_bias  depth=3072
    #pragma HLS INTERFACE m_axi port=scale_w2 offset=slave bundle=gmem_bias  depth=768

    // W1 size: 3072 * 768 / 32 = 73728
    #pragma HLS INTERFACE m_axi port=W1       offset=slave bundle=gmem_w1    depth=73728 max_widen_bitwidth=256
    // W2 size: 768 * 3072 / 32 = 73728
    #pragma HLS INTERFACE m_axi port=W2       offset=slave bundle=gmem_w2    depth=73728 max_widen_bitwidth=256
    
    #pragma HLS INTERFACE m_axi port=out      offset=slave bundle=gmem_o     depth=18912 max_widen_bitwidth=256

    // NEW: AXI Master for the intermediate FP32 HBM buffer
    #pragma HLS INTERFACE m_axi port=HBM_FP32_Intermediate offset=slave bundle=gmem_inter depth=75648 max_widen_bitwidth=256

    // ========================================================================
    // AXI-Lite Control Register Interfaces
    // ========================================================================
    #pragma HLS INTERFACE s_axilite port=x        bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma    bundle=control
    #pragma HLS INTERFACE s_axilite port=beta     bundle=control
    #pragma HLS INTERFACE s_axilite port=W1       bundle=control
    #pragma HLS INTERFACE s_axilite port=W2       bundle=control
    #pragma HLS INTERFACE s_axilite port=b1       bundle=control
    #pragma HLS INTERFACE s_axilite port=b2       bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_w1 bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_w2 bundle=control
    #pragma HLS INTERFACE s_axilite port=out      bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_FP32_Intermediate bundle=control // NEW
    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // ========================================================================
    // On-Chip Memory Allocation (SRAM / BRAM / URAM)
    // ========================================================================
    
    // LayerNorm buffers
    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8
    
    static int64_dt X_package[SEQ_LEN][EMBED_DIM_package];
    float scaler_X[SEQ_LEN];
    // Factor=16 ensures 128 MACs/cycle throughput in MLP1
    #pragma HLS ARRAY_PARTITION variable=X_package type=cyclic factor=16 dim=2
    #pragma HLS BIND_STORAGE variable=X_package type=ram_2p impl=uram

    // global_max buffer (Kept in BRAM since it's only 197 floats)
    float global_max[SEQ_LEN];
    #pragma HLS BIND_STORAGE variable=global_max type=ram_2p impl=bram

    // MLP1 Quantized Output buffer
    static int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package];
    float scaler_MLP1[SEQ_LEN];
    // Factor=16 ensures 128 MACs/cycle throughput in MLP2
    #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out dim=2 type=cyclic factor=16
    #pragma HLS BIND_STORAGE variable=URAM_MLP1_Quant_Out type=ram_2p impl=uram
    #pragma HLS BIND_STORAGE variable=scaler_MLP1 type=ram_2p impl=bram

    // ========================================================================
    // Execution Flow 
    // ========================================================================
    
    // Step 1: Load LayerNorm Weights
    load_beta_gamma_HBM2SRAM(gamma, beta, SRAM_gamma, SRAM_beta);
    
    // Step 2: Compute LayerNorm and pack feature map X
    do_layernorm_dataflow(x, SRAM_gamma, SRAM_beta, scaler_X, X_package);

    // ========================================================================
    // Phase 1: Execute MLP1, GELU, and Burst Write to HBM
    // ========================================================================
    // Step 3 & 4: MLP1 compute, GELU activation, and pack to HBM
    run_mlp1_and_gelu_dataflow(
        X_package, 
        W1, 
        b1, 
        scale_w1, 
        scaler_X, 
        HBM_FP32_Intermediate, // Replaced URAM array with HBM pointer
        global_max
    );

    // ========================================================================
    // Phase 2: Burst Read from HBM and Dynamic Quantization
    // ========================================================================
    // Step 5: Load FP32 from HBM and quantize to INT8 for MLP2
    run_quantize_mlp1_dataflow(
        HBM_FP32_Intermediate, 
        global_max, 
        URAM_MLP1_Quant_Out, 
        scaler_MLP1
    );

    // Step 6: MLP Layer 2 (Projection, Dequantization & Residual Connection)
    run_full_mlp_layer2(
        URAM_MLP1_Quant_Out, 
        W2, 
        x, // Pass 'x' explicitly for the residual component
        out, 
        scaler_MLP1, 
        b2, 
        scale_w2
    );
}
}

// =======================================================================
//   Helper Functions for Memory Loading
// =======================================================================

// Load input X from HBM to FIFO (256-bit chunks)
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

// Load input X from FIFO to on-chip BRAM and unpack to FP32
void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM])
{
   load_X_col:
   for (int j = 0; j < EMBED_DIM / 8; j++) {
       #pragma HLS PIPELINE II=1
       
       uint256_dt temp_chunk = FIFO_X.read();
       
       unpack_write_loop:
       for(int k = 0; k < 8; k++) {
           #pragma HLS UNROLL 
           int start_bit = k * 32;
           int end_bit = start_bit + 31;
           
           uint32_t raw_bits = temp_chunk.range(end_bit, start_bit);
           
           union { uint32_t i_val; float f_val; } converter;
           
           converter.i_val = raw_bits;
           BRAM_X[j * 8 + k] = converter.f_val; 
       }
   }
}

// Load LayerNorm weights (gamma and beta) from HBM to SRAM
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
            SRAM_gamma[i * 8 + k] = conv_gamma.f_val;
            
            uint32_t raw_beta_bits = temp_beta_chunk.range(end_bit, start_bit);
            union { uint32_t i_val; float f_val; } conv_beta;
            conv_beta.i_val = raw_beta_bits;
            SRAM_beta[i * 8 + k] = conv_beta.f_val;
        }
    }
}

// Load FP32 Bias or Scale arrays of size MLP_DIM (3072) from HBM to BRAM
void load_Bias_HBM2BRAM_MLP(
    float *HBM_b,
    float BRAM_b[MLP_DIM]
) {
    load_b_mlp_loop:
    for (int i = 0; i < MLP_DIM; i++) {
        #pragma HLS PIPELINE II=1
        BRAM_b[i] = HBM_b[i];
    }
}

// Load FP32 Bias or Scale arrays of size EMBED_DIM (768) from HBM to BRAM
void load_Bias_HBM2BRAM(
    float *HBM_b,
    float BRAM_b[EMBED_DIM]
) {
    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1
        BRAM_b[i] = HBM_b[i];
    }
}

// Load W1 Tile from HBM to FIFO
void load_W1_tile_HBM2FIFO(
    uint256_dt *HBM_W1,
    hls::stream<uint256_dt> &FIFO_W1,
    int iter
) {
    // W1 is pre-transposed externally to [3072][768]
    // 256-bit contains 32 bytes (8-bit elements). 768 / 32 = 24 chunks per row.
    int row_offset = iter * TILE_OUT_DIM; 
    int chunks_per_row = EMBED_DIM / 32; 
    int base_addr_idx = row_offset * chunks_per_row; 

    load_w1_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) {
        int current_row_start = base_addr_idx + (i * chunks_per_row);

        load_w1_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = HBM_W1[current_row_start + j];
            FIFO_W1.write(temp_chunk);
        }
    }
}

// =======================================================================
//   LayerNorm Compute Core (2-Pass / 2N Algorithm)
// =======================================================================

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
    #pragma HLS ARRAY_PARTITION variable=gamma type=cyclic factor= 8
    #pragma HLS ARRAY_PARTITION variable=beta  type=cyclic factor= 8 

    float norm_buffer[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=norm_buffer type=cyclic factor=16
    
    // -------------------------------------------------------------------
    // Pass 1: Compute Global Mean
    // -------------------------------------------------------------------
    float sum = 0.0f;
    
    compute_mean_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){     
        #pragma HLS PIPELINE
        float local_sum = 0.0f;
        
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            local_sum += BRAM_X[j+k];
        }
        sum += local_sum;
    }
    
    float global_mean = sum * EMBED_DIM_INV; 

    // -------------------------------------------------------------------
    // Pass 2: Compute Global Variance
    // -------------------------------------------------------------------
    float variance_sum = 0.0f;
    
    compute_variance_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE
        float local_var_sum = 0.0f;
        
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            float diff = BRAM_X[j+k] - global_mean;
            local_var_sum += diff * diff;
        }
        variance_sum += local_var_sum;
    }
    
    float var = variance_sum * EMBED_DIM_INV; 
    float inv_std_out = 1.0f / std::sqrt(var + EPSILON); 
    float max_val = 0.0f;
    
    // -------------------------------------------------------------------
    // Pass 3: Normalize and Find Maximum Absolute Value
    // -------------------------------------------------------------------
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
    
    // -------------------------------------------------------------------
    // Pass 4: Quantize to int8 and Pack into int64_dt
    // -------------------------------------------------------------------
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
            float clamped_val = std::max(-128.0f, std::min(127.0f, scaled_val));
            int8_dt q_val = static_cast<int8_dt>(clamped_val >= 0 ? clamped_val + 0.5f : clamped_val - 0.5f);
            
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

// =======================================================================
//   Compute Core: MLP Layer 1 (Linear + Dequantize to FP32)
// =======================================================================

void compute_linear_mlp1_tile(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package], 
    hls::stream<uint256_dt> &FIFO_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
) {
    #pragma HLS INLINE off

    #ifdef __SYNTHESIS__
    int64_dt URAM_W1[TILE_OUT_DIM][EMBED_DIM_package];
    #else
    static int64_dt URAM_W1[TILE_OUT_DIM][EMBED_DIM_package];
    #endif

    // Read 16 packs of 64-bit simultaneously (128 MACs/cycle)
    #pragma HLS ARRAY_PARTITION variable=X_package dim=2 type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=URAM_W1 dim=2 type=cyclic factor=16
    #pragma HLS BIND_STORAGE variable=URAM_W1 type=ram_2p impl=uram

    // -------------------------------------------------------------------
    // Step 1: Unpack W1 from FIFO and load into URAM
    // -------------------------------------------------------------------
    load_w1_local_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) { 
        load_w1_local_col:
        for (int j = 0; j < 24; j++) {
            #pragma HLS PIPELINE II=4
            uint256_dt w_chunk = FIFO_W1.read();
            
            unpack_w1_loop:
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 64;
                int end_bit = start_bit + 63;
                URAM_W1[i][j * 4 + k] = w_chunk.range(end_bit, start_bit);
            }
        }
    }

    // -------------------------------------------------------------------
    // Step 2: Matrix Multiplication (128 MACs per cycle) & Dequantization
    // -------------------------------------------------------------------
    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off 
        
        float token_scale_X = scale_X[i];

        compute_tile_out:
        for (int j = 0; j < TILE_OUT_DIM; j++) { 
            int32_dt acc = 0;

            dot_product_step:
            for (int step = 0; step < (EMBED_DIM_package / 16); step++) {
                #pragma HLS PIPELINE II=1
                int32_dt cycle_acc = 0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL
                    int k_pkg = step * 16 + p;
                    
                    int64_dt x_pkg = X_package[i][k_pkg];
                    int64_dt w_pkg = URAM_W1[j][k_pkg];

                    mac_element_loop:
                    for (int r = 0; r < 8; r++) {
                        #pragma HLS UNROLL
                        int start_bit = r * 8;
                        int end_bit = start_bit + 7;
                        
                        int8_dt x_val = (int8_dt)x_pkg.range(end_bit, start_bit);
                        int8_dt w_val = (int8_dt)w_pkg.range(end_bit, start_bit);
                        
                        cycle_acc += x_val * w_val;
                    }
                }
                acc += cycle_acc;
            }

            // ---------------------------------------------------------------
            // Step 3: Dequantize to FP32 and Write to FIFO
            // ---------------------------------------------------------------
            int out_idx = iter * TILE_OUT_DIM + j; 
            
            float w_scale = scale_w1[out_idx];
            float w_bias  = BRAM_b1[out_idx];
            float combined_scale = token_scale_X * w_scale; 
            
            float fp32_out = (float)acc * combined_scale + w_bias;
            
            FIFO_GELU_in.write(fp32_out);
        }
    }
}

void process_mlp1_single_tile_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint256_dt> FIFO_W1("FIFO_W1");
#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_W1 depth=32
#endif

    load_W1_tile_HBM2FIFO(HBM_W1, FIFO_W1, iter);

    compute_linear_mlp1_tile(
        X_package,
        FIFO_W1,
        FIFO_GELU_in,
        BRAM_b1,
        scale_w1,
        scale_X,
        iter
    );
}

void run_full_mlp_layer1(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in, 
    float *HBM_b1,
    float *HBM_scale_w1,
    float scale_X[SEQ_LEN]
) {
    float SRAM_b1[MLP_DIM];
    float SRAM_scale_w1[MLP_DIM];
    
    #pragma HLS ARRAY_PARTITION variable=SRAM_b1 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_w1 type=cyclic factor=8
    #pragma HLS BIND_STORAGE variable=SRAM_b1 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SRAM_scale_w1 type=ram_2p impl=bram

    load_Bias_HBM2BRAM_MLP(HBM_b1, SRAM_b1);
    load_Bias_HBM2BRAM_MLP(HBM_scale_w1, SRAM_scale_w1);

    const int NUM_TILES = MLP_DIM / TILE_OUT_DIM; 

    full_mlp1_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        process_mlp1_single_tile_dataflow(
            X_package,
            HBM_W1,
            FIFO_GELU_in,
            SRAM_b1,
            SRAM_scale_w1,
            scale_X,
            iter
        );
    }
}

// // =======================================================================
// //   Dataflow Wrapper for MLP1 & GELU
// // =======================================================================
// void run_mlp1_and_gelu_dataflow(
//     int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
//     uint256_dt *HBM_W1,
//     float *HBM_b1,
//     float *HBM_scale_w1,
//     float scale_X[SEQ_LEN],
//     int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2],
//     float global_max[SEQ_LEN]
// ) {
//     #pragma HLS DATAFLOW
    
//     hls::stream<float> FIFO_GELU_in("FIFO_GELU_in");
// #ifdef __SYNTHESIS__
//     #pragma HLS STREAM variable=FIFO_GELU_in depth=64
// #endif

//     run_full_mlp_layer1(
//         X_package, 
//         HBM_W1, 
//         FIFO_GELU_in, 
//         HBM_b1, 
//         HBM_scale_w1, 
//         scale_X
//     );

//     compute_gelu_and_buffer_fp32(
//         FIFO_GELU_in, 
//         URAM_MLP1_FP32, 
//         global_max
//     );
// }

// =======================================================================
//   Phase 1 Dataflow Wrapper: MLP1 -> GELU -> Store to HBM
// =======================================================================
void run_mlp1_and_gelu_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    uint256_dt *HBM_W1,
    float *HBM_b1,
    float *HBM_scale_w1,
    float scale_X[SEQ_LEN],
    uint256_dt *HBM_FP32_Intermediate, // Replaced URAM with HBM pointer
    float global_max[SEQ_LEN]
) {
    #pragma HLS DATAFLOW
    
    // Internal streams for Phase 1
    hls::stream<float> FIFO_GELU_in("FIFO_GELU_in");
    hls::stream<uint256_dt> FIFO_GELU_out("FIFO_GELU_out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_GELU_in depth=64
    #pragma HLS STREAM variable=FIFO_GELU_out depth=32
#endif

    // Step 3: Execute MLP Layer 1 (outputs FP32 to FIFO)
    run_full_mlp_layer1(
        X_package, 
        HBM_W1, 
        FIFO_GELU_in, 
        HBM_b1, 
        HBM_scale_w1, 
        scale_X
    );

    // Step 4.1: Compute GELU and pack into 256-bit chunks
    compute_gelu_and_write_FIFO(
        FIFO_GELU_in, 
        FIFO_GELU_out, 
        global_max
    );

    // Step 4.2: Burst write packed FP32 data to HBM
    store_FP32_FIFO2HBM(
        FIFO_GELU_out, 
        HBM_FP32_Intermediate
    );
}

// =======================================================================
//   Phase 2 Dataflow Wrapper: Load from HBM -> Dynamic Quantize
// =======================================================================
void run_quantize_mlp1_dataflow(
    uint256_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM / INT8_PACK],
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS DATAFLOW

    // Internal stream for Phase 2
    hls::stream<uint256_dt> FIFO_Quant_in("FIFO_Quant_in");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Quant_in depth=32
#endif

    // Step 5.1: Burst read packed FP32 data from HBM
    load_FP32_HBM2FIFO(
        HBM_FP32_Intermediate,
        FIFO_Quant_in
    );

    // Step 5.2: Unpack, dynamically quantize to INT8, and store in URAM
    quantize_mlp1_output_from_FIFO(
        FIFO_Quant_in,
        global_max,
        URAM_MLP1_Quant_Out,
        scaler_MLP1
    );
}

// =======================================================================
//   Hardware-Optimized GELU Approximation
// =======================================================================
// GELU(x) ≈ x * Sigmoid(1.702x)
// Sigmoid is computed using hls::exp, mapping directly to DSPs and MUXes.

inline float gelu_hw(float x) {
    #pragma HLS INLINE
    
    float z = 1.702f * x;
    float ax = std::abs(z);
    
    // Compute denominator: 1 + e^(-|z|)
    float e = hls::exp(-ax);
    float denom = 1.0f + e;
    float sig = 1.0f / denom;
    
    // Hardware MUX to handle positive and negative ranges seamlessly
    float final_sig = (x < 0.0f) ? (1.0f - sig) : sig;
    
    return x * final_sig;
}

// inline float gelu_hw(float x) {
//     #pragma HLS INLINE
    
//     const float BETA_2 = 0.254f;
//     const float C_CONST = -2.57088f; // -alpha * log2(e)
    
//     float g_x = (x <= 0.0f) ? x : (x - BETA_2);
    
//     float exp_val = hls::exp2(C_CONST * g_x);
    
//     float denom = 1.0f + exp_val;
    
//     // log_val = log2(1 + 2^(C * g(x)))
//     float log_val = hls::log2(denom);
    
//     // sig_opt = 2^(-log_val)
//     float sig_opt = hls::exp2(-log_val);
    
//     return x * sig_opt;
// }

// // =======================================================================
// //   Phase 1: GELU Compute & Packed FP32 URAM Buffering
// // =======================================================================
// void compute_gelu_and_buffer_fp32(
//     hls::stream<float> &FIFO_GELU_in,
//     int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2], 
//     float global_max[SEQ_LEN]
// ) {
//     #pragma HLS INLINE off
    
//     #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_FP32 dim=2 type=cyclic factor=2
//     #pragma HLS BIND_STORAGE variable=URAM_MLP1_FP32 type=ram_2p impl=uram
//     #pragma HLS BIND_STORAGE variable=global_max type=ram_2p impl=bram

//     tile_loop:
//     for (int iter = 0; iter < MLP1_NUM_TILES; iter++) {
//         row_loop:
//         for (int i = 0; i < SEQ_LEN; i++) {
//             float local_row_max = 0.0f;
            
//             col_loop:
//             for (int j = 0; j < TILE_OUT_DIM / 2; j++) {
//                 #pragma HLS PIPELINE II=2
                
//                 float in_val_0 = FIFO_GELU_in.read();
//                 float in_val_1 = FIFO_GELU_in.read();
                
//                 float gelu_val_0 = gelu_hw(in_val_0);
//                 float gelu_val_1 = gelu_hw(in_val_1);
                
//                 union { float f; uint32_t i; } c0, c1;
//                 c0.f = gelu_val_0;
//                 c1.f = gelu_val_1;
                
//                 int64_dt pack = 0;
//                 pack.range(31, 0) = c0.i;
//                 pack.range(63, 32) = c1.i;
                
//                 int col_idx = iter * (TILE_OUT_DIM / 2) + j;
//                 URAM_MLP1_FP32[i][col_idx] = pack;
                
//                 local_row_max = std::max({local_row_max, std::abs(gelu_val_0), std::abs(gelu_val_1)});
//             }
            
//             if (iter == 0) {
//                 global_max[i] = local_row_max;
//             } else {
//                 global_max[i] = std::max(global_max[i], local_row_max);
//             }
//         }
//     }
// }

// // =======================================================================
// //   Phase 2: Dynamic Quantization to INT8 (Post-GELU)
// // =======================================================================
// void quantize_mlp1_output(
//     int64_dt URAM_MLP1_FP32[SEQ_LEN][MLP_DIM / 2],
//     float global_max[SEQ_LEN],
//     int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM / INT8_PACK],
//     float scaler_MLP1[SEQ_LEN]
// ) {
//     #pragma HLS INLINE off
    
//     #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_FP32 dim=2 type=cyclic factor=2
//     #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out dim=2 type=cyclic factor=4
//     #pragma HLS BIND_STORAGE variable=URAM_MLP1_Quant_Out type=ram_2p impl=uram

//     row_quant_loop:
//     for (int i = 0; i < SEQ_LEN; i++) {
//         #pragma HLS LOOP_FLATTEN off
        
//         float max_val = std::max(global_max[i], EPSILON);
//         float scale = 127.0f / max_val;
//         scaler_MLP1[i] = max_val / 127.0f; 
        
//         // 3072 / 8 = 384 cycles per row (if II=1)
//         col_quant_loop:
//         for (int j = 0; j < MLP_DIM / INT8_PACK; j++) {
//             #pragma HLS PIPELINE II=2
            
//             int64_dt packed_int8 = 0;
            
//             unpack_and_quantize_loop:
//             for (int k = 0; k < 4; k++) {
//                 #pragma HLS UNROLL
                
//                 int read_idx = j * 4 + k;
//                 int64_dt fp32_pack = URAM_MLP1_FP32[i][read_idx];
                
//                 union { float f; uint32_t i; } c0, c1;
//                 c0.i = fp32_pack.range(31, 0);
//                 c1.i = fp32_pack.range(63, 32);
                
//                 int8_dt q0 = (int8_dt)std::round(std::max(-128.0f, std::min(127.0f, c0.f * scale)));
//                 int8_dt q1 = (int8_dt)std::round(std::max(-128.0f, std::min(127.0f, c1.f * scale)));
                
//                 int start_bit_0 = (k * 2) * 8;
//                 int start_bit_1 = (k * 2 + 1) * 8;
//                 packed_int8.range(start_bit_0 + 7, start_bit_0) = q0;
//                 packed_int8.range(start_bit_1 + 7, start_bit_1) = q1;
//             }
            
//             URAM_MLP1_Quant_Out[i][j] = packed_int8;
//         }
//     }
// }

// =======================================================================
//   Function 1: Compute GELU and Write to FIFO
// =======================================================================
void compute_gelu_and_write_FIFO(
    hls::stream<float> &FIFO_GELU_in,
    hls::stream<uint256_dt> &FIFO_GELU_out, 
    float global_max[SEQ_LEN]
) {
    #pragma HLS INLINE off
    #pragma HLS BIND_STORAGE variable=global_max type=ram_2p impl=bram

    tile_loop:
    for (int iter = 0; iter < MLP1_NUM_TILES; iter++) {
        row_loop:
        for (int i = 0; i < SEQ_LEN; i++) {
            float local_row_max = 0.0f;
            
            // Process 8 floats to pack into one 256-bit chunk
            col_loop:
            for (int j = 0; j < TILE_OUT_DIM / 8; j++) {
                #pragma HLS PIPELINE II=8
                
                uint256_dt pack = 0;
                float local_max_chunk = 0.0f;

                pack_loop:
                for (int k = 0; k < 8; k++) {
                    float in_val = FIFO_GELU_in.read();
                    float gelu_val = gelu_hw(in_val);
                    
                    union { float f; uint32_t i; } conv;
                    conv.f = gelu_val;
                    
                    int start_bit = k * 32;
                    int end_bit = start_bit + 31;
                    pack.range(end_bit, start_bit) = conv.i;
                    
                    local_max_chunk = std::max(local_max_chunk, std::abs(gelu_val));
                }
                FIFO_GELU_out.write(pack);
                local_row_max = std::max(local_row_max, local_max_chunk);
            }
            
            if (iter == 0) {
                global_max[i] = local_row_max;
            } else {
                global_max[i] = std::max(global_max[i], local_row_max);
            }
        }
    }
}

// =======================================================================
//   Function 2: Store FP32 Intermediate Data from FIFO to HBM
// =======================================================================
void store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint256_dt *HBM_FP32_Intermediate
) {
    #pragma HLS INLINE off
    
    // Total 256-bit chunks = (197 tokens * 3072 dimension) / 8 floats per chunk
    const int TOTAL_CHUNKS = (SEQ_LEN * MLP_DIM) / 8;
    
    store_hbm_loop:
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        HBM_FP32_Intermediate[i] = FIFO_GELU_out.read();
    }
}

// =======================================================================
//   Function 3: Load FP32 Intermediate Data from HBM to FIFO
// =======================================================================
void load_FP32_HBM2FIFO(
    uint256_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
) {
    #pragma HLS INLINE off
    
    const int TOTAL_CHUNKS = (SEQ_LEN * MLP_DIM) / 8;
    
    load_hbm_loop:
    for (int i = 0; i < TOTAL_CHUNKS; i++) {
        #pragma HLS PIPELINE II=1
        FIFO_Quant_in.write(HBM_FP32_Intermediate[i]);
    }
}

// =======================================================================
//   Function 4: Read from FIFO and perform Dynamic Quantization
// =======================================================================
void quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM / INT8_PACK], // INT8_PACK = 8
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS INLINE off
    
    // Bind memory for the output quantized URAM
    #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out dim=2 type=cyclic factor=16
    #pragma HLS BIND_STORAGE variable=URAM_MLP1_Quant_Out type=ram_2p impl=uram

    // CRITICAL FIX: Loop order MUST match Phase 1 (Tile -> Row -> Col) 
    // to correctly consume the interleaved FIFO data from HBM.
    tile_quant_loop:
    for (int iter = 0; iter < MLP1_NUM_TILES; iter++) {
        
        row_quant_loop:
        for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS LOOP_FLATTEN off
            
            // Calculate the scale for the current sequence/token
            float max_val = std::max(global_max[i], EPSILON);
            float scale = 127.0f / max_val;
            
            // Only update scaler_MLP1 once per sequence to avoid redundant writes
            if (iter == 0) {
                scaler_MLP1[i] = max_val / 127.0f; 
            }
            
            col_quant_loop:
            for (int j = 0; j < TILE_OUT_DIM / INT8_PACK; j++) {
                #pragma HLS PIPELINE II=1
                
                // Read 8 FP32 values simultaneously
                uint256_dt fp32_pack = FIFO_Quant_in.read();
                int64_dt packed_int8 = 0;
                
                unpack_and_quantize_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL
                    
                    // Extract 32-bit raw data
                    int in_start_bit = k * 32;
                    int in_end_bit = in_start_bit + 31;
                    uint32_t raw_bits = fp32_pack.range(in_end_bit, in_start_bit);
                    
                    // Convert back to float
                    union { uint32_t i; float f; } conv;
                    conv.i = raw_bits;
                    
                    // Quantize to INT8
                    float scaled_val = conv.f * scale;
                    
                    // Hardware-friendly rounding (avoids expensive std::round instantiation)
                    float clamped_val = std::max(-128.0f, std::min(127.0f, scaled_val));
                    int8_dt q_val = static_cast<int8_dt>(clamped_val >= 0 ? clamped_val + 0.5f : clamped_val - 0.5f);
                    
                    // Pack into 64-bit int64_dt
                    int out_start_bit = k * 8;
                    int out_end_bit = out_start_bit + 7;
                    packed_int8.range(out_end_bit, out_start_bit) = q_val;
                }
                
                // Calculate the global column index for the URAM
                int global_col_idx = iter * (TILE_OUT_DIM / INT8_PACK) + j;
                URAM_MLP1_Quant_Out[i][global_col_idx] = packed_int8;
            }
        }
    }
}
// =======================================================================
//   Helper Functions for MLP2 Memory Loading & Storing
// =======================================================================

// Load W2 Tile from HBM to FIFO
// W2 is pre-transposed externally to [768][3072]
void load_W2_tile_HBM2FIFO(
    uint256_dt *HBM_W2,
    hls::stream<uint256_dt> &FIFO_W2,
    int iter
) {
    int row_offset = iter * TILE_SIZE; 
    int chunks_per_row = MLP_DIM / 32; // 3072 / 32 bytes = 96 chunks (256-bit each)
    int base_addr_idx = row_offset * chunks_per_row; 

    load_w2_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start = base_addr_idx + (i * chunks_per_row);
        
        load_w2_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = HBM_W2[current_row_start + j];
            FIFO_W2.write(temp_chunk);
        }
    }
}

// Load Residual (FP32 X) Tile from HBM to FIFO
void load_Residual_Tile_HBM2FIFO(
    uint256_dt *HBM_X,         
    hls::stream<uint256_dt> &FIFO_Res,
    int iter                  
) {
    int chunks_per_row = EMBED_DIM / FP32_PACK; // 768 / 8 = 96
    int chunks_per_tile = TILE_SIZE / FP32_PACK; // 64 / 8 = 8
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

// Store Final FP32 Result (After Residual) to HBM
void store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint256_dt *HBM_Out,
    int iter
) {
    int chunks_per_row = EMBED_DIM / FP32_PACK; 
    int chunks_per_tile = TILE_SIZE / FP32_PACK; 
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

// =======================================================================
//   Compute Core: MLP Layer 2 (Projection -> Dequantize -> Residual)
// =======================================================================

void compute_mlp2_proj_and_residual(
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], 
    hls::stream<uint256_dt> &FIFO_W2,                        
    hls::stream<uint256_dt> &FIFO_Res,                       
    hls::stream<uint256_dt> &FIFO_Final_Out,                 
    float scaler_MLP1[SEQ_LEN],                              
    float BRAM_b2[EMBED_DIM],                                
    float scale_w2[EMBED_DIM],                               
    int iter
) {
    #pragma HLS INLINE off

    #ifdef __SYNTHESIS__
    int64_dt URAM_W2[TILE_SIZE][MLP_DIM_package];
    #else
    static int64_dt URAM_W2[TILE_SIZE][MLP_DIM_package];
    #endif

    // Factor = 16 to read 16 chunks of 64-bit simultaneously (128 MACs)
    #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out dim=2 type=cyclic factor=16
    #pragma HLS ARRAY_PARTITION variable=URAM_W2 dim=2 type=cyclic factor=16
    #pragma HLS BIND_STORAGE variable=URAM_W2 type=ram_2p impl=uram

    // -------------------------------------------------------------------
    // Step 1: Unpack W2 Tile from FIFO and load into URAM
    // -------------------------------------------------------------------
    load_w2_row:
    for (int i = 0; i < TILE_SIZE; i++) { 
        load_w2_col:
        for (int j = 0; j < MLP_DIM / 32; j++) { 
            #pragma HLS PIPELINE II=4
            uint256_dt w2_chunk = FIFO_W2.read();
            
            unpack_w2_loop:
            for (int k = 0; k < 4; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 64;
                int end_bit = start_bit + 63;
                URAM_W2[i][j * 4 + k] = w2_chunk.range(end_bit, start_bit);
            }
        }
    }

    // -------------------------------------------------------------------
    // Step 2 & 3: Matrix Multiplication, Dequantize, and Residual Add
    // -------------------------------------------------------------------
    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        
        int32_dt acc_buffer[TILE_SIZE];
        #pragma HLS BIND_STORAGE variable=acc_buffer type=ram_2p impl=lutram

        // === Step 2: Dot Product (128 MACs per cycle) ===
        compute_tile:
        for (int j = 0; j < TILE_SIZE; j++) { 
            int32_dt acc = 0;
            
            // 384 packages / 16 parallel reads = 24 steps
            dot_product_step:
            for (int step = 0; step < (MLP_DIM_package / 16); step++) {
                #pragma HLS PIPELINE II=1
                int32_dt cycle_acc = 0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL
                    int pack_idx = step * 16 + p;
                    
                    int64_dt in_pack = URAM_MLP1_Quant_Out[i][pack_idx];
                    int64_dt w2_pack = URAM_W2[j][pack_idx];

                    mac_element_loop:
                    for (int r = 0; r < INT8_PACK; r++) {
                        #pragma HLS UNROLL
                        int start_bit = r * 8;
                        int end_bit = start_bit + 7;
                        
                        int8_dt in_val = (int8_dt)in_pack.range(end_bit, start_bit);
                        int8_dt w2_val = (int8_dt)w2_pack.range(end_bit, start_bit);
                        
                        cycle_acc += in_val * w2_val;
                    }
                }
                acc += cycle_acc;
            }
            acc_buffer[j] = acc;
        }

        // === Step 3: Dequantize, Add Residual, and Pack ===
        float token_scaler = scaler_MLP1[i];
        uint256_dt res_chunk = 0;
        
        float pack_array[FP32_PACK]; 
        #pragma HLS ARRAY_PARTITION variable=pack_array type=complete

        post_process_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            
            // Read new 256-bit FP32 residual chunk every 8 elements
            if (j % FP32_PACK == 0) {
                res_chunk = FIFO_Res.read();
            }

            int32_dt final_acc = acc_buffer[j];
            int weight_idx = iter * TILE_SIZE + j; 
            
            float w_scale = scale_w2[weight_idx];
            float w_bias  = BRAM_b2[weight_idx];
            float combined_scale = token_scaler * w_scale; 
            
            // FP32 projection value
            float proj_val = (float)final_acc * combined_scale + w_bias;

            // Extract matching FP32 residual from chunk
            int res_start_bit = (j % FP32_PACK) * 32;
            int res_end_bit = res_start_bit + 31;
            union { float f; uint32_t i; } res_conv;
            res_conv.i = res_chunk.range(res_end_bit, res_start_bit);
            
            // Add residual
            pack_array[j % FP32_PACK] = proj_val + res_conv.f;

            // Write 256-bit block out every 8 elements
            if ((j % FP32_PACK == FP32_PACK - 1) || (j == TILE_SIZE - 1)) {
                uint256_dt out_chunk = 0;
                
                pack_write_loop:
                for (int k = 0; k < FP32_PACK; k++) {
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

// =======================================================================
//   Dataflow Wrapper for a Single W2 Tile
// =======================================================================

void process_mlp2_single_tile_dataflow(
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], 
    uint256_dt *HBM_W2,
    uint256_dt *HBM_X_Res,       
    uint256_dt *HBM_Out,         
    float scaler_MLP1[SEQ_LEN],   
    float BRAM_b2[EMBED_DIM],    
    float scale_w2[EMBED_DIM],   
    int iter                     
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint256_dt> FIFO_W2("FIFO_W2");
    hls::stream<uint256_dt> FIFO_Res("FIFO_Res");
    hls::stream<uint256_dt> FIFO_Final_Out("FIFO_Final_Out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_W2 depth=32
    #pragma HLS STREAM variable=FIFO_Res depth=32
    #pragma HLS STREAM variable=FIFO_Final_Out depth=32
#endif
    
    // Load W2 tile
    load_W2_tile_HBM2FIFO(HBM_W2, FIFO_W2, iter);

    // Load original FP32 Residual Input (X_res)
    load_Residual_Tile_HBM2FIFO(HBM_X_Res, FIFO_Res, iter);

    // Execute Projection & Residual Addition
    compute_mlp2_proj_and_residual(
        URAM_MLP1_Quant_Out, 
        FIFO_W2, 
        FIFO_Res, 
        FIFO_Final_Out, 
        scaler_MLP1, 
        BRAM_b2, 
        scale_w2, 
        iter
    );

    // Store final token values to HBM
    store_Final_Tile_FIFO2HBM(FIFO_Final_Out, HBM_Out, iter);
}

// =======================================================================
//   MLP Layer 2 Full Module Top-Level
// =======================================================================

void run_full_mlp_layer2(
    int64_dt URAM_MLP1_Quant_Out[SEQ_LEN][MLP_DIM_package], 
    uint256_dt *HBM_W2,          
    uint256_dt *HBM_X_Res,      // Original Input X used for residual
    uint256_dt *HBM_Out,         
    float scaler_MLP1[SEQ_LEN],  
    float *HBM_b2,
    float *HBM_scale_w2
) {
    float SRAM_b2[EMBED_DIM];
    float SRAM_scale_w2[EMBED_DIM];
    
    // Partition factor = 8 aligns with post-process read requirements
    #pragma HLS ARRAY_PARTITION variable=SRAM_b2 type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_w2 type=cyclic factor=8
    #pragma HLS BIND_STORAGE variable=SRAM_b2 type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=SRAM_scale_w2 type=ram_2p impl=bram

    // Assume function exists to load EMBED_DIM (768) size arrays
    load_Bias_HBM2BRAM(HBM_b2, SRAM_b2);
    load_Bias_HBM2BRAM(HBM_scale_w2, SRAM_scale_w2);

    const int NUM_TILES = EMBED_DIM / TILE_SIZE; // 768 / 64 = 12 Tiles

    full_mlp2_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        process_mlp2_single_tile_dataflow(
            URAM_MLP1_Quant_Out,
            HBM_W2,
            HBM_X_Res,
            HBM_Out,
            scaler_MLP1,
            SRAM_b2,
            SRAM_scale_w2,
            iter
        );
    }
}