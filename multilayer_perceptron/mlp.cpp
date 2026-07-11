#include "mlp.h"
//version: 20260712


extern "C" {

void mlp(
    // Input Feature Map: 197 × 768 FP32
    uint512_dt *x,

    // LayerNorm Weights: 768 FP32
    uint512_dt *gamma,
    uint512_dt *beta,

    // MLP INT8 Weights
    uint512_dt *W1,
    uint512_dt *W2,

    // Biases
    float *b1,
    float *b2,

    // Weight Dequantization Scales
    float *scale_w1,
    float *scale_w2,

    // Final Output: 197 × 768 FP32
    uint512_dt *out,

    // MLP1 GELU FP32 Intermediate:
    // 197 × 3072 FP32
    uint512_dt *HBM_FP32_Intermediate
) {
    // =====================================================================
    // AXI Master Interfaces
    // =====================================================================

    // Input:
    #pragma HLS INTERFACE m_axi port=x                         \
        offset=slave bundle=gmem_mlp                          \
        depth=9456 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // LayerNorm gamma:
    #pragma HLS INTERFACE m_axi port=gamma                     \
        offset=slave bundle=gmem_mlp                          \
        depth=48 max_widen_bitwidth=512                       \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // LayerNorm beta:
    #pragma HLS INTERFACE m_axi port=beta                      \
        offset=slave bundle=gmem_mlp                          \
        depth=48 max_widen_bitwidth=512                       \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // Bias and scale arrays remain float pointers.
    #pragma HLS INTERFACE m_axi port=b1                        \
        offset=slave bundle=gmem_mlp                          \
        depth=3072 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=b2                        \
        offset=slave bundle=gmem_mlp                          \
        depth=768 max_widen_bitwidth=512                      \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=scale_w1                  \
        offset=slave bundle=gmem_mlp                          \
        depth=3072 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=scale_w2                  \
        offset=slave bundle=gmem_mlp                          \
        depth=768 max_widen_bitwidth=512                      \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // W1:
    #pragma HLS INTERFACE m_axi port=W1                        \
        offset=slave bundle=gmem_mlp                          \
        depth=36864 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // W2:
    #pragma HLS INTERFACE m_axi port=W2                        \
        offset=slave bundle=gmem_mlp                          \
        depth=36864 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    // Final output:
    #pragma HLS INTERFACE m_axi port=out                       \
        offset=slave bundle=gmem_mlp                          \
        depth=9456 max_widen_bitwidth=512                     \
        max_write_burst_length=64                             \
        num_write_outstanding=16

    // MLP1 FP32 intermediate:
    #pragma HLS INTERFACE m_axi port=HBM_FP32_Intermediate     \
        offset=slave bundle=gmem_mlp                          \
        depth=37824 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        max_write_burst_length=64                             \
        num_read_outstanding=16                               \
        num_write_outstanding=16

    // =====================================================================
    // AXI-Lite Control Register Interfaces
    // =====================================================================

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

    #pragma HLS INTERFACE s_axilite               \
        port=HBM_FP32_Intermediate                \
        bundle=control

    #pragma HLS INTERFACE s_axilite port=return   bundle=control

    // =====================================================================
    // LayerNorm On-Chip Buffers
    // =====================================================================

    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_beta  \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_gamma \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_beta \
        type=ram_2p impl=bram

    // =====================================================================
    // Quantized LayerNorm Output
    // =====================================================================

    static int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package];

    float scaler_X[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=8 dim=2

    #pragma HLS BIND_STORAGE variable=X_package \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=scaler_X \
        type=ram_2p impl=bram

    // =====================================================================
    // GELU Row Maximum
    // =====================================================================

    float global_max[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    // =====================================================================
    // Quantized MLP1 Output
    // =====================================================================

    static int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package];

    float scaler_MLP1[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out \
        type=cyclic factor=8 dim=2

    #pragma HLS BIND_STORAGE variable=URAM_MLP1_Quant_Out \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=scaler_MLP1 \
        type=ram_2p impl=bram

    // =====================================================================
    // Execution Flow
    // =====================================================================

    // Step 1: Load LayerNorm gamma and beta
    load_beta_gamma_HBM2SRAM(
        gamma,
        beta,
        SRAM_gamma,
        SRAM_beta
    );

    // Step 2: LayerNorm and INT8 quantization
    do_layernorm_dataflow(
        x,
        SRAM_gamma,
        SRAM_beta,
        scaler_X,
        X_package
    );

    // Step 3 and 4:
    // MLP1 → dequantization → GELU → HBM intermediate
    run_mlp1_and_gelu_dataflow(
        X_package,
        W1,
        b1,
        scale_w1,
        scaler_X,
        HBM_FP32_Intermediate,
        global_max
    );

    // Step 5:
    // HBM intermediate → dynamic INT8 quantization
    run_quantize_mlp1_dataflow(
        HBM_FP32_Intermediate,
        global_max,
        URAM_MLP1_Quant_Out,
        scaler_MLP1
    );

    // Step 6:
    // MLP2 → dequantization → residual connection
    run_full_mlp_layer2(
        URAM_MLP1_Quant_Out,
        W2,
        x,
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

// Load input X from HBM to FIFO:
void load_X_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
) {
    const int TOTAL_WORDS = (SEQ_LEN * EMBED_DIM) / 16;

    load_X_loop:
    for (int i = 0; i < TOTAL_WORDS; i++) {
        #pragma HLS PIPELINE II=1

        uint512_dt temp_chunk = HBM_X[i];
        FIFO_X.write(temp_chunk);
    }
}


// Load input X from FIFO to on-chip BRAM and unpack to FP32
void load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
) {
    load_X_col:
    for (int j = 0; j < EMBED_DIM / 16; j++) {
        #pragma HLS PIPELINE II=1

        uint512_dt temp_chunk = FIFO_X.read();

        unpack_write_loop:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            const int start_bit = k * 32;
            const int end_bit   = start_bit + 31;

            uint32_t raw_bits =
                temp_chunk.range(end_bit, start_bit);

            union {
                uint32_t i_val;
                float f_val;
            } converter;

            converter.i_val = raw_bits;

            BRAM_X[j * 16 + k] = converter.f_val;
        }
    }
}


// Load LayerNorm weights gamma and beta from HBM to SRAM
void load_beta_gamma_HBM2SRAM(
    const uint512_dt HBM_gamma[EMBED_DIM / 16],
    const uint512_dt HBM_beta[EMBED_DIM / 16],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
) {
    #pragma HLS INLINE off

    load_weights_loop:
    for (int i = 0; i < EMBED_DIM / 16; i++) {
        #pragma HLS PIPELINE II=2

        uint512_dt temp_gamma_chunk = HBM_gamma[i];
        uint512_dt temp_beta_chunk  = HBM_beta[i];

        unpack_write_loop:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            const int start_bit = k * 32;
            const int end_bit   = start_bit + 31;

            uint32_t raw_gamma_bits =
                temp_gamma_chunk.range(end_bit, start_bit);

            uint32_t raw_beta_bits =
                temp_beta_chunk.range(end_bit, start_bit);

            union {
                uint32_t i_val;
                float f_val;
            } conv_gamma, conv_beta;

            conv_gamma.i_val = raw_gamma_bits;
            conv_beta.i_val  = raw_beta_bits;

            SRAM_gamma[i * 16 + k] = conv_gamma.f_val;
            SRAM_beta[i * 16 + k]  = conv_beta.f_val;
        }
    }
}


// Load FP32 Bias or Scale arrays of size MLP_DIM.
// This remains unchanged because the top-level ports are float pointers.
void load_Bias_HBM2BRAM_MLP(
    const float *HBM_b,
    float BRAM_b[MLP_DIM]
) {
    load_b_mlp_loop:
    for (int i = 0; i < MLP_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] = HBM_b[i];
    }
}


// Load FP32 Bias or Scale arrays of size EMBED_DIM.
// This remains unchanged because the top-level ports are float pointers.
void load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
) {
    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] = HBM_b[i];
    }
}


// Load W1 tile from HBM to FIFO
void load_W1_tile_HBM2FIFO(
    const uint512_dt *HBM_W1,
    hls::stream<uint512_dt> &FIFO_W1,
    int iter
) {
    // W1 is pre-transposed externally to [3072][768].
    const int row_offset      = iter * TILE_OUT_DIM;
    const int chunks_per_row  = EMBED_DIM / 64;
    const int base_addr_idx   = row_offset * chunks_per_row;

    load_w1_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) {
        const int current_row_start =
            base_addr_idx + i * chunks_per_row;

        load_w1_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1

            uint512_dt temp_chunk =
                HBM_W1[current_row_start + j];

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

    #pragma HLS ARRAY_PARTITION variable=BRAM_X type=cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=gamma type=cyclic factor= 8
    #pragma HLS ARRAY_PARTITION variable=beta  type=cyclic factor= 8 

    float norm_buffer[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=norm_buffer type=cyclic factor=8
    
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
    float scale = 127.0f / max_val;
    
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
            float clamped_val = std::max(-127.0f, std::min(127.0f, scaled_val));
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
    hls::stream<uint512_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_beta \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_gamma \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_beta \
        type=ram_2p impl=bram

    seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        #pragma HLS stable variable=SRAM_gamma
        #pragma HLS stable variable=SRAM_beta

        float SRAM_X[EMBED_DIM];

        #pragma HLS ARRAY_PARTITION variable=SRAM_X \
            type=cyclic factor=8 dim=1

        #pragma HLS BIND_STORAGE variable=SRAM_X \
            type=ram_2p impl=bram

        load_X_FIFO2BRAM(
            FIFO_X,
            SRAM_X
        );

        compute_mean_variance(
            SRAM_X,
            SRAM_gamma,
            SRAM_beta,
            scaler,
            X_package,
            i
        );
    }
}

void do_layernorm_dataflow(
    const uint512_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt> FIFO_X("FIFO_X");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_X depth=16
#endif

    #pragma HLS stable variable=SRAM_gamma
    #pragma HLS stable variable=SRAM_beta

    load_X_HBM2FIFO(
        x,
        FIFO_X
    );

    layernorm_compute(
        FIFO_X,
        scaler_X,
        SRAM_gamma,
        SRAM_beta,
        X_package
    );
}

// =======================================================================
//   Compute Core: MLP Layer 1 (Linear + Dequantize to FP32)
// =======================================================================
void compute_linear_mlp1_tile(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    hls::stream<uint512_dt> &FIFO_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
) {
    #pragma HLS INLINE off

#ifdef __SYNTHESIS__
    int64_dt URAM_W1
        [TILE_OUT_DIM][EMBED_DIM_package];
#else
    static int64_dt URAM_W1
        [TILE_OUT_DIM][EMBED_DIM_package];
#endif

    // 8 URAM banks × 2 ports = 16 package reads per cycle.

    #pragma HLS ARRAY_PARTITION variable=URAM_W1 \
        dim=2 type=cyclic factor=8

    #pragma HLS BIND_STORAGE variable=URAM_W1 \
        type=ram_2p impl=uram

    // -------------------------------------------------------------------
    // Step 1: Load W1 tile
    //
    // One 512-bit word contains eight 64-bit packages.
    // EMBED_DIM_package = 96, so each row contains 96 / 8 = 12 words.
    // -------------------------------------------------------------------

    load_w1_local_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) {
        load_w1_local_col:
        for (int j = 0; j < EMBED_DIM_package / 8; j++) {
            #pragma HLS PIPELINE II=1

            uint512_dt w_chunk = FIFO_W1.read();

            unpack_w1_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL

                const int start_bit = k * 64;
                const int end_bit   = start_bit + 63;

                URAM_W1[i][j * 8 + k] =
                    w_chunk.range(end_bit, start_bit);
            }
        }
    }

    // -------------------------------------------------------------------
    // Step 2: Matrix multiplication
    // -------------------------------------------------------------------

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float token_scale_X = scale_X[i];

        compute_tile_out:
        for (int j = 0; j < TILE_OUT_DIM; j++) {
            int32_dt acc = 0;

            // 96 packages / 16 packages per cycle = 6 cycles.
            dot_product_step:
            for (
                int step = 0;
                step < EMBED_DIM_package / 16;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int32_dt cycle_acc = 0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL

                    int k_pkg = step * 16 + p;

                    int64_dt x_pkg =
                        X_package[i][k_pkg];

                    int64_dt w_pkg =
                        URAM_W1[j][k_pkg];

                    mac_element_loop:
                    for (int r = 0; r < 8; r++) {
                        #pragma HLS UNROLL

                        const int start_bit = r * 8;
                        const int end_bit   = start_bit + 7;

                        int8_dt x_val =
                            static_cast<int8_dt>(
                                x_pkg.range(
                                    end_bit,
                                    start_bit
                                )
                            );

                        int8_dt w_val =
                            static_cast<int8_dt>(
                                w_pkg.range(
                                    end_bit,
                                    start_bit
                                )
                            );

                        cycle_acc += x_val * w_val;
                    }
                }

                acc += cycle_acc;
            }

            // -----------------------------------------------------------
            // Step 3: Dequantization and FIFO output
            // -----------------------------------------------------------

            const int out_idx =
                iter * TILE_OUT_DIM + j;

            float w_scale = scale_w1[out_idx];
            float w_bias  = BRAM_b1[out_idx];

            float combined_scale =
                token_scale_X * w_scale;

            float fp32_out =
                static_cast<float>(acc) *
                combined_scale +
                w_bias;

            FIFO_GELU_in.write(fp32_out);
        }
    }
}

void process_mlp1_single_tile_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt> FIFO_W1("FIFO_W1");

#ifdef __SYNTHESIS__
    // 原本為 32 × 256 bit。
    // 512-bit 後使用 depth=16，維持相同 FIFO bit capacity。
    #pragma HLS STREAM variable=FIFO_W1 depth=16
#endif

    load_W1_tile_HBM2FIFO(
        HBM_W1,
        FIFO_W1,
        iter
    );

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
    const uint512_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    const float *HBM_b1,
    const float *HBM_scale_w1,
    float scale_X[SEQ_LEN]
) {
    float SRAM_b1[MLP_DIM];
    float SRAM_scale_w1[MLP_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_b1 \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_w1 \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_b1 \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_scale_w1 \
        type=ram_2p impl=bram

    load_Bias_HBM2BRAM_MLP(
        HBM_b1,
        SRAM_b1
    );

    load_Bias_HBM2BRAM_MLP(
        HBM_scale_w1,
        SRAM_scale_w1
    );

    const int NUM_TILES =
        MLP_DIM / TILE_OUT_DIM;

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

// =======================================================================
//   Phase 1 Dataflow Wrapper: MLP1 -> GELU -> Store to HBM
// =======================================================================
void run_mlp1_and_gelu_dataflow(
    int64_dt X_package[SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    const float *HBM_b1,
    const float *HBM_scale_w1,
    float scale_X[SEQ_LEN],
    uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    // MLP1 output remains one FP32 value per stream transaction.
    hls::stream<float> FIFO_GELU_in("FIFO_GELU_in");

    // GELU stage still packs eight FP32 values into 256 bits.
    // The final HBM writer combines two 256-bit words into one 512-bit word.
    hls::stream<uint256_dt> FIFO_GELU_out("FIFO_GELU_out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_GELU_in  depth=64
    #pragma HLS STREAM variable=FIFO_GELU_out depth=32
#endif

    // Step 3: MLP Layer 1
    run_full_mlp_layer1(
        X_package,
        HBM_W1,
        FIFO_GELU_in,
        HBM_b1,
        HBM_scale_w1,
        scale_X
    );

    // Step 4.1: GELU and pack eight FP32 values into 256 bits
    compute_gelu_and_write_FIFO(
        FIFO_GELU_in,
        FIFO_GELU_out,
        global_max
    );

    // Step 4.2: Combine two 256-bit packets and write one 512-bit HBM word
    store_FP32_FIFO2HBM(
        FIFO_GELU_out,
        HBM_FP32_Intermediate
    );
}

// =======================================================================
//   Phase 2 Dataflow Wrapper: Load from HBM -> Dynamic Quantize
// =======================================================================
void run_quantize_mlp1_dataflow(
    const uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    // Keep the quantization datapath at 256 bits.
    // The HBM loader splits each 512-bit word into two 256-bit packets.
    hls::stream<uint256_dt> FIFO_Quant_in("FIFO_Quant_in");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Quant_in depth=32
#endif

    // Step 5.1:
    // Read one 512-bit HBM word and emit two 256-bit FIFO packets.
    load_FP32_HBM2FIFO(
        HBM_FP32_Intermediate,
        FIFO_Quant_in
    );

    // Step 5.2:
    // Each 256-bit packet contains eight FP32 values.
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
float gelu_hw(float x) {
    #pragma HLS INLINE

    float z = 1.702f * x;
    float ax = std::abs(z);

    float e = hls::exp(-ax);
    float denom = 1.0f + e;
    float sig = 1.0f / denom;

    float final_sig =
        (x < 0.0f) ? (1.0f - sig) : sig;

    return x * final_sig;
}

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

            col_loop:
            for (int j = 0; j < TILE_OUT_DIM / 8; j++) {
                #pragma HLS PIPELINE II=8

                uint256_dt pack = 0;
                float local_max_chunk = 0.0f;

                pack_loop:
                for (int k = 0; k < 8; k++) {

                    float in_val = FIFO_GELU_in.read();
                    float gelu_val = gelu_hw(in_val);

                    union {
                        float f;
                        uint32_t i;
                    } conv;

                    conv.f = gelu_val;

                    const int start_bit = k * 32;
                    const int end_bit   = start_bit + 31;

                    pack.range(end_bit, start_bit) = conv.i;

                    float abs_val = std::abs(gelu_val);

                    local_max_chunk =
                        std::max(local_max_chunk, abs_val);
                }

                FIFO_GELU_out.write(pack);

                local_row_max =
                    std::max(local_row_max, local_max_chunk);
            }

            if (iter == 0) {
                global_max[i] = local_row_max;
            } else {
                global_max[i] =
                    std::max(global_max[i], local_row_max);
            }
        }
    }
}


// =======================================================================
//   Function 2: Store FP32 Intermediate Data from FIFO to HBM
//
//   Internal:
//       2 × 256-bit FIFO packets
//
//   External:
//       1 × 512-bit HBM word
// =======================================================================

void store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint512_dt *HBM_FP32_Intermediate
) {
    #pragma HLS INLINE off

    const int TOTAL_WORDS =
        (SEQ_LEN * MLP_DIM) / 16;

    store_hbm_loop:
    for (int i = 0; i < TOTAL_WORDS; i++) {
        #pragma HLS PIPELINE II=2

        uint256_dt lower = FIFO_GELU_out.read();
        uint256_dt upper = FIFO_GELU_out.read();

        uint512_dt hbm_word = 0;

        hbm_word.range(255, 0)   = lower;
        hbm_word.range(511, 256) = upper;

        HBM_FP32_Intermediate[i] = hbm_word;
    }
}


// =======================================================================
//   Function 3: Load FP32 Intermediate Data from HBM to FIFO
//
//   External:
//       1 × 512-bit HBM word
//
//   Internal:
//       2 × 256-bit FIFO packets
// =======================================================================

void load_FP32_HBM2FIFO(
    const uint512_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
) {
    #pragma HLS INLINE off

    const int TOTAL_WORDS =
        (SEQ_LEN * MLP_DIM) / 16;

    load_hbm_loop:
    for (int i = 0; i < TOTAL_WORDS; i++) {
        #pragma HLS PIPELINE II=2

        uint512_dt hbm_word =
            HBM_FP32_Intermediate[i];

        FIFO_Quant_in.write(
            hbm_word.range(255, 0)
        );

        FIFO_Quant_in.write(
            hbm_word.range(511, 256)
        );
    }
}


// =======================================================================
//   Function 4: Read from FIFO and perform Dynamic Quantization
// =======================================================================

void quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM / INT8_PACK],
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS INLINE off

    tile_quant_loop:
    for (int iter = 0; iter < MLP1_NUM_TILES; iter++) {

        row_quant_loop:
        for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS LOOP_FLATTEN off

            float max_val =
                std::max(global_max[i], EPSILON);

            float scale = 127.0f / max_val;

            if (iter == 0) {
                scaler_MLP1[i] =
                    max_val / 127.0f;
            }

            col_quant_loop:
            for (
                int j = 0;
                j < TILE_OUT_DIM / INT8_PACK;
                j++
            ) {
                #pragma HLS PIPELINE II=1

                // 256 bit = 8 個 FP32。
                uint256_dt fp32_pack =
                    FIFO_Quant_in.read();

                int64_dt packed_int8 = 0;

                unpack_and_quantize_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL

                    const int in_start_bit = k * 32;
                    const int in_end_bit   =
                        in_start_bit + 31;

                    uint32_t raw_bits =
                        fp32_pack.range(
                            in_end_bit,
                            in_start_bit
                        );

                    union {
                        uint32_t i;
                        float f;
                    } conv;

                    conv.i = raw_bits;

                    float scaled_val =
                        conv.f * scale;

                    float clamped_val =
                        std::max(
                            -127.0f,
                            std::min(127.0f, scaled_val)
                        );

                    int8_dt q_val =
                        static_cast<int8_dt>(
                            clamped_val >= 0.0f
                                ? clamped_val + 0.5f
                                : clamped_val - 0.5f
                        );

                    const int out_start_bit = k * 8;
                    const int out_end_bit   =
                        out_start_bit + 7;

                    packed_int8.range(
                        out_end_bit,
                        out_start_bit
                    ) = q_val;
                }

                const int global_col_idx =
                    iter *
                    (TILE_OUT_DIM / INT8_PACK) +
                    j;

                URAM_MLP1_Quant_Out
                    [i][global_col_idx] =
                    packed_int8;
            }
        }
    }
}


// =======================================================================
//   Helper Functions for MLP2 Memory Loading & Storing
// =======================================================================


// =======================================================================
//   Load W2 Tile from HBM to FIFO
// =======================================================================

void load_W2_tile_HBM2FIFO(
    const uint512_dt *HBM_W2,
    hls::stream<uint512_dt> &FIFO_W2,
    int iter
) {
    const int row_offset =
        iter * TILE_SIZE;

    const int chunks_per_row =
        MLP_DIM / 64;

    const int base_addr_idx =
        row_offset * chunks_per_row;

    load_w2_row:
    for (int i = 0; i < TILE_SIZE; i++) {

        const int current_row_start =
            base_addr_idx +
            i * chunks_per_row;

        load_w2_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1

            uint512_dt temp_chunk =
                HBM_W2[
                    current_row_start + j
                ];

            FIFO_W2.write(temp_chunk);
        }
    }
}


// =======================================================================
//   Load Residual FP32 Tile from HBM to FIFO
// =======================================================================

void load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
) {
    const int words_per_row =
        EMBED_DIM / 16;

    const int words_per_tile =
        TILE_SIZE / 16;

    const int tile_col_offset =
        iter * words_per_tile;

    load_res_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {

        const int row_start_idx =
            i * words_per_row;

        load_res_tile_loop:
        for (int j = 0; j < words_per_tile; j++) {
            #pragma HLS PIPELINE II=2

            const int target_idx =
                row_start_idx +
                tile_col_offset +
                j;

            uint512_dt hbm_word =
                HBM_X[target_idx];

            FIFO_Res.write(
                hbm_word.range(255, 0)
            );

            FIFO_Res.write(
                hbm_word.range(511, 256)
            );
        }
    }
}


// =======================================================================
//   Store Final FP32 Result to HBM
//
//   Internal FIFO:
//       2 × 256 bit
//
//   External HBM:
//       1 × 512 bit
// =======================================================================

void store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint512_dt *HBM_Out,
    int iter
) {
    const int words_per_row =
        EMBED_DIM / 16;

    const int words_per_tile =
        TILE_SIZE / 16;

    const int tile_col_offset =
        iter * words_per_tile;

    store_res_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {

        const int row_start_idx =
            i * words_per_row;

        store_res_tile_loop:
        for (int j = 0; j < words_per_tile; j++) {
            #pragma HLS PIPELINE II=2

            uint256_dt lower =
                FIFO_Final_Out.read();

            uint256_dt upper =
                FIFO_Final_Out.read();

            uint512_dt hbm_word = 0;

            hbm_word.range(255, 0)   = lower;
            hbm_word.range(511, 256) = upper;

            const int target_idx =
                row_start_idx +
                tile_col_offset +
                j;

            HBM_Out[target_idx] =
                hbm_word;
        }
    }
}


// =======================================================================
//   Compute Core: MLP Layer 2
//
//   Projection → Dequantize → Residual
// =======================================================================

void compute_mlp2_proj_and_residual(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    hls::stream<uint256_dt> &FIFO_Final_Out,

    float scaler_MLP1[SEQ_LEN],
    float BRAM_b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],
    int iter
) {
    #pragma HLS INLINE off

#ifdef __SYNTHESIS__
    int64_dt URAM_W2
        [TILE_SIZE][MLP_DIM_package];
#else
    static int64_dt URAM_W2
        [TILE_SIZE][MLP_DIM_package];
#endif
    #pragma HLS ARRAY_PARTITION variable=URAM_W2 \
        dim=2 type=cyclic factor=8

    #pragma HLS BIND_STORAGE variable=URAM_W2 \
        type=ram_2p impl=uram

    // -------------------------------------------------------------------
    // Step 1: Load W2 tile
    // -------------------------------------------------------------------

    load_w2_row:
    for (int i = 0; i < TILE_SIZE; i++) {

        load_w2_col:
        for (
            int j = 0;
            j < MLP_DIM_package / 8;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            uint512_dt w2_chunk =
                FIFO_W2.read();

            unpack_w2_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL

                const int start_bit =
                    k * 64;

                const int end_bit =
                    start_bit + 63;

                URAM_W2[i][j * 8 + k] =
                    w2_chunk.range(
                        end_bit,
                        start_bit
                    );
            }
        }
    }

    // -------------------------------------------------------------------
    // Step 2 and 3:
    //
    // Matrix multiplication
    // Dequantization
    // Residual connection
    // -------------------------------------------------------------------

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        int32_dt acc_buffer[TILE_SIZE];

        #pragma HLS BIND_STORAGE variable=acc_buffer \
            type=ram_2p impl=lutram

        // ---------------------------------------------------------------
        // Dot product
        // ---------------------------------------------------------------

        compute_tile:
        for (int j = 0; j < TILE_SIZE; j++) {

            int32_dt acc = 0;

            dot_product_step:
            for (
                int step = 0;
                step < MLP_DIM_package / 16;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int32_dt cycle_acc = 0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL

                    const int pack_idx =
                        step * 16 + p;

                    int64_dt in_pack =
                        URAM_MLP1_Quant_Out
                            [i][pack_idx];

                    int64_dt w2_pack =
                        URAM_W2[j][pack_idx];

                    mac_element_loop:
                    for (
                        int r = 0;
                        r < INT8_PACK;
                        r++
                    ) {
                        #pragma HLS UNROLL

                        const int start_bit =
                            r * 8;

                        const int end_bit =
                            start_bit + 7;

                        int8_dt in_val =
                            static_cast<int8_dt>(
                                in_pack.range(
                                    end_bit,
                                    start_bit
                                )
                            );

                        int8_dt w2_val =
                            static_cast<int8_dt>(
                                w2_pack.range(
                                    end_bit,
                                    start_bit
                                )
                            );

                        cycle_acc +=
                            in_val * w2_val;
                    }
                }

                acc += cycle_acc;
            }

            acc_buffer[j] = acc;
        }

        // ---------------------------------------------------------------
        // Dequantization, bias, residual and output packing
        // ---------------------------------------------------------------

        float token_scaler =
            scaler_MLP1[i];

        uint256_dt res_chunk = 0;

        float pack_array[FP32_PACK];

        #pragma HLS ARRAY_PARTITION variable=pack_array \
            type=complete

        post_process_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1

            if ((j % FP32_PACK) == 0) {
                res_chunk =
                    FIFO_Res.read();
            }

            int32_dt final_acc =
                acc_buffer[j];

            const int weight_idx =
                iter * TILE_SIZE + j;

            float w_scale =
                scale_w2[weight_idx];

            float w_bias =
                BRAM_b2[weight_idx];

            float combined_scale =
                token_scaler * w_scale;

            float proj_val =
                static_cast<float>(final_acc) *
                combined_scale +
                w_bias;

            const int res_start_bit =
                (j % FP32_PACK) * 32;

            const int res_end_bit =
                res_start_bit + 31;

            union {
                float f;
                uint32_t i;
            } res_conv;

            res_conv.i =
                res_chunk.range(
                    res_end_bit,
                    res_start_bit
                );

            pack_array[j % FP32_PACK] =
                proj_val + res_conv.f;

            if (
                (j % FP32_PACK) ==
                    (FP32_PACK - 1)
            ) {
                uint256_dt out_chunk = 0;

                pack_write_loop:
                for (
                    int k = 0;
                    k < FP32_PACK;
                    k++
                ) {
                    #pragma HLS UNROLL

                    union {
                        float f;
                        uint32_t i;
                    } out_conv;

                    out_conv.f =
                        pack_array[k];

                    const int out_start =
                        k * 32;

                    const int out_end =
                        out_start + 31;

                    out_chunk.range(
                        out_end,
                        out_start
                    ) = out_conv.i;
                }

                FIFO_Final_Out.write(
                    out_chunk
                );
            }
        }
    }
}


// =======================================================================
//   Sequential Loader for W2 and Residual
// =======================================================================

void load_w2_and_res_sequential(
    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,

    int iter
) {
    #pragma HLS INLINE off

    load_W2_tile_HBM2FIFO(
        HBM_W2,
        FIFO_W2,
        iter
    );

    load_Residual_Tile_HBM2FIFO(
        HBM_X_Res,
        FIFO_Res,
        iter
    );
}


// =======================================================================
//   Dataflow Wrapper for a Single W2 Tile
// =======================================================================

void process_mlp2_single_tile_dataflow(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float scaler_MLP1[SEQ_LEN],
    float BRAM_b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],

    int iter
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt>
        FIFO_W2("FIFO_W2");

    hls::stream<uint256_dt>
        FIFO_Res("FIFO_Res");

    hls::stream<uint256_dt>
        FIFO_Final_Out("FIFO_Final_Out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_W2 depth=64
    #pragma HLS STREAM variable=FIFO_Res depth=64
    #pragma HLS STREAM variable=FIFO_Final_Out depth=32
#endif

    load_w2_and_res_sequential(
        HBM_W2,
        HBM_X_Res,
        FIFO_W2,
        FIFO_Res,
        iter
    );

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

    store_Final_Tile_FIFO2HBM(
        FIFO_Final_Out,
        HBM_Out,
        iter
    );
}


// =======================================================================
//   MLP Layer 2 Full Module
// =======================================================================

void run_full_mlp_layer2(
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],

    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float scaler_MLP1[SEQ_LEN],

    const float *HBM_b2,
    const float *HBM_scale_w2
) {
    float SRAM_b2[EMBED_DIM];
    float SRAM_scale_w2[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_b2 \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_w2 \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_b2 \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_scale_w2 \
        type=ram_2p impl=bram

    load_Bias_HBM2BRAM(
        HBM_b2,
        SRAM_b2
    );

    load_Bias_HBM2BRAM(
        HBM_scale_w2,
        SRAM_scale_w2
    );

    const int NUM_TILES =
        EMBED_DIM / TILE_SIZE;

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