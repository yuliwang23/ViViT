#include "FFN1.h"
// version: 20260721_four_kernel_v0
//========================================
//============INPUT=======================
// uint512_dt *x,
// uint512_dt *gamma,
// uint512_dt *beta,
// uint512_dt *W1,
// float *b1,
// float *scale_w1,
//============OUTPUT======================
// uint512_dt *HBM_FP32_Intermediate,
// float *HBM_GELU_Max
//========================================

extern "C" {

void FFN1(
    uint512_dt *x,
    uint512_dt *gamma,
    uint512_dt *beta,
    uint512_dt *W1,
    float *b1,
    float *scale_w1,
    uint512_dt *HBM_FP32_Intermediate,
    float *HBM_GELU_Max
) {
    #pragma HLS INTERFACE m_axi port=x                         \
        offset=slave bundle=gmem_ffn1                         \
        depth=9456 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=gamma                     \
        offset=slave bundle=gmem_ffn1                         \
        depth=48 max_widen_bitwidth=512                       \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=beta                      \
        offset=slave bundle=gmem_ffn1                         \
        depth=48 max_widen_bitwidth=512                       \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=W1                        \
        offset=slave bundle=gmem_ffn1                         \
        depth=36864 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=b1                        \
        offset=slave bundle=gmem_ffn1                         \
        depth=3072 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=scale_w1                  \
        offset=slave bundle=gmem_ffn1                         \
        depth=3072 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=HBM_FP32_Intermediate     \
        offset=slave bundle=gmem_ffn1                         \
        depth=37824 max_widen_bitwidth=512                    \
        max_write_burst_length=64                             \
        num_write_outstanding=16

    #pragma HLS INTERFACE m_axi port=HBM_GELU_Max              \
        offset=slave bundle=gmem_ffn1                         \
        depth=197 max_widen_bitwidth=512                      \
        max_write_burst_length=64                             \
        num_write_outstanding=16

    #pragma HLS INTERFACE s_axilite port=x                     bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma                 bundle=control
    #pragma HLS INTERFACE s_axilite port=beta                  bundle=control
    #pragma HLS INTERFACE s_axilite port=W1                    bundle=control
    #pragma HLS INTERFACE s_axilite port=b1                    bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_w1              bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_FP32_Intermediate bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_GELU_Max          bundle=control
    #pragma HLS INTERFACE s_axilite port=return                bundle=control

    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_beta \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_gamma \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_beta \
        type=ram_2p impl=bram

    static int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package];

    float scaler_X[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=8 dim=2

    #pragma HLS BIND_STORAGE variable=X_package \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=scaler_X \
        type=ram_2p impl=bram

    float global_max[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    ffn1_load_beta_gamma_HBM2SRAM(
        gamma,
        beta,
        SRAM_gamma,
        SRAM_beta
    );

    ffn1_do_layernorm_dataflow(
        x,
        SRAM_gamma,
        SRAM_beta,
        scaler_X,
        X_package
    );

    ffn1_run_mlp1_and_gelu_dataflow(
        X_package,
        W1,
        b1,
        scale_w1,
        scaler_X,
        HBM_FP32_Intermediate,
        global_max
    );

    ffn1_store_GELU_max_HBM(
        global_max,
        HBM_GELU_Max
    );
}

}

static void ffn1_load_X_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
) {
    const int TOTAL_WORDS =
        (SEQ_LEN * EMBED_DIM) / 16;

    load_X_loop:
    for (int i = 0; i < TOTAL_WORDS; i++) {
        #pragma HLS PIPELINE II=1

        FIFO_X.write(
            HBM_X[i]
        );
    }
}

static void ffn1_load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
) {
    load_X_col:
    for (int j = 0; j < EMBED_DIM / 16; j++) {
        #pragma HLS PIPELINE II=1

        uint512_dt temp_chunk =
            FIFO_X.read();

        unpack_write_loop:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            const int start_bit =
                k * 32;

            const int end_bit =
                start_bit + 31;

            uint32_t raw_bits =
                temp_chunk.range(
                    end_bit,
                    start_bit
                );

            union {
                uint32_t i_val;
                float f_val;
            } converter;

            converter.i_val =
                raw_bits;

            BRAM_X[j * 16 + k] =
                converter.f_val;
        }
    }
}

static void ffn1_load_beta_gamma_HBM2SRAM(
    const uint512_dt HBM_gamma[EMBED_DIM / 16],
    const uint512_dt HBM_beta[EMBED_DIM / 16],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
) {
    #pragma HLS INLINE off

    load_weights_loop:
    for (int i = 0; i < EMBED_DIM / 16; i++) {
        #pragma HLS PIPELINE II=2

        uint512_dt temp_gamma_chunk =
            HBM_gamma[i];

        uint512_dt temp_beta_chunk =
            HBM_beta[i];

        unpack_write_loop:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            const int start_bit =
                k * 32;

            const int end_bit =
                start_bit + 31;

            uint32_t raw_gamma_bits =
                temp_gamma_chunk.range(
                    end_bit,
                    start_bit
                );

            uint32_t raw_beta_bits =
                temp_beta_chunk.range(
                    end_bit,
                    start_bit
                );

            union {
                uint32_t i_val;
                float f_val;
            } conv_gamma, conv_beta;

            conv_gamma.i_val =
                raw_gamma_bits;

            conv_beta.i_val =
                raw_beta_bits;

            SRAM_gamma[i * 16 + k] =
                conv_gamma.f_val;

            SRAM_beta[i * 16 + k] =
                conv_beta.f_val;
        }
    }
}

static void ffn1_load_Bias_HBM2BRAM_MLP(
    const float *HBM_b,
    float BRAM_b[MLP_DIM]
) {
    load_b_mlp_loop:
    for (int i = 0; i < MLP_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] =
            HBM_b[i];
    }
}

static void ffn1_load_W1_tile_HBM2FIFO(
    const uint512_dt *HBM_W1,
    hls::stream<uint512_dt> &FIFO_W1,
    int iter
) {
    const int row_offset =
        iter * TILE_OUT_DIM;

    const int chunks_per_row =
        EMBED_DIM / 64;

    const int base_addr_idx =
        row_offset * chunks_per_row;

    load_w1_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) {
        const int current_row_start =
            base_addr_idx +
            i * chunks_per_row;

        load_w1_col:
        for (int j = 0; j < chunks_per_row; j++) {
            #pragma HLS PIPELINE II=1

            FIFO_W1.write(
                HBM_W1[current_row_start + j]
            );
        }
    }
}

static void ffn1_compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array
        [SEQ_LEN][EMBED_DIM_package],
    int seq_idx
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_X \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=gamma \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=beta \
        type=cyclic factor=8 dim=1

    float norm_buffer[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=norm_buffer \
        type=cyclic factor=8 dim=1

    float sum =
        0.0f;

    compute_mean_loop:
    for (int j = 0; j < EMBED_DIM; j += 4) {
        #pragma HLS PIPELINE II=1

        float local_sum =
            0.0f;

        compute_mean_lane:
        for (int k = 0; k < 4; k++) {
            #pragma HLS UNROLL

            local_sum +=
                BRAM_X[j + k];
        }

        sum +=
            local_sum;
    }

    float global_mean =
        sum *
        EMBED_DIM_INV;

    float variance_sum =
        0.0f;

    compute_variance_loop:
    for (int j = 0; j < EMBED_DIM; j += 4) {
        #pragma HLS PIPELINE II=1

        float local_var_sum =
            0.0f;

        compute_variance_lane:
        for (int k = 0; k < 4; k++) {
            #pragma HLS UNROLL

            float diff =
                BRAM_X[j + k] -
                global_mean;

            local_var_sum +=
                diff *
                diff;
        }

        variance_sum +=
            local_var_sum;
    }

    float var =
        variance_sum *
        EMBED_DIM_INV;

    float inv_std_out =
        1.0f /
        std::sqrt(
            var +
            EPSILON
        );

    float max_val =
        0.0f;

    compute_norm_and_find_max_loop:
    for (int j = 0; j < EMBED_DIM; j += 4) {
        #pragma HLS PIPELINE II=1

        float local_max =
            0.0f;

        compute_norm_lane:
        for (int k = 0; k < 4; k++) {
            #pragma HLS UNROLL

            const int idx =
                j + k;

            float centered =
                BRAM_X[idx] -
                global_mean;

            float normalized =
                centered *
                inv_std_out;

            float norm_val =
                normalized *
                gamma[idx] +
                beta[idx];

            norm_buffer[idx] =
                norm_val;

            local_max =
                std::max(
                    local_max,
                    std::abs(norm_val)
                );
        }

        max_val =
            std::max(
                max_val,
                local_max
            );
    }

    max_val =
        std::max(
            max_val,
            EPSILON
        );

    float scale =
        127.0f /
        max_val;

    quant_loop:
    for (int j = 0; j < EMBED_DIM; j += 8) {
        #pragma HLS PIPELINE II=2

        int64_dt chunk =
            0;

        quant_lane:
        for (int k = 0; k < 8; k++) {
            #pragma HLS UNROLL factor=4

            const int idx =
                j + k;

            float scaled_val =
                norm_buffer[idx] *
                scale;

            float clamped_val =
                std::max(
                    -127.0f,
                    std::min(
                        127.0f,
                        scaled_val
                    )
                );

            int8_dt q_val =
                static_cast<int8_dt>(
                    clamped_val >= 0.0f
                        ? clamped_val + 0.5f
                        : clamped_val - 0.5f
                );

            const int bit_offset =
                k * 8;

            chunk.range(
                bit_offset + 7,
                bit_offset
            ) = q_val;
        }

        const int pkg_idx =
            j >> 3;

        X_package_array
            [seq_idx][pkg_idx] =
            chunk;
    }

    scaler_array[seq_idx] =
        1.0f /
        scale;
}

static void ffn1_layernorm_compute(
    hls::stream<uint512_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package]
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

        ffn1_load_X_FIFO2BRAM(
            FIFO_X,
            SRAM_X
        );

        ffn1_compute_mean_variance(
            SRAM_X,
            SRAM_gamma,
            SRAM_beta,
            scaler,
            X_package,
            i
        );
    }
}

static void ffn1_do_layernorm_dataflow(
    const uint512_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt>
        FIFO_X("FIFO_X");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_X depth=16
#endif

    #pragma HLS stable variable=SRAM_gamma
    #pragma HLS stable variable=SRAM_beta

    ffn1_load_X_HBM2FIFO(
        x,
        FIFO_X
    );

    ffn1_layernorm_compute(
        FIFO_X,
        scaler_X,
        SRAM_gamma,
        SRAM_beta,
        X_package
    );
}

static void ffn1_compute_linear_mlp1_tile(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],
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

    #pragma HLS ARRAY_PARTITION variable=URAM_W1 \
        dim=2 type=cyclic factor=8

    #pragma HLS BIND_STORAGE variable=URAM_W1 \
        type=ram_2p impl=uram

    load_w1_local_row:
    for (int i = 0; i < TILE_OUT_DIM; i++) {
        load_w1_local_col:
        for (
            int j = 0;
            j < EMBED_DIM_package / 8;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            uint512_dt w_chunk =
                FIFO_W1.read();

            unpack_w1_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL

                const int start_bit =
                    k * 64;

                const int end_bit =
                    start_bit + 63;

                URAM_W1[i][j * 8 + k] =
                    w_chunk.range(
                        end_bit,
                        start_bit
                    );
            }
        }
    }

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float token_scale_X =
            scale_X[i];

        compute_tile_out:
        for (
            int j = 0;
            j < TILE_OUT_DIM;
            j++
        ) {
            int32_dt acc =
                0;

            dot_product_step:
            for (
                int step = 0;
                step < EMBED_DIM_package / 16;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int32_dt cycle_acc =
                    0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL

                    const int k_pkg =
                        step * 16 + p;

                    int64_dt x_pkg =
                        X_package[i][k_pkg];

                    int64_dt w_pkg =
                        URAM_W1[j][k_pkg];

                    mac_element_loop:
                    for (int r = 0; r < 8; r++) {
                        #pragma HLS UNROLL

                        const int start_bit =
                            r * 8;

                        const int end_bit =
                            start_bit + 7;

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

                        cycle_acc +=
                            x_val *
                            w_val;
                    }
                }

                acc +=
                    cycle_acc;
            }

            const int out_idx =
                iter *
                TILE_OUT_DIM +
                j;

            float combined_scale =
                token_scale_X *
                scale_w1[out_idx];

            float fp32_out =
                static_cast<float>(acc) *
                combined_scale +
                BRAM_b1[out_idx];

            FIFO_GELU_in.write(
                fp32_out
            );
        }
    }
}

static void ffn1_process_mlp1_single_tile_dataflow(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    hls::stream<float> &FIFO_GELU_in,
    float BRAM_b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    float scale_X[SEQ_LEN],
    int iter
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt>
        FIFO_W1("FIFO_W1");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_W1 depth=16
#endif

    ffn1_load_W1_tile_HBM2FIFO(
        HBM_W1,
        FIFO_W1,
        iter
    );

    ffn1_compute_linear_mlp1_tile(
        X_package,
        FIFO_W1,
        FIFO_GELU_in,
        BRAM_b1,
        scale_w1,
        scale_X,
        iter
    );
}

static void ffn1_run_full_mlp_layer1(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],
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

    ffn1_load_Bias_HBM2BRAM_MLP(
        HBM_b1,
        SRAM_b1
    );

    ffn1_load_Bias_HBM2BRAM_MLP(
        HBM_scale_w1,
        SRAM_scale_w1
    );

    const int NUM_TILES =
        MLP_DIM /
        TILE_OUT_DIM;

    full_mlp1_loop:
    for (
        int iter = 0;
        iter < NUM_TILES;
        iter++
    ) {
        ffn1_process_mlp1_single_tile_dataflow(
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

static float ffn1_gelu_hw(float x) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=4

    const float SQRT_2_OVER_PI =
        0.7978845608028654f;

    const float CUBIC_COEFF =
        0.044715f;

    float x_squared =
        x *
        x;

    float x_cubed =
        x_squared *
        x;

    float cubic_term =
        CUBIC_COEFF *
        x_cubed;

    float tanh_input =
        SQRT_2_OVER_PI *
        (
            x +
            cubic_term
        );

    float tanh_out =
        hls::tanhf(
            tanh_input
        );

    return
        0.5f *
        x *
        (
            1.0f +
            tanh_out
        );
}

static void ffn1_compute_gelu_and_write_FIFO(
    hls::stream<float> &FIFO_GELU_in,
    hls::stream<uint256_dt> &FIFO_GELU_out,
    float global_max[SEQ_LEN]
) {
    #pragma HLS INLINE off

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    #pragma HLS ALLOCATION \
        instances=ffn1_gelu_hw \
        limit=1 \
        function

    tile_loop:
    for (
        int iter = 0;
        iter < MLP1_NUM_TILES;
        iter++
    ) {
        row_loop:
        for (
            int i = 0;
            i < SEQ_LEN;
            i++
        ) {
            float local_row_max =
                0.0f;

            col_loop:
            for (
                int j = 0;
                j < TILE_OUT_DIM / 8;
                j++
            ) {
                uint32_t packed_bits[8];
                float abs_values[8];

                #pragma HLS ARRAY_PARTITION variable=packed_bits \
                    type=complete dim=1

                #pragma HLS ARRAY_PARTITION variable=abs_values \
                    type=complete dim=1

                gelu_lane_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS PIPELINE II=4

                    float gelu_val =
                        ffn1_gelu_hw(
                            FIFO_GELU_in.read()
                        );

                    union {
                        float f;
                        uint32_t i;
                    } conv;

                    conv.f =
                        gelu_val;

                    packed_bits[k] =
                        conv.i;

                    abs_values[k] =
                        std::abs(
                            gelu_val
                        );
                }

                uint256_dt pack =
                    0;

                float local_max_chunk =
                    0.0f;

                gelu_pack_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL

                    const int start_bit =
                        k * 32;

                    const int end_bit =
                        start_bit + 31;

                    pack.range(
                        end_bit,
                        start_bit
                    ) = packed_bits[k];

                    local_max_chunk =
                        std::max(
                            local_max_chunk,
                            abs_values[k]
                        );
                }

                FIFO_GELU_out.write(
                    pack
                );

                local_row_max =
                    std::max(
                        local_row_max,
                        local_max_chunk
                    );
            }

            if (iter == 0) {
                global_max[i] =
                    local_row_max;
            } else {
                global_max[i] =
                    std::max(
                        global_max[i],
                        local_row_max
                    );
            }
        }
    }
}

static void ffn1_store_FP32_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_GELU_out,
    uint512_dt *HBM_FP32_Intermediate
) {
    #pragma HLS INLINE off

    const int TOTAL_WORDS =
        (SEQ_LEN * MLP_DIM) /
        16;

    store_hbm_loop:
    for (
        int i = 0;
        i < TOTAL_WORDS;
        i++
    ) {
        #pragma HLS PIPELINE II=2

        uint256_dt lower =
            FIFO_GELU_out.read();

        uint256_dt upper =
            FIFO_GELU_out.read();

        uint512_dt hbm_word =
            0;

        hbm_word.range(
            255,
            0
        ) = lower;

        hbm_word.range(
            511,
            256
        ) = upper;

        HBM_FP32_Intermediate[i] =
            hbm_word;
    }
}

static void ffn1_run_mlp1_and_gelu_dataflow(
    int64_dt X_package
        [SEQ_LEN][EMBED_DIM_package],
    const uint512_dt *HBM_W1,
    const float *HBM_b1,
    const float *HBM_scale_w1,
    float scale_X[SEQ_LEN],
    uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<float>
        FIFO_GELU_in("FIFO_GELU_in");

    hls::stream<uint256_dt>
        FIFO_GELU_out("FIFO_GELU_out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_GELU_in depth=64
    #pragma HLS STREAM variable=FIFO_GELU_out depth=32
#endif

    ffn1_run_full_mlp_layer1(
        X_package,
        HBM_W1,
        FIFO_GELU_in,
        HBM_b1,
        HBM_scale_w1,
        scale_X
    );

    ffn1_compute_gelu_and_write_FIFO(
        FIFO_GELU_in,
        FIFO_GELU_out,
        global_max
    );

    ffn1_store_FP32_FIFO2HBM(
        FIFO_GELU_out,
        HBM_FP32_Intermediate
    );
}

static void ffn1_store_GELU_max_HBM(
    float global_max[SEQ_LEN],
    float *HBM_GELU_Max
) {
    #pragma HLS INLINE off

    store_gelu_max_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        HBM_GELU_Max[i] =
            global_max[i];
    }
}
