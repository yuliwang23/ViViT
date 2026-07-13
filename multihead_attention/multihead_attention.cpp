#include "multihead_attention.h"
//version: 20260713

extern "C" {

void multihead_attention(
    uint512_dt *x,
    uint512_dt *gamma,
    uint512_dt *beta,
    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,
    uint512_dt *Wo,
    float *bq,
    float *bk,
    float *bv,
    float *bo,
    float *scale_wq,
    float *scale_wk,
    float *scale_wv,
    float *scale_wo,
    uint512_dt *HBM_Attention_Out,
    uint512_dt *out
) {
    #pragma HLS INTERFACE m_axi \
        port=x \
        offset=slave \
        bundle=gmem_mha \
        depth=9456 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=gamma \
        offset=slave \
        bundle=gmem_mha \
        depth=48 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=beta \
        offset=slave \
        bundle=gmem_mha \
        depth=48 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=Wq \
        offset=slave \
        bundle=gmem_mha \
        depth=9216 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=Wk \
        offset=slave \
        bundle=gmem_mha \
        depth=9216 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=Wv \
        offset=slave \
        bundle=gmem_mha \
        depth=9216 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=Wo \
        offset=slave \
        bundle=gmem_mha \
        depth=9216 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=bq \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=bk \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=bv \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=bo \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=scale_wq \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=scale_wk \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=scale_wv \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=scale_wo \
        offset=slave \
        bundle=gmem_mha \
        depth=768 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=HBM_Attention_Out \
        offset=slave \
        bundle=gmem_mha \
        depth=9456 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=out \
        offset=slave \
        bundle=gmem_mha \
        depth=9456 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE s_axilite port=x                 bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma             bundle=control
    #pragma HLS INTERFACE s_axilite port=beta              bundle=control
    #pragma HLS INTERFACE s_axilite port=Wq                bundle=control
    #pragma HLS INTERFACE s_axilite port=Wk                bundle=control
    #pragma HLS INTERFACE s_axilite port=Wv                bundle=control
    #pragma HLS INTERFACE s_axilite port=Wo                bundle=control
    #pragma HLS INTERFACE s_axilite port=bq                bundle=control
    #pragma HLS INTERFACE s_axilite port=bk                bundle=control
    #pragma HLS INTERFACE s_axilite port=bv                bundle=control
    #pragma HLS INTERFACE s_axilite port=bo                bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wq          bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wk          bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wv          bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wo          bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_Attention_Out bundle=control
    #pragma HLS INTERFACE s_axilite port=out               bundle=control
    #pragma HLS INTERFACE s_axilite port=return            bundle=control

    float SRAM_gamma[EMBED_DIM];
    float SRAM_beta[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=SRAM_beta \
        type=cyclic factor=8

    #pragma HLS BIND_STORAGE variable=SRAM_gamma \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_beta \
        type=ram_2p impl=bram

    float SRAM_bq[EMBED_DIM];
    float SRAM_bk[EMBED_DIM];
    float SRAM_bv[EMBED_DIM];

    float SRAM_swq[EMBED_DIM];
    float SRAM_swk[EMBED_DIM];
    float SRAM_swv[EMBED_DIM];

    #pragma HLS BIND_STORAGE variable=SRAM_bq \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_bk \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_bv \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_swq \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_swk \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_swv \
        type=ram_2p impl=bram

    static int64_dt
        X_package[SEQ_LEN][EMBED_DIM_package];

    float scaler_X[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    #pragma HLS BIND_STORAGE variable=X_package \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=scaler_X \
        type=ram_2p impl=bram

    float global_max[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    load_beta_gamma_HBM2SRAM(
        gamma,
        beta,
        SRAM_gamma,
        SRAM_beta
    );

    load_Bias_HBM2BRAM(
        bq,
        SRAM_bq
    );

    load_Bias_HBM2BRAM(
        bk,
        SRAM_bk
    );

    load_Bias_HBM2BRAM(
        bv,
        SRAM_bv
    );

    load_Bias_HBM2BRAM(
        scale_wq,
        SRAM_swq
    );

    load_Bias_HBM2BRAM(
        scale_wk,
        SRAM_swk
    );

    load_Bias_HBM2BRAM(
        scale_wv,
        SRAM_swv
    );

    do_layernorm_dataflow(
        x,
        SRAM_gamma,
        SRAM_beta,
        scaler_X,
        X_package
    );

    head_loop:
    for (
        int head_idx = 0;
        head_idx < NUM_HEADS;
        head_idx++
    ) {
        process_attention_one_head_dataflow(
            X_package,
            Wq,
            Wk,
            Wv,
            SRAM_bq,
            SRAM_bk,
            SRAM_bv,
            SRAM_swq,
            SRAM_swk,
            SRAM_swv,
            scaler_X,
            HBM_Attention_Out,
            global_max,
            head_idx
        );
    }

    run_full_output_projection(
        HBM_Attention_Out,
        Wo,
        x,
        out,
        global_max,
        bo,
        scale_wo
    );
}

}

void load_X_HBM2FIFO(
    uint512_dt *HBM_X,
    hls::stream<uint512_dt> &FIFO_X
) {
    constexpr int TOTAL_X_WORDS =
        (SEQ_LEN * EMBED_DIM) / FP32_PER_512;

    load_X_loop:
    for (int i = 0; i < TOTAL_X_WORDS; i++) {
        #pragma HLS PIPELINE II=1

        uint512_dt temp_chunk =
            HBM_X[i];

        FIFO_X.write(
            temp_chunk
        );
    }
}

void load_X_FIFO2BRAM(
    hls::stream<uint512_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
) {
    constexpr int WORDS_PER_TOKEN =
        EMBED_DIM / FP32_PER_512;

    load_X_col:
    for (int j = 0; j < WORDS_PER_TOKEN; j++) {
        #pragma HLS PIPELINE II=1

        uint512_dt temp_chunk =
            FIFO_X.read();

        unpack_write_loop:
        for (int k = 0; k < FP32_PER_512; k++) {
            #pragma HLS UNROLL

            int start_bit =
                k * 32;

            int end_bit =
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

            BRAM_X[
                j * FP32_PER_512 + k
            ] = converter.f_val;
        }
    }
}

void load_beta_gamma_HBM2SRAM(
    uint512_dt HBM_gamma[
        EMBED_DIM / FP32_PER_512
    ],

    uint512_dt HBM_beta[
        EMBED_DIM / FP32_PER_512
    ],

    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
) {
    #pragma HLS INLINE off

    constexpr int PARAM_WORDS =
        EMBED_DIM / FP32_PER_512;

    load_weights_loop:
    for (int i = 0; i < PARAM_WORDS; i++) {
        #pragma HLS PIPELINE II=2

        uint512_dt temp_gamma_chunk =
            HBM_gamma[i];

        uint512_dt temp_beta_chunk =
            HBM_beta[i];

        unpack_write_loop:
        for (int k = 0; k < FP32_PER_512; k++) {
            #pragma HLS UNROLL

            int start_bit =
                k * 32;

            int end_bit =
                start_bit + 31;

            uint32_t raw_gamma_bits =
                temp_gamma_chunk.range(
                    end_bit,
                    start_bit
                );

            union {
                uint32_t i_val;
                float f_val;
            } conv_gamma;

            conv_gamma.i_val =
                raw_gamma_bits;

            SRAM_gamma[
                i * FP32_PER_512 + k
            ] = conv_gamma.f_val;

            uint32_t raw_beta_bits =
                temp_beta_chunk.range(
                    end_bit,
                    start_bit
                );

            union {
                uint32_t i_val;
                float f_val;
            } conv_beta;

            conv_beta.i_val =
                raw_beta_bits;

            SRAM_beta[
                i * FP32_PER_512 + k
            ] = conv_beta.f_val;
        }
    }
}

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float scaler_array[SEQ_LEN],
    int64_dt X_package_array
        [SEQ_LEN]
        [EMBED_DIM_package],
    int seq_idx
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_X \
        type=cyclic factor=16

    #pragma HLS ARRAY_PARTITION variable=gamma \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=beta \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=X_package_array \
        type=cyclic factor=6 dim=2

    float norm_buffer[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=norm_buffer \
        type=cyclic factor=16

    float sum =
        0.0f;

    compute_mean_loop:
    for (int j = 0; j < EMBED_DIM; j += 16) {
        #pragma HLS PIPELINE

        float local_sum =
            0.0f;

        compute_mean_unroll:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            local_sum +=
                BRAM_X[j + k];
        }

        sum +=
            local_sum;
    }

    float global_mean =
        sum * EMBED_DIM_INV;

    float variance_sum =
        0.0f;

    compute_variance_loop:
    for (int j = 0; j < EMBED_DIM; j += 16) {
        #pragma HLS PIPELINE

        float local_var_sum =
            0.0f;

        compute_variance_unroll:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            float diff =
                BRAM_X[j + k]
                - global_mean;

            local_var_sum +=
                diff * diff;
        }

        variance_sum +=
            local_var_sum;
    }

    float var =
        variance_sum
        * EMBED_DIM_INV;

    float inv_std_out =
        1.0f
        / std::sqrt(
            var + EPSILON
        );

    float max_val =
        0.0f;

    compute_norm_and_find_max_loop:
    for (int j = 0; j < EMBED_DIM; j += 16) {
        #pragma HLS PIPELINE II=1

        float local_max =
            0.0f;

        compute_norm_unroll:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            int idx =
                j + k;

            float norm_val =
                (BRAM_X[idx] - global_mean)
                * inv_std_out
                * gamma[idx]
                + beta[idx];

            norm_buffer[idx] =
                norm_val;

            float abs_val =
                std::abs(
                    norm_val
                );

            local_max =
                std::max(
                    local_max,
                    abs_val
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
        127.0f / max_val;

    quant_loop:
    for (int j = 0; j < EMBED_DIM; j += 16) {
        #pragma HLS PIPELINE II=1

        int64_dt chunk0 =
            0;

        int64_dt chunk1 =
            0;

        pack_loop:
        for (int k = 0; k < 16; k++) {
            #pragma HLS UNROLL

            int idx =
                j + k;

            float scaled_val =
                norm_buffer[idx]
                * scale;

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

            int bit_offset =
                (k % INT8_PACK)
                * 8;

            if (k < INT8_PACK) {
                chunk0.range(
                    bit_offset + 7,
                    bit_offset
                ) = q_val;
            } else {
                chunk1.range(
                    bit_offset + 7,
                    bit_offset
                ) = q_val;
            }
        }

        int pkg_idx =
            j / INT8_PACK;

        X_package_array[
            seq_idx
        ][pkg_idx] =
            chunk0;

        X_package_array[
            seq_idx
        ][pkg_idx + 1] =
            chunk1;
    }

    scaler_array[seq_idx] =
        1.0f / scale;
}

void layernorm_compute(
    hls::stream<uint512_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package]
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=SRAM_gamma \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=SRAM_beta \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    #pragma HLS BIND_STORAGE variable=SRAM_gamma \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_beta \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=X_package \
        type=ram_2p impl=uram

    seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off
        #pragma HLS stable variable=SRAM_gamma
        #pragma HLS stable variable=SRAM_beta

        float SRAM_X[EMBED_DIM];

        #pragma HLS ARRAY_PARTITION variable=SRAM_X \
            type=cyclic factor=16

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

void load_Weight_tile_HBM2FIFO(
    uint512_dt *W,
    hls::stream<uint512_dt> &FIFO_W,
    int iter
) {
    constexpr int CHUNKS_PER_ROW =
        EMBED_DIM / INT8_PER_512;

    int row_offset =
        iter * TILE_SIZE;

    int base_addr_idx =
        row_offset
        * CHUNKS_PER_ROW;

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start =
            base_addr_idx
            + i * CHUNKS_PER_ROW;

        load_tile_col:
        for (int j = 0; j < CHUNKS_PER_ROW; j++) {
            #pragma HLS PIPELINE II=1

            FIFO_W.write(
                W[current_row_start + j]
            );
        }
    }
}

void load_Bias_HBM2BRAM(
    float b[EMBED_DIM],
    float BRAM_b[EMBED_DIM]
) {
    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] =
            b[i];
    }
}

void load_All_Weights_Parallel(
    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

    int64_dt URAM_Wq
        [TILE_SIZE]
        [EMBED_DIM_package],

    int64_dt URAM_Wk
        [TILE_SIZE]
        [EMBED_DIM_package],

    int64_dt URAM_Wv
        [TILE_SIZE]
        [EMBED_DIM_package]
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=URAM_Wq \
        type=cyclic factor=6 dim=2

    #pragma HLS ARRAY_PARTITION variable=URAM_Wk \
        type=cyclic factor=6 dim=2

    #pragma HLS ARRAY_PARTITION variable=URAM_Wv \
        type=cyclic factor=6 dim=2

    #pragma HLS BIND_STORAGE variable=URAM_Wq \
        type=ram_t2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_Wk \
        type=ram_t2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_Wv \
        type=ram_t2p impl=uram

    constexpr int CHUNKS_PER_ROW =
        EMBED_DIM / INT8_PER_512;

    load_local_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        load_local_col:
        for (int j = 0; j < CHUNKS_PER_ROW; j++) {
            #pragma HLS PIPELINE II=1

            uint512_dt q_chunk =
                FIFO_Wq.read();

            uint512_dt k_chunk =
                FIFO_Wk.read();

            uint512_dt v_chunk =
                FIFO_Wv.read();

            unpack_loop:
            for (int k = 0; k < INT64_PER_512; k++) {
                #pragma HLS UNROLL

                int start_bit =
                    k * 64;

                int end_bit =
                    start_bit + 63;

                int package_idx =
                    j * INT64_PER_512
                    + k;

                URAM_Wq[i][package_idx] =
                    q_chunk.range(
                        end_bit,
                        start_bit
                    );

                URAM_Wk[i][package_idx] =
                    k_chunk.range(
                        end_bit,
                        start_bit
                    );

                URAM_Wv[i][package_idx] =
                    v_chunk.range(
                        end_bit,
                        start_bit
                    );
            }
        }
    }
}

static int28_dt dot_product_96_int8(
    int64_dt input_packages[
        QKV_PACKS_PER_STEP
    ],

    int64_dt weight_packages[
        QKV_PACKS_PER_STEP
    ]
) {
    #pragma HLS INLINE

    #pragma HLS ARRAY_PARTITION variable=input_packages \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=weight_packages \
        type=complete

    int16_dt product[96];

    ap_int<17> level_1[48];
    ap_int<18> level_2[24];
    ap_int<19> level_3[12];
    ap_int<20> level_4[6];
    ap_int<21> level_5[3];

    #pragma HLS ARRAY_PARTITION variable=product \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_1 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_2 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_3 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_4 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_5 \
        type=complete

    package_multiply_loop:
    for (int p = 0; p < QKV_PACKS_PER_STEP; p++) {
        #pragma HLS UNROLL

        element_multiply_loop:
        for (int r = 0; r < INT8_PACK; r++) {
            #pragma HLS UNROLL

            int start_bit =
                r * 8;

            int end_bit =
                start_bit + 7;

            int8_dt input_val =
                (int8_dt)
                input_packages[p].range(
                    end_bit,
                    start_bit
                );

            int8_dt weight_val =
                (int8_dt)
                weight_packages[p].range(
                    end_bit,
                    start_bit
                );

            product[
                p * INT8_PACK + r
            ] = input_val * weight_val;
        }
    }

    reduce_level_1:
    for (int i = 0; i < 48; i++) {
        #pragma HLS UNROLL

        level_1[i] =
            product[i * 2]
            + product[i * 2 + 1];
    }

    reduce_level_2:
    for (int i = 0; i < 24; i++) {
        #pragma HLS UNROLL

        level_2[i] =
            level_1[i * 2]
            + level_1[i * 2 + 1];
    }

    reduce_level_3:
    for (int i = 0; i < 12; i++) {
        #pragma HLS UNROLL

        level_3[i] =
            level_2[i * 2]
            + level_2[i * 2 + 1];
    }

    reduce_level_4:
    for (int i = 0; i < 6; i++) {
        #pragma HLS UNROLL

        level_4[i] =
            level_3[i * 2]
            + level_3[i * 2 + 1];
    }

    reduce_level_5:
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL

        level_5[i] =
            level_4[i * 2]
            + level_4[i * 2 + 1];
    }

    ap_int<22> partial_0 =
        level_5[0]
        + level_5[1];

    ap_int<22> partial_1 =
        level_5[2];

    ap_int<23> total =
        partial_0
        + partial_1;

    return (int28_dt)total;
}

void compute_linear_qkv_one_head(
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package],

    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,

    float BRAM_bq[EMBED_DIM],
    float BRAM_bk[EMBED_DIM],
    float BRAM_bv[EMBED_DIM],

    float scale_wq[EMBED_DIM],
    float scale_wk[EMBED_DIM],
    float scale_wv[EMBED_DIM],

    float scale_X[SEQ_LEN],

    int iter
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    #pragma HLS BIND_STORAGE variable=X_package \
        type=ram_2p impl=uram

#ifdef __SYNTHESIS__

    int64_dt URAM_Wq
        [TILE_SIZE]
        [EMBED_DIM_package];

    int64_dt URAM_Wk
        [TILE_SIZE]
        [EMBED_DIM_package];

    int64_dt URAM_Wv
        [TILE_SIZE]
        [EMBED_DIM_package];

#else

    static int64_dt URAM_Wq
        [TILE_SIZE]
        [EMBED_DIM_package];

    static int64_dt URAM_Wk
        [TILE_SIZE]
        [EMBED_DIM_package];

    static int64_dt URAM_Wv
        [TILE_SIZE]
        [EMBED_DIM_package];

#endif

    #pragma HLS ARRAY_PARTITION variable=URAM_Wq \
        type=cyclic factor=6 dim=2

    #pragma HLS ARRAY_PARTITION variable=URAM_Wk \
        type=cyclic factor=6 dim=2

    #pragma HLS ARRAY_PARTITION variable=URAM_Wv \
        type=cyclic factor=6 dim=2

    #pragma HLS BIND_STORAGE variable=URAM_Wq \
        type=ram_t2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_Wk \
        type=ram_t2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_Wv \
        type=ram_t2p impl=uram

    int28_dt BRAM_Acc_Q[TILE_SIZE];
    int28_dt BRAM_Acc_K[TILE_SIZE];
    int28_dt BRAM_Acc_V[TILE_SIZE];

    #pragma HLS BIND_STORAGE variable=BRAM_Acc_Q \
        type=ram_2p impl=lutram

    #pragma HLS BIND_STORAGE variable=BRAM_Acc_K \
        type=ram_2p impl=lutram

    #pragma HLS BIND_STORAGE variable=BRAM_Acc_V \
        type=ram_2p impl=lutram

    load_All_Weights_Parallel(
        FIFO_Wq,
        FIFO_Wk,
        FIFO_Wv,
        URAM_Wq,
        URAM_Wk,
        URAM_Wv
    );

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        int28_dt acc_q =
            0;

        int28_dt acc_k =
            0;

        int28_dt acc_v =
            0;

        compute_qkv_linear:
        for (
            int linear_idx = 0;
            linear_idx <
                TILE_SIZE * QKV_DOT_STEPS;
            linear_idx++
        ) {
            #pragma HLS PIPELINE II=1

            int output_idx =
                linear_idx
                / QKV_DOT_STEPS;

            int step_idx =
                linear_idx
                % QKV_DOT_STEPS;

            int package_base =
                step_idx
                * QKV_PACKS_PER_STEP;

            int64_dt input_packages[
                QKV_PACKS_PER_STEP
            ];

            int64_dt weight_q_packages[
                QKV_PACKS_PER_STEP
            ];

            int64_dt weight_k_packages[
                QKV_PACKS_PER_STEP
            ];

            int64_dt weight_v_packages[
                QKV_PACKS_PER_STEP
            ];

            #pragma HLS ARRAY_PARTITION variable=input_packages \
                type=complete

            #pragma HLS ARRAY_PARTITION variable=weight_q_packages \
                type=complete

            #pragma HLS ARRAY_PARTITION variable=weight_k_packages \
                type=complete

            #pragma HLS ARRAY_PARTITION variable=weight_v_packages \
                type=complete

            read_package_loop:
            for (
                int p = 0;
                p < QKV_PACKS_PER_STEP;
                p++
            ) {
                #pragma HLS UNROLL

                int package_idx =
                    package_base + p;

                input_packages[p] =
                    X_package[i][package_idx];

                weight_q_packages[p] =
                    URAM_Wq[
                        output_idx
                    ][package_idx];

                weight_k_packages[p] =
                    URAM_Wk[
                        output_idx
                    ][package_idx];

                weight_v_packages[p] =
                    URAM_Wv[
                        output_idx
                    ][package_idx];
            }

            int28_dt cycle_q =
                dot_product_96_int8(
                    input_packages,
                    weight_q_packages
                );

            int28_dt cycle_k =
                dot_product_96_int8(
                    input_packages,
                    weight_k_packages
                );

            int28_dt cycle_v =
                dot_product_96_int8(
                    input_packages,
                    weight_v_packages
                );

            int28_dt next_q;
            int28_dt next_k;
            int28_dt next_v;

            if (step_idx == 0) {
                next_q = cycle_q;
                next_k = cycle_k;
                next_v = cycle_v;
            } else {
                next_q =
                    static_cast<int28_dt>(
                        acc_q + cycle_q
                    );

                next_k =
                    static_cast<int28_dt>(
                        acc_k + cycle_k
                    );

                next_v =
                    static_cast<int28_dt>(
                        acc_v + cycle_v
                    );
            }
            acc_q =
                next_q;

            acc_k =
                next_k;

            acc_v =
                next_v;

            if (
                step_idx ==
                QKV_DOT_STEPS - 1
            ) {
                BRAM_Acc_Q[output_idx] =
                    next_q;

                BRAM_Acc_K[output_idx] =
                    next_k;

                BRAM_Acc_V[output_idx] =
                    next_v;
            }
        }

        write_out_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1

            int weight_idx =
                iter * TILE_SIZE + j;

            float input_scale =
                scale_X[i];

            float f_q =
                (float)BRAM_Acc_Q[j]
                * input_scale
                * scale_wq[weight_idx]
                + BRAM_bq[weight_idx];

            float f_k =
                (float)BRAM_Acc_K[j]
                * input_scale
                * scale_wk[weight_idx]
                + BRAM_bk[weight_idx];

            float f_v =
                (float)BRAM_Acc_V[j]
                * input_scale
                * scale_wv[weight_idx]
                + BRAM_bv[weight_idx];

            FIFO_Q_out.write(
                f_q
            );

            FIFO_K_out.write(
                f_k
            );

            FIFO_V_out.write(
                f_v
            );
        }
    }
}

void do_layernorm_dataflow(
    uint512_dt *x,
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    float scaler_X[SEQ_LEN],
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package]
) {
    #pragma HLS DATAFLOW

    hls::stream<uint512_dt>
        FIFO_X("FIFO_X");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_X depth=32
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

void load_qkv_weights_sequential(
    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,

    hls::stream<uint512_dt> &FIFO_Wq,
    hls::stream<uint512_dt> &FIFO_Wk,
    hls::stream<uint512_dt> &FIFO_Wv,

    int head_idx
) {
    #pragma HLS INLINE off

    constexpr int CHUNKS_PER_ROW =
        EMBED_DIM / INT8_PER_512;

    int row_offset =
        head_idx * TILE_SIZE;

    int base_addr_idx =
        row_offset
        * CHUNKS_PER_ROW;

    load_qkv_row_loop:
    for (int row = 0; row < TILE_SIZE; row++) {
        int current_row_start =
            base_addr_idx
            + row * CHUNKS_PER_ROW;

        load_qkv_word_loop:
        for (
            int word_idx = 0;
            word_idx < CHUNKS_PER_ROW;
            word_idx++
        ) {
            #pragma HLS PIPELINE II=3

            int address =
                current_row_start
                + word_idx;

            uint512_dt q_chunk =
                Wq[address];

            uint512_dt k_chunk =
                Wk[address];

            uint512_dt v_chunk =
                Wv[address];

            FIFO_Wq.write(
                q_chunk
            );

            FIFO_Wk.write(
                k_chunk
            );

            FIFO_Wv.write(
                v_chunk
            );
        }
    }
}

void process_attention_one_head_dataflow(
    int64_dt X_package
        [SEQ_LEN]
        [EMBED_DIM_package],

    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,

    float SRAM_bq[EMBED_DIM],
    float SRAM_bk[EMBED_DIM],
    float SRAM_bv[EMBED_DIM],

    float SRAM_swq[EMBED_DIM],
    float SRAM_swk[EMBED_DIM],
    float SRAM_swv[EMBED_DIM],

    float scaler_X[SEQ_LEN],

    uint512_dt *HBM_Attention_Out,

    float global_max[SEQ_LEN],

    int head_idx
) {
    #pragma HLS DATAFLOW

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    hls::stream<uint512_dt>
        FIFO_Wq("FIFO_Wq");

    hls::stream<uint512_dt>
        FIFO_Wk("FIFO_Wk");

    hls::stream<uint512_dt>
        FIFO_Wv("FIFO_Wv");

    hls::stream<float>
        FIFO_Q_out("FIFO_Q_out");

    hls::stream<float>
        FIFO_K_out("FIFO_K_out");

    hls::stream<float>
        FIFO_V_out("FIFO_V_out");

    int64_dt URAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK];

    int64_dt URAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK];

    int64_dt URAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK];

    float scale_Q[SEQ_LEN];
    float scale_K[SEQ_LEN];
    float scale_V[HEAD_DIM];

    #pragma HLS ARRAY_PARTITION variable=URAM_Q \
        dim=2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=URAM_K \
        dim=2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=URAM_V_T \
        dim=2 type=cyclic factor=4

    #pragma HLS BIND_STORAGE variable=URAM_Q \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_K \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=URAM_V_T \
        type=ram_2p impl=uram

    hls::stream<int8_dt>
        FIFO_Softmax_out(
            "FIFO_Softmax_out"
        );

    hls::stream<float>
        FIFO_Softmax_scaler_out(
            "FIFO_Softmax_scaler"
        );

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wq depth=32
    #pragma HLS STREAM variable=FIFO_Wk depth=32
    #pragma HLS STREAM variable=FIFO_Wv depth=32

    #pragma HLS STREAM variable=FIFO_Q_out depth=32
    #pragma HLS STREAM variable=FIFO_K_out depth=32
    #pragma HLS STREAM variable=FIFO_V_out depth=32

    #pragma HLS STREAM variable=FIFO_Softmax_out depth=200
    #pragma HLS STREAM variable=FIFO_Softmax_scaler_out depth=2
#endif

    load_qkv_weights_sequential(
        Wq,
        Wk,
        Wv,
        FIFO_Wq,
        FIFO_Wk,
        FIFO_Wv,
        head_idx
    );

    compute_linear_qkv_one_head(
        X_package,
        FIFO_Wq,
        FIFO_Wk,
        FIFO_Wv,
        FIFO_Q_out,
        FIFO_K_out,
        FIFO_V_out,
        SRAM_bq,
        SRAM_bk,
        SRAM_bv,
        SRAM_swq,
        SRAM_swk,
        SRAM_swv,
        scaler_X,
        head_idx
    );

    qkv_sram_and_quantize(
        FIFO_Q_out,
        FIFO_K_out,
        FIFO_V_out,
        URAM_Q,
        URAM_K,
        URAM_V_T,
        scale_Q,
        scale_K,
        scale_V
    );

    compute_score_and_softmax(
        URAM_Q,
        URAM_K,
        scale_Q,
        scale_K,
        FIFO_Softmax_out,
        FIFO_Softmax_scaler_out
    );

    compute_attention_output_one_head(
        FIFO_Softmax_out,
        FIFO_Softmax_scaler_out,
        URAM_V_T,
        scale_V,
        HBM_Attention_Out,
        global_max,
        head_idx
    );
}

void qkv_sram_and_quantize(
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,

    int64_dt URAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],

    float scaler_Q[SEQ_LEN],
    float scaler_K[SEQ_LEN],
    float scaler_V[HEAD_DIM]
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=URAM_Q \
        dim=2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=URAM_K \
        dim=2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=URAM_V_T \
        dim=2 type=cyclic factor=4

    float Buffer_V[SEQ_LEN][HEAD_DIM];

    #pragma HLS BIND_STORAGE variable=Buffer_V \
        type=ram_2p impl=bram

    process_qk_and_buffer_v_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float local_q[HEAD_DIM];
        float local_k[HEAD_DIM];

        #pragma HLS ARRAY_PARTITION variable=local_q \
            type=complete

        #pragma HLS ARRAY_PARTITION variable=local_k \
            type=complete

        float max_q =
            0.0f;

        float max_k =
            0.0f;

        read_and_find_max:
        for (int j = 0; j < HEAD_DIM; j++) {
            #pragma HLS PIPELINE II=1

            float q_val =
                FIFO_Q_out.read();

            float k_val =
                FIFO_K_out.read();

            float v_val =
                FIFO_V_out.read();

            local_q[j] =
                q_val;

            local_k[j] =
                k_val;

            Buffer_V[i][j] =
                v_val;

            max_q =
                std::max(
                    max_q,
                    std::abs(q_val)
                );

            max_k =
                std::max(
                    max_k,
                    std::abs(k_val)
                );
        }

        max_q =
            std::max(
                max_q,
                EPSILON
            );

        max_k =
            std::max(
                max_k,
                EPSILON
            );

        float scale_q_val =
            127.0f / max_q;

        float scale_k_val =
            127.0f / max_k;

        pack_and_write_qk_loop:
        for (
            int j = 0;
            j < HEAD_DIM / INT8_PACK;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt pack_q =
                0;

            int64_dt pack_k =
                0;

            pack_qk_loop:
            for (int k = 0; k < INT8_PACK; k++) {
                #pragma HLS UNROLL

                int idx =
                    j * INT8_PACK + k;

                float q_scaled =
                    local_q[idx]
                    * scale_q_val;

                float k_scaled =
                    local_k[idx]
                    * scale_k_val;

                float q_clamped =
                    std::max(
                        -127.0f,
                        std::min(
                            127.0f,
                            q_scaled
                        )
                    );

                float k_clamped =
                    std::max(
                        -127.0f,
                        std::min(
                            127.0f,
                            k_scaled
                        )
                    );

                int8_dt q_val =
                    (int8_dt)
                    std::round(
                        q_clamped
                    );

                int8_dt k_val =
                    (int8_dt)
                    std::round(
                        k_clamped
                    );

                int start_bit =
                    k * 8;

                int end_bit =
                    start_bit + 7;

                pack_q.range(
                    end_bit,
                    start_bit
                ) = q_val;

                pack_k.range(
                    end_bit,
                    start_bit
                ) = k_val;
            }

            URAM_Q[i][j] =
                pack_q;

            URAM_K[i][j] =
                pack_k;
        }

        scaler_Q[i] =
            1.0f / scale_q_val;

        scaler_K[i] =
            1.0f / scale_k_val;
    }

    transpose_pad_and_pack_v_loop:
    for (int j = 0; j < HEAD_DIM; j++) {
        #pragma HLS LOOP_FLATTEN off

        float max_v =
            0.0f;

        float local_v_col[SEQ_LEN];

        find_v_col_max_loop:
        for (int i = 0; i < SEQ_LEN; i++) {
            #pragma HLS PIPELINE II=1

            float val =
                Buffer_V[i][j];

            local_v_col[i] =
                val;

            max_v =
                std::max(
                    max_v,
                    std::abs(val)
                );
        }

        max_v =
            std::max(
                max_v,
                EPSILON
            );

        float scale_v_val =
            127.0f / max_v;

        write_v_transposed_packed_loop:
        for (
            int i = 0;
            i < SEQ_LEN_PADDED / INT8_PACK;
            i++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt pack_v =
                0;

            pack_v_loop:
            for (int k = 0; k < INT8_PACK; k++) {
                #pragma HLS UNROLL

                int orig_idx =
                    i * INT8_PACK + k;

                int8_dt v_val =
                    0;

                if (orig_idx < SEQ_LEN) {
                    float scaled_val =
                        local_v_col[orig_idx]
                        * scale_v_val;

                    float clamped_val =
                        std::max(
                            -127.0f,
                            std::min(
                                127.0f,
                                scaled_val
                            )
                        );

                    v_val =
                        (int8_dt)
                        std::round(
                            clamped_val
                        );
                }

                int start_bit =
                    k * 8;

                int end_bit =
                    start_bit + 7;

                pack_v.range(
                    end_bit,
                    start_bit
                ) = v_val;
            }

            URAM_V_T[j][i] =
                pack_v;
        }

        scaler_V[j] =
            1.0f / scale_v_val;
    }
}

void compute_score_and_softmax(
    int64_dt URAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    int64_dt URAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],

    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],

    hls::stream<int8_dt> &FIFO_Softmax_out,

    hls::stream<float>
        &FIFO_Softmax_scaler_out
) {
    #pragma HLS ARRAY_PARTITION variable=URAM_Q \
        dim=2 type=cyclic factor=4

    #pragma HLS ARRAY_PARTITION variable=URAM_K \
        dim=2 type=cyclic factor=4

    #pragma HLS INLINE off

    row_score_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float sq_i =
            scale_Q[i];

        float score_buffer[SEQ_LEN_PADDED];
        float exp_buffer[SEQ_LEN_PADDED];

        #pragma HLS BIND_STORAGE variable=score_buffer \
            type=ram_2p impl=lutram

        #pragma HLS BIND_STORAGE variable=exp_buffer \
            type=ram_2p impl=lutram

        float row_max =
            NEG_INF;

        pass1_dot_max:
        for (
            int k = 0;
            k < SEQ_LEN_PADDED;
            k++
        ) {
            #pragma HLS PIPELINE II=1

            float raw_score =
                NEG_INF;

            if (k < SEQ_LEN) {
                int32_dt partial_sum[8];

                #pragma HLS ARRAY_PARTITION variable=partial_sum \
                    type=complete

                score_package_loop:
                for (
                    int p = 0;
                    p < HEAD_DIM / INT8_PACK;
                    p++
                ) {
                    #pragma HLS UNROLL

                    int64_dt q_pack =
                        URAM_Q[i][p];

                    int64_dt k_pack =
                        URAM_K[k][p];

                    int32_dt package_sum =
                        0;

                    score_mac_loop:
                    for (
                        int r = 0;
                        r < INT8_PACK;
                        r++
                    ) {
                        #pragma HLS UNROLL

                        int start_bit =
                            r * 8;

                        int end_bit =
                            start_bit + 7;

                        int8_dt q_val =
                            (int8_dt)
                            q_pack.range(
                                end_bit,
                                start_bit
                            );

                        int8_dt k_val =
                            (int8_dt)
                            k_pack.range(
                                end_bit,
                                start_bit
                            );

                        package_sum +=
                            q_val * k_val;
                    }

                    partial_sum[p] =
                        package_sum;
                }

                int32_dt sum_0 =
                    partial_sum[0]
                    + partial_sum[1];

                int32_dt sum_1 =
                    partial_sum[2]
                    + partial_sum[3];

                int32_dt sum_2 =
                    partial_sum[4]
                    + partial_sum[5];

                int32_dt sum_3 =
                    partial_sum[6]
                    + partial_sum[7];

                int32_dt acc =
                    (sum_0 + sum_1)
                    + (sum_2 + sum_3);

                raw_score =
                    static_cast<float>(acc)
                    * ATTENTION_SCALE
                    * sq_i
                    * scale_K[k];
            }

            score_buffer[k] =
                raw_score;

            if (raw_score > row_max) {
                row_max =
                    raw_score;
            }
        }

        float exp_sum =
            0.0f;

        pass2_exp_sum:
        for (
            int k = 0;
            k < SEQ_LEN_PADDED;
            k++
        ) {
            #pragma HLS PIPELINE II=1

            float score =
                score_buffer[k];

            float exp_val =
                0.0f;

            if (k < SEQ_LEN) {
                exp_val =
                    hls::exp(
                        score - row_max
                    );
            }

            exp_buffer[k] =
                exp_val;

            exp_sum +=
                exp_val;
        }

        float inv_sum =
            1.0f / exp_sum;

        float row_scaler =
            inv_sum / 127.0f;

        FIFO_Softmax_scaler_out.write(
            row_scaler
        );

        pass3_quantize_write:
        for (
            int k = 0;
            k < SEQ_LEN_PADDED;
            k++
        ) {
            #pragma HLS PIPELINE II=1

            float exp_val =
                exp_buffer[k];

            int8_dt q_softmax =
                0;

            if (k < SEQ_LEN) {
                q_softmax =
                    (int8_dt)
                    std::round(
                        exp_val * 127.0f
                    );
            }

            FIFO_Softmax_out.write(
                q_softmax
            );
        }
    }
}

void compute_attention_output_one_head(
    hls::stream<int8_dt> &FIFO_Softmax_out,

    hls::stream<float>
        &FIFO_Softmax_scaler_out,

    int64_dt URAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],

    float scale_V[HEAD_DIM],

    uint512_dt *HBM_Attention_Out,

    float global_max[SEQ_LEN],

    int head_idx
) {
    #pragma HLS ARRAY_PARTITION variable=URAM_V_T \
        dim=2 type=cyclic factor=4

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    #pragma HLS INLINE off

    row_compute_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float softmax_scaler =
            FIFO_Softmax_scaler_out.read();

        int8_dt line_buffer[SEQ_LEN_PADDED];

        #pragma HLS ARRAY_PARTITION variable=line_buffer \
            type=cyclic factor=32

        read_softmax_loop:
        for (
            int k = 0;
            k < SEQ_LEN_PADDED;
            k++
        ) {
            #pragma HLS PIPELINE II=1

            line_buffer[k] =
                FIFO_Softmax_out.read();
        }

        float local_max =
            0.0f;

        float out_row_buffer[HEAD_DIM];

        #pragma HLS ARRAY_PARTITION variable=out_row_buffer \
            type=cyclic factor=16

        compute_v_loop:
        for (int c = 0; c < HEAD_DIM; c++) {
            #pragma HLS LOOP_FLATTEN off

            int32_dt acc =
                0;

            dot_product_step:
            for (int step = 0; step < 7; step++) {
                #pragma HLS PIPELINE II=1

                int32_dt partial_sum[4];

                #pragma HLS ARRAY_PARTITION variable=partial_sum \
                    type=complete

                int pack_count =
                    step == 6
                        ? 1
                        : 4;

                mac_pack_loop:
                for (int p = 0; p < 4; p++) {
                    #pragma HLS UNROLL

                    int32_dt package_sum =
                        0;

                    if (p < pack_count) {
                        int pack_idx =
                            step * 4 + p;

                        int64_dt v_pack =
                            URAM_V_T[c][pack_idx];

                        mac_element_loop:
                        for (
                            int r = 0;
                            r < INT8_PACK;
                            r++
                        ) {
                            #pragma HLS UNROLL

                            int k_idx =
                                pack_idx
                                * INT8_PACK
                                + r;

                            int8_dt s_val =
                                line_buffer[k_idx];

                            int8_dt v_val =
                                (int8_dt)
                                v_pack.range(
                                    r * 8 + 7,
                                    r * 8
                                );

                            package_sum +=
                                s_val * v_val;
                        }
                    }

                    partial_sum[p] =
                        package_sum;
                }

                int32_dt cycle_acc =
                    (partial_sum[0]
                        + partial_sum[1])
                    + (partial_sum[2]
                        + partial_sum[3]);

                acc +=
                    cycle_acc;
            }

            float out_val =
                (float)acc
                * softmax_scaler
                * scale_V[c];

            out_row_buffer[c] =
                out_val;

            local_max =
                std::max(
                    local_max,
                    std::abs(out_val)
                );
        }

        if (head_idx == 0) {
            global_max[i] =
                local_max;
        } else {
            global_max[i] =
                std::max(
                    global_max[i],
                    local_max
                );
        }

        write_head_to_hbm_loop:
        for (
            int word_idx = 0;
            word_idx < ATTN_WORDS_PER_HEAD;
            word_idx++
        ) {
            #pragma HLS PIPELINE II=1

            uint512_dt out_chunk =
                0;

            pack_fp32_loop:
            for (
                int lane = 0;
                lane < FP32_PER_512;
                lane++
            ) {
                #pragma HLS UNROLL

                int channel_idx =
                    word_idx
                    * FP32_PER_512
                    + lane;

                union {
                    float f;
                    uint32_t i;
                } converter;

                converter.f =
                    out_row_buffer[
                        channel_idx
                    ];

                int start_bit =
                    lane * 32;

                int end_bit =
                    start_bit + 31;

                out_chunk.range(
                    end_bit,
                    start_bit
                ) = converter.i;
            }

            int target_addr =
                i * ATTN_WORDS_PER_TOKEN
                + head_idx
                    * ATTN_WORDS_PER_HEAD
                + word_idx;

            HBM_Attention_Out[target_addr] =
                out_chunk;
        }
    }
}

void prepare_attention_quant_params(
    float global_max[SEQ_LEN],
    float quant_scale[SEQ_LEN],
    float out_scaler[SEQ_LEN]
) {
    #pragma HLS INLINE off

    prepare_scale_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        float max_val =
            std::max(global_max[i], EPSILON);

        quant_scale[i] =
            127.0f / max_val;

        out_scaler[i] =
            max_val * (1.0f / 127.0f);
    }
}

void load_quantize_attention_HBM2URAM(
    uint512_dt *HBM_Attention_Out,

    float quant_scale[SEQ_LEN],

    uint512_dt URAM_Attention_Q[
        SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW
    ]
) {
    #pragma HLS INLINE off

    constexpr int FP32_WORDS_PER_ROW =
        EMBED_DIM / FP32_PER_512;

    constexpr int FP32_WORDS_PER_INT8_WORD =
        INT8_PER_512 / FP32_PER_512;

    token_loop:
    for (int token = 0; token < SEQ_LEN; token++) {
        float token_quant_scale =
            quant_scale[token];

        quantized_word_loop:
        for (
            int quant_word_idx = 0;
            quant_word_idx < OPROJ_INT8_WORDS_PER_ROW;
            quant_word_idx++
        ) {
            int64_dt quant_packages[INT64_PER_512];

            #pragma HLS ARRAY_PARTITION variable=quant_packages \
                type=complete

            source_word_loop:
            for (
                int source_word_idx = 0;
                source_word_idx < FP32_WORDS_PER_INT8_WORD;
                source_word_idx++
            ) {
                int hbm_word_idx =
                    token * FP32_WORDS_PER_ROW
                    + quant_word_idx
                        * FP32_WORDS_PER_INT8_WORD
                    + source_word_idx;

                uint512_dt fp32_chunk =
                    HBM_Attention_Out[hbm_word_idx];

                quantize_half_loop:
                for (int half = 0; half < 2; half++) {
                    #pragma HLS PIPELINE II=1

                    int64_dt packed_int8 = 0;

                    quantize_lane_loop:
                    for (
                        int lane = 0;
                        lane < INT8_PACK;
                        lane++
                    ) {
                        #pragma HLS UNROLL

                        int fp32_lane =
                            half * INT8_PACK + lane;

                        int fp32_start =
                            fp32_lane * 32;

                        int fp32_end =
                            fp32_start + 31;

                        union {
                            float f;
                            uint32_t i;
                        } converter;

                        converter.i =
                            (uint32_t)fp32_chunk.range(
                                fp32_end,
                                fp32_start
                            );

                        float scaled_val =
                            converter.f
                            * token_quant_scale;

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
                                std::round(clamped_val)
                            );

                        packed_int8.range(
                            lane * 8 + 7,
                            lane * 8
                        ) = q_val;
                    }

                    int package_idx =
                        source_word_idx * 2 + half;

                    quant_packages[package_idx] =
                        packed_int8;
                }
            }

            uint512_dt quantized_word = 0;

            pack_quantized_word_loop:
            for (
                int package_idx = 0;
                package_idx < INT64_PER_512;
                package_idx++
            ) {
                #pragma HLS UNROLL

                int start_bit =
                    package_idx * 64;

                int end_bit =
                    start_bit + 63;

                quantized_word.range(
                    end_bit,
                    start_bit
                ) = quant_packages[package_idx];
            }

            int uram_addr =
                token * OPROJ_INT8_WORDS_PER_ROW
                + quant_word_idx;

            URAM_Attention_Q[uram_addr] =
                quantized_word;
        }
    }
}

void load_Residual_Tile_HBM2FIFO(
    uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_Res,

    int iter
) {
    constexpr int WORDS_PER_ROW =
        EMBED_DIM / FP32_PER_512;

    constexpr int WORDS_PER_TILE =
        TILE_SIZE / FP32_PER_512;

    int tile_col_offset =
        iter * WORDS_PER_TILE;

    load_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start =
            i * WORDS_PER_ROW;

        load_residual_tile_loop:
        for (int j = 0; j < WORDS_PER_TILE; j++) {
            #pragma HLS PIPELINE II=1

            int target_idx =
                row_start
                + tile_col_offset
                + j;

            FIFO_Res.write(
                HBM_X_Res[target_idx]
            );
        }
    }
}

void store_Final_Tile_FIFO2HBM(
    hls::stream<uint512_dt> &FIFO_Final_Out,
    uint512_dt *HBM_Out,
    int iter
) {
    constexpr int WORDS_PER_ROW =
        EMBED_DIM / FP32_PER_512;

    constexpr int WORDS_PER_TILE =
        TILE_SIZE / FP32_PER_512;

    int tile_col_offset =
        iter * WORDS_PER_TILE;

    store_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start =
            i * WORDS_PER_ROW;

        store_tile_loop:
        for (int j = 0; j < WORDS_PER_TILE; j++) {
            #pragma HLS PIPELINE II=1

            int target_idx =
                row_start
                + tile_col_offset
                + j;

            HBM_Out[target_idx] =
                FIFO_Final_Out.read();
        }
    }
}

static int32_dt dot_product_128_int8(
    uint512_dt input_word_0,
    uint512_dt input_word_1,

    uint512_dt weight_word_0,
    uint512_dt weight_word_1
) {
    #pragma HLS INLINE

    int32_dt product[128];

    int32_dt level_1[64];
    int32_dt level_2[32];
    int32_dt level_3[16];
    int32_dt level_4[8];
    int32_dt level_5[4];
    int32_dt level_6[2];

    #pragma HLS ARRAY_PARTITION variable=product \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_1 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_2 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_3 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_4 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_5 \
        type=complete

    #pragma HLS ARRAY_PARTITION variable=level_6 \
        type=complete

    multiply_word_0_loop:
    for (int idx = 0; idx < INT8_PER_512; idx++) {
        #pragma HLS UNROLL

        int start_bit =
            idx * 8;

        int end_bit =
            start_bit + 7;

        int8_dt input_val =
            (int8_dt)input_word_0.range(
                end_bit,
                start_bit
            );

        int8_dt weight_val =
            (int8_dt)weight_word_0.range(
                end_bit,
                start_bit
            );

        product[idx] =
            input_val * weight_val;
    }

    multiply_word_1_loop:
    for (int idx = 0; idx < INT8_PER_512; idx++) {
        #pragma HLS UNROLL

        int start_bit =
            idx * 8;

        int end_bit =
            start_bit + 7;

        int8_dt input_val =
            (int8_dt)input_word_1.range(
                end_bit,
                start_bit
            );

        int8_dt weight_val =
            (int8_dt)weight_word_1.range(
                end_bit,
                start_bit
            );

        product[INT8_PER_512 + idx] =
            input_val * weight_val;
    }

    reduce_level_1:
    for (int idx = 0; idx < 64; idx++) {
        #pragma HLS UNROLL

        level_1[idx] =
            product[idx * 2]
            + product[idx * 2 + 1];
    }

    reduce_level_2:
    for (int idx = 0; idx < 32; idx++) {
        #pragma HLS UNROLL

        level_2[idx] =
            level_1[idx * 2]
            + level_1[idx * 2 + 1];
    }

    reduce_level_3:
    for (int idx = 0; idx < 16; idx++) {
        #pragma HLS UNROLL

        level_3[idx] =
            level_2[idx * 2]
            + level_2[idx * 2 + 1];
    }

    reduce_level_4:
    for (int idx = 0; idx < 8; idx++) {
        #pragma HLS UNROLL

        level_4[idx] =
            level_3[idx * 2]
            + level_3[idx * 2 + 1];
    }

    reduce_level_5:
    for (int idx = 0; idx < 4; idx++) {
        #pragma HLS UNROLL

        level_5[idx] =
            level_4[idx * 2]
            + level_4[idx * 2 + 1];
    }

    reduce_level_6:
    for (int idx = 0; idx < 2; idx++) {
        #pragma HLS UNROLL

        level_6[idx] =
            level_5[idx * 2]
            + level_5[idx * 2 + 1];
    }

    return level_6[0] + level_6[1];
}

void compute_output_proj_and_residual(
    uint512_dt URAM_Attention_Q[
        SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW
    ],

    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,
    hls::stream<uint512_dt> &FIFO_Final_Out,

    float out_scaler[SEQ_LEN],

    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],

    int iter
) {
    #pragma HLS INLINE off

#ifdef __SYNTHESIS__

    uint512_dt URAM_Wo
        [TILE_SIZE]
        [OPROJ_INT8_WORDS_PER_ROW];

    #pragma HLS BIND_STORAGE variable=URAM_Wo \
        type=ram_2p impl=uram

#else

    static uint512_dt URAM_Wo
        [TILE_SIZE]
        [OPROJ_INT8_WORDS_PER_ROW];

#endif

    load_wo_row:
    for (int row = 0; row < TILE_SIZE; row++) {
        load_wo_col:
        for (
            int word_idx = 0;
            word_idx < OPROJ_INT8_WORDS_PER_ROW;
            word_idx++
        ) {
            #pragma HLS PIPELINE II=1

            URAM_Wo[row][word_idx] =
                FIFO_Wo.read();
        }
    }

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        int32_dt acc_buffer[TILE_SIZE];

        #pragma HLS BIND_STORAGE variable=acc_buffer \
            type=ram_2p impl=lutram

        int attention_row_base =
            i * OPROJ_INT8_WORDS_PER_ROW;

        compute_tile:
        for (int j = 0; j < TILE_SIZE; j++) {
            int32_dt acc = 0;

            dot_product_step:
            for (
                int step = 0;
                step < OPROJ_DOT_STEPS;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int word_offset =
                    step * OPROJ_WORDS_PER_STEP;

                int attention_addr_0 =
                    attention_row_base
                    + word_offset;

                int attention_addr_1 =
                    attention_addr_0 + 1;

                uint512_dt input_word_0 =
                    URAM_Attention_Q[
                        attention_addr_0
                    ];

                uint512_dt input_word_1 =
                    URAM_Attention_Q[
                        attention_addr_1
                    ];

                uint512_dt weight_word_0 =
                    URAM_Wo[j][word_offset];

                uint512_dt weight_word_1 =
                    URAM_Wo[j][word_offset + 1];

                int32_dt cycle_acc =
                    dot_product_128_int8(
                        input_word_0,
                        input_word_1,
                        weight_word_0,
                        weight_word_1
                    );

                int32_dt next_acc;

                if (step == 0) {
                    next_acc =
                        cycle_acc;
                } else {
                    next_acc =
                        acc + cycle_acc;
                }

                acc =
                    next_acc;

                if (
                    step ==
                    OPROJ_DOT_STEPS - 1
                ) {
                    acc_buffer[j] =
                        next_acc;
                }
            }
        }

        float token_scaler =
            out_scaler[i];

        uint512_dt res_chunk = 0;

        float pack_array[FP32_PER_512];

        #pragma HLS ARRAY_PARTITION variable=pack_array \
            type=complete

        post_process_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1

            int pack_lane =
                j % FP32_PER_512;

            if (pack_lane == 0) {
                res_chunk =
                    FIFO_Res.read();
            }

            int32_dt final_acc =
                acc_buffer[j];

            int weight_idx =
                iter * TILE_SIZE + j;

            float combined_scale =
                token_scaler
                * scale_wo[weight_idx];

            float proj_val =
                (float)final_acc
                * combined_scale
                + BRAM_bo[weight_idx];

            int res_start_bit =
                pack_lane * 32;

            int res_end_bit =
                res_start_bit + 31;

            union {
                float f;
                uint32_t i;
            } res_converter;

            res_converter.i =
                (uint32_t)res_chunk.range(
                    res_end_bit,
                    res_start_bit
                );

            pack_array[pack_lane] =
                proj_val
                + res_converter.f;

            if (
                pack_lane ==
                FP32_PER_512 - 1
            ) {
                uint512_dt out_chunk = 0;

                pack_write_loop:
                for (
                    int lane = 0;
                    lane < FP32_PER_512;
                    lane++
                ) {
                    #pragma HLS UNROLL

                    union {
                        float f;
                        uint32_t i;
                    } out_converter;

                    out_converter.f =
                        pack_array[lane];

                    int out_start =
                        lane * 32;

                    int out_end =
                        out_start + 31;

                    out_chunk.range(
                        out_end,
                        out_start
                    ) = out_converter.i;
                }

                FIFO_Final_Out.write(
                    out_chunk
                );
            }
        }
    }
}

void load_wo_residual_sequential(
    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,

    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,

    int iter
) {
    #pragma HLS INLINE off

    load_Weight_tile_HBM2FIFO(
        HBM_Wo,
        FIFO_Wo,
        iter
    );

    load_Residual_Tile_HBM2FIFO(
        HBM_X_Res,
        FIFO_Res,
        iter
    );
}

void process_o_proj_single_tile_dataflow(
    uint512_dt URAM_Attention_Q[
        SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW
    ],

    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float out_scaler[SEQ_LEN],

    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],

    int iter
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt>
        FIFO_Wo("FIFO_Wo");

    hls::stream<uint512_dt>
        FIFO_Res("FIFO_Res");

    hls::stream<uint512_dt>
        FIFO_Final_Out("FIFO_Final_Out");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wo depth=32
    #pragma HLS STREAM variable=FIFO_Res depth=32
    #pragma HLS STREAM variable=FIFO_Final_Out depth=32
#endif

    #pragma HLS stable variable=URAM_Attention_Q
    #pragma HLS stable variable=out_scaler
    #pragma HLS stable variable=BRAM_bo
    #pragma HLS stable variable=scale_wo

    load_wo_residual_sequential(
        HBM_Wo,
        HBM_X_Res,

        FIFO_Wo,
        FIFO_Res,

        iter
    );

    compute_output_proj_and_residual(
        URAM_Attention_Q,

        FIFO_Wo,
        FIFO_Res,
        FIFO_Final_Out,

        out_scaler,

        BRAM_bo,
        scale_wo,

        iter
    );

    store_Final_Tile_FIFO2HBM(
        FIFO_Final_Out,
        HBM_Out,
        iter
    );
}

void run_full_output_projection(
    uint512_dt *HBM_Attention_Out,
    uint512_dt *HBM_Wo,
    uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,

    float global_max[SEQ_LEN],

    float *HBM_bo,
    float *HBM_scale_wo
) {
    float SRAM_bo[EMBED_DIM];
    float SRAM_scale_wo[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=SRAM_bo \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_wo \
        type=cyclic factor=8

    #pragma HLS BIND_STORAGE variable=SRAM_bo \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_scale_wo \
        type=ram_2p impl=bram

    float quant_scale[SEQ_LEN];
    float out_scaler[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=quant_scale \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=out_scaler \
        type=ram_2p impl=bram

    static uint512_dt URAM_Attention_Q[
        SEQ_LEN * OPROJ_INT8_WORDS_PER_ROW
    ];

    #pragma HLS BIND_STORAGE variable=URAM_Attention_Q \
        type=ram_2p impl=uram

    load_Bias_HBM2BRAM(
        HBM_bo,
        SRAM_bo
    );

    load_Bias_HBM2BRAM(
        HBM_scale_wo,
        SRAM_scale_wo
    );

    prepare_attention_quant_params(
        global_max,
        quant_scale,
        out_scaler
    );

    load_quantize_attention_HBM2URAM(
        HBM_Attention_Out,
        quant_scale,
        URAM_Attention_Q
    );

    constexpr int NUM_TILES =
        EMBED_DIM / TILE_SIZE;

    full_proj_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        process_o_proj_single_tile_dataflow(
            URAM_Attention_Q,

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