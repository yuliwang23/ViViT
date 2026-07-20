#include "mha_attention.h"

// version: 20260721_four_kernel_v0
//========================================
//============INPUT=======================
// uint512_dt *x,
// uint512_dt *gamma,
// uint512_dt *beta,
// uint512_dt *Wq,
// uint512_dt *Wk,
// uint512_dt *Wv,
// float *bq,
// float *bk,
// float *bv,
// float *scale_wq,
// float *scale_wk,
// float *scale_wv,
//============OUTPUT======================
// uint512_dt *HBM_Attention_Out,
// float *HBM_Attention_Max
//========================================
extern "C" {

void mha_attention(
    uint512_dt *x,
    uint512_dt *gamma,
    uint512_dt *beta,
    uint512_dt *Wq,
    uint512_dt *Wk,
    uint512_dt *Wv,
    float *bq,
    float *bk,
    float *bv,
    float *scale_wq,
    float *scale_wk,
    float *scale_wv,
    uint512_dt *HBM_Attention_Out,
    float *HBM_Attention_Max
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
        port=HBM_Attention_Out \
        offset=slave \
        bundle=gmem_mha \
        depth=9456 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE m_axi \
        port=HBM_Attention_Max \
        offset=slave \
        bundle=gmem_mha \
        depth=197 \
        max_widen_bitwidth=512

    #pragma HLS INTERFACE s_axilite port=x                 bundle=control
    #pragma HLS INTERFACE s_axilite port=gamma             bundle=control
    #pragma HLS INTERFACE s_axilite port=beta              bundle=control
    #pragma HLS INTERFACE s_axilite port=Wq                bundle=control
    #pragma HLS INTERFACE s_axilite port=Wk                bundle=control
    #pragma HLS INTERFACE s_axilite port=Wv                bundle=control
    #pragma HLS INTERFACE s_axilite port=bq                bundle=control
    #pragma HLS INTERFACE s_axilite port=bk                bundle=control
    #pragma HLS INTERFACE s_axilite port=bv                bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wq          bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wk          bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wv          bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_Attention_Out bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_Attention_Max bundle=control
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

    static int64_dt X_package[SEQ_LEN][EMBED_DIM_package];
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

    process_all_heads_fifo_dataflow(
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
        global_max
    );

    store_attention_max_HBM(
        global_max,
        HBM_Attention_Max
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
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=gamma \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=beta \
        type=cyclic factor=8

    #pragma HLS ARRAY_PARTITION variable=X_package_array \
        type=cyclic factor=6 dim=2

    float norm_buffer[EMBED_DIM];

    #pragma HLS ARRAY_PARTITION variable=norm_buffer \
        type=cyclic factor=8

    float sum =
        0.0f;

    compute_mean_loop:
    for (int j = 0; j < EMBED_DIM; j += 8) {
        #pragma HLS PIPELINE

        float local_sum =
            0.0f;

        compute_mean_unroll:
        for (int k = 0; k < 8; k++) {
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
    for (int j = 0; j < EMBED_DIM; j += 8) {
        #pragma HLS PIPELINE

        float local_var_sum =
            0.0f;

        compute_variance_unroll:
        for (int k = 0; k < 8; k++) {
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
    for (int j = 0; j < EMBED_DIM; j += 8) {
        #pragma HLS PIPELINE II=1

        float local_max =
            0.0f;

        compute_norm_unroll:
        for (int k = 0; k < 8; k++) {
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
    for (int j = 0; j < EMBED_DIM; j += INT8_PACK) {
        #pragma HLS PIPELINE II=1

        int64_dt chunk =
            0;

        pack_loop:
        for (int k = 0; k < INT8_PACK; k++) {
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
                k * 8;

            chunk.range(
                bit_offset + 7,
                bit_offset
            ) = q_val;
        }

        X_package_array[
            seq_idx
        ][j / INT8_PACK] =
            chunk;
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
            type=cyclic factor=8

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
            linear_idx < TILE_SIZE * QKV_DOT_STEPS;
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

void qkv_sram_and_quantize_to_head_fifo(
    hls::stream<float> &FIFO_Q_out,
    hls::stream<float> &FIFO_K_out,
    hls::stream<float> &FIFO_V_out,
    hls::stream<int64_dt> &FIFO_Q_head,
    hls::stream<int64_dt> &FIFO_K_head,
    hls::stream<int64_dt> &FIFO_V_head,
    hls::stream<float> &FIFO_scale_Q_head,
    hls::stream<float> &FIFO_scale_K_head,
    hls::stream<float> &FIFO_scale_V_head
) {
    #pragma HLS INLINE off

    float Buffer_V_T[HEAD_DIM][SEQ_LEN_PADDED];
    float max_v_col[HEAD_DIM];

    #pragma HLS ARRAY_PARTITION variable=Buffer_V_T \
        type=cyclic factor=INT8_PACK dim=2

    #pragma HLS BIND_STORAGE variable=Buffer_V_T \
        type=ram_2p impl=bram

    #pragma HLS ARRAY_PARTITION variable=max_v_col \
        type=complete

    init_v_max_loop:
    for (int j = 0; j < HEAD_DIM; j++) {
        #pragma HLS PIPELINE II=1

        max_v_col[j] =
            0.0f;
    }

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

            Buffer_V_T[j][i] =
                v_val;

            float abs_v =
                std::abs(
                    v_val
                );

            max_v_col[j] =
                std::max(
                    max_v_col[j],
                    abs_v
                );

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
            int pack_idx = 0;
            pack_idx < HEAD_DIM / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt pack_q =
                0;

            int64_dt pack_k =
                0;

            pack_qk_loop:
            for (int lane = 0; lane < INT8_PACK; lane++) {
                #pragma HLS UNROLL

                int idx =
                    pack_idx * INT8_PACK
                    + lane;

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
                    static_cast<int8_dt>(
                        q_clamped >= 0.0f
                            ? q_clamped + 0.5f
                            : q_clamped - 0.5f
                    );

                int8_dt k_val =
                    static_cast<int8_dt>(
                        k_clamped >= 0.0f
                            ? k_clamped + 0.5f
                            : k_clamped - 0.5f
                    );

                int start_bit =
                    lane * 8;

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

            FIFO_Q_head.write(
                pack_q
            );

            FIFO_K_head.write(
                pack_k
            );
        }

        FIFO_scale_Q_head.write(
            1.0f / scale_q_val
        );

        FIFO_scale_K_head.write(
            1.0f / scale_k_val
        );
    }

    quantize_transposed_v_loop:
    for (int j = 0; j < HEAD_DIM; j++) {
        #pragma HLS LOOP_FLATTEN off

        float max_v =
            std::max(
                max_v_col[j],
                EPSILON
            );

        float scale_v_val =
            127.0f / max_v;

        write_v_transposed_packed_loop:
        for (
            int pack_idx = 0;
            pack_idx < SEQ_LEN_PADDED / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt pack_v =
                0;

            pack_v_loop:
            for (int lane = 0; lane < INT8_PACK; lane++) {
                #pragma HLS UNROLL

                int orig_idx =
                    pack_idx * INT8_PACK
                    + lane;

                int8_dt v_val =
                    0;

                if (orig_idx < SEQ_LEN) {
                    float scaled_val =
                        Buffer_V_T[j][orig_idx]
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
                        static_cast<int8_dt>(
                            clamped_val >= 0.0f
                                ? clamped_val + 0.5f
                                : clamped_val - 0.5f
                        );
                }

                int start_bit =
                    lane * 8;

                int end_bit =
                    start_bit + 7;

                pack_v.range(
                    end_bit,
                    start_bit
                ) = v_val;
            }

            FIFO_V_head.write(
                pack_v
            );
        }

        FIFO_scale_V_head.write(
            1.0f / scale_v_val
        );
    }
}

void load_one_head_from_fifo(
    hls::stream<int64_dt> &FIFO_Q_head,
    hls::stream<int64_dt> &FIFO_K_head,
    hls::stream<int64_dt> &FIFO_V_head,
    hls::stream<float> &FIFO_scale_Q_head,
    hls::stream<float> &FIFO_scale_K_head,
    hls::stream<float> &FIFO_scale_V_head,
    int64_dt BRAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    int64_dt BRAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    int64_dt BRAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],
    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],
    float scale_V[HEAD_DIM]
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_Q \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_K \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_V_T \
        type=cyclic factor=SV_PACKS_PER_CYCLE dim=2

    #pragma HLS ARRAY_PARTITION variable=scale_V \
        type=cyclic factor=SV_CHANNELS_PER_CYCLE

    #pragma HLS BIND_STORAGE variable=BRAM_Q \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=BRAM_K \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=BRAM_V_T \
        type=ram_t2p impl=bram

    load_qk_rows_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        load_qk_packs_loop:
        for (
            int pack_idx = 0;
            pack_idx < HEAD_DIM / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            BRAM_Q[i][pack_idx] =
                FIFO_Q_head.read();

            BRAM_K[i][pack_idx] =
                FIFO_K_head.read();
        }

        scale_Q[i] =
            FIFO_scale_Q_head.read();

        scale_K[i] =
            FIFO_scale_K_head.read();
    }

    load_v_columns_loop:
    for (int c = 0; c < HEAD_DIM; c++) {
        load_v_packs_loop:
        for (
            int pack_idx = 0;
            pack_idx < SEQ_LEN_PADDED / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            BRAM_V_T[c][pack_idx] =
                FIFO_V_head.read();
        }

        scale_V[c] =
            FIFO_scale_V_head.read();
    }
}

void compute_score_and_softmax_packed_fifo_buffer(
    int64_dt BRAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    int64_dt BRAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],
    hls::stream<int64_dt> &FIFO_Softmax_packed,
    hls::stream<float>
        &FIFO_Softmax_scaler_out
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_Q \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_K \
        type=complete dim=2

    #pragma HLS BIND_STORAGE variable=BRAM_Q \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=BRAM_K \
        type=ram_2p impl=bram

    row_score_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float sq_i =
            scale_Q[i];

        float score_buffer[SEQ_LEN_PADDED];
        float exp_buffer[SEQ_LEN_PADDED];

        #pragma HLS BIND_STORAGE variable=score_buffer \
            type=ram_2p impl=lutram

        #pragma HLS ARRAY_PARTITION variable=exp_buffer \
            type=cyclic factor=INT8_PACK

        #pragma HLS BIND_STORAGE variable=exp_buffer \
            type=ram_2p impl=lutram

        float max_partial[ROW_MAX_ACC_SLOTS];

        #pragma HLS ARRAY_PARTITION variable=max_partial \
            type=complete

        init_max_partial_loop:
        for (int slot = 0; slot < ROW_MAX_ACC_SLOTS; slot++) {
            #pragma HLS UNROLL

            max_partial[slot] =
                NEG_INF;
        }

        float qk_row_scale =
            ATTENTION_SCALE
            * sq_i;

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
                        BRAM_Q[i][p];

                    int64_dt k_pack =
                        BRAM_K[k][p];

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
                    * qk_row_scale
                    * scale_K[k];
            }

            score_buffer[k] =
                raw_score;

            int slot =
                k & (ROW_MAX_ACC_SLOTS - 1);

            max_partial[slot] =
                std::max(
                    max_partial[slot],
                    raw_score
                );
        }

        float max_reduce_l1[ROW_MAX_ACC_SLOTS / 2];
        float max_reduce_l2[ROW_MAX_ACC_SLOTS / 4];

        #pragma HLS ARRAY_PARTITION variable=max_reduce_l1 \
            type=complete

        #pragma HLS ARRAY_PARTITION variable=max_reduce_l2 \
            type=complete

        reduce_max_level_1:
        for (int r = 0; r < ROW_MAX_ACC_SLOTS / 2; r++) {
            #pragma HLS UNROLL

            max_reduce_l1[r] =
                std::max(
                    max_partial[r * 2],
                    max_partial[r * 2 + 1]
                );
        }

        reduce_max_level_2:
        for (int r = 0; r < ROW_MAX_ACC_SLOTS / 4; r++) {
            #pragma HLS UNROLL

            max_reduce_l2[r] =
                std::max(
                    max_reduce_l1[r * 2],
                    max_reduce_l1[r * 2 + 1]
                );
        }

        float row_max =
            std::max(
                max_reduce_l2[0],
                max_reduce_l2[1]
            );

        float exp_partial[SOFTMAX_ACC_SLOTS];

        #pragma HLS ARRAY_PARTITION variable=exp_partial \
            type=complete

        init_exp_partial_loop:
        for (int slot = 0; slot < SOFTMAX_ACC_SLOTS; slot++) {
            #pragma HLS UNROLL

            exp_partial[slot] =
                0.0f;
        }

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

            int slot =
                k & (SOFTMAX_ACC_SLOTS - 1);

            exp_partial[slot] =
                exp_partial[slot]
                + exp_val;
        }

        float reduce_l1[SOFTMAX_ACC_SLOTS / 2];
        float reduce_l2[SOFTMAX_ACC_SLOTS / 4];
        float reduce_l3[SOFTMAX_ACC_SLOTS / 8];

        #pragma HLS ARRAY_PARTITION variable=reduce_l1 \
            type=complete

        #pragma HLS ARRAY_PARTITION variable=reduce_l2 \
            type=complete

        #pragma HLS ARRAY_PARTITION variable=reduce_l3 \
            type=complete

        reduce_exp_level_1:
        for (int r = 0; r < SOFTMAX_ACC_SLOTS / 2; r++) {
            #pragma HLS UNROLL

            reduce_l1[r] =
                exp_partial[r * 2]
                + exp_partial[r * 2 + 1];
        }

        reduce_exp_level_2:
        for (int r = 0; r < SOFTMAX_ACC_SLOTS / 4; r++) {
            #pragma HLS UNROLL

            reduce_l2[r] =
                reduce_l1[r * 2]
                + reduce_l1[r * 2 + 1];
        }

        reduce_exp_level_3:
        for (int r = 0; r < SOFTMAX_ACC_SLOTS / 8; r++) {
            #pragma HLS UNROLL

            reduce_l3[r] =
                reduce_l2[r * 2]
                + reduce_l2[r * 2 + 1];
        }

        float exp_sum =
            reduce_l3[0]
            + reduce_l3[1];

        float inv_sum =
            1.0f / exp_sum;

        float row_scaler =
            inv_sum / 127.0f;

        FIFO_Softmax_scaler_out.write(
            row_scaler
        );

        pass3_quantize_pack_write:
        for (
            int pack_idx = 0;
            pack_idx < SEQ_LEN_PADDED / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt packed_word =
                0;

            quantize_softmax_lanes:
            for (int lane = 0; lane < INT8_PACK; lane++) {
                #pragma HLS UNROLL

                int k =
                    pack_idx * INT8_PACK
                    + lane;

                float exp_val =
                    exp_buffer[k];

                int8_dt q_softmax =
                    0;

                if (k < SEQ_LEN) {
                    float scaled_val =
                        exp_val * 127.0f;

                    q_softmax =
                        static_cast<int8_dt>(
                            scaled_val + 0.5f
                        );
                }

                packed_word.range(
                    lane * 8 + 7,
                    lane * 8
                ) = q_softmax;
            }

            FIFO_Softmax_packed.write(
                packed_word
            );
        }
    }
}

void compute_attention_output_one_head_from_packed_fifo(
    hls::stream<int64_dt> &FIFO_Softmax_packed,
    hls::stream<float> &FIFO_Softmax_row_scaler,
    int64_dt BRAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],
    float scale_V[HEAD_DIM],
    uint512_dt *HBM_Attention_Out,
    float global_max[SEQ_LEN],
    int head_idx
) {
    #pragma HLS INLINE off

    #pragma HLS ARRAY_PARTITION variable=BRAM_V_T \
        type=cyclic factor=SV_PACKS_PER_CYCLE dim=2

    #pragma HLS ARRAY_PARTITION variable=scale_V \
        type=cyclic factor=SV_CHANNELS_PER_CYCLE

    #pragma HLS BIND_STORAGE variable=BRAM_V_T \
        type=ram_t2p impl=bram

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    row_compute_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        float softmax_scaler =
            FIFO_Softmax_row_scaler.read();

        int8_dt line_buffer[SEQ_LEN_PADDED];

        #pragma HLS ARRAY_PARTITION variable=line_buffer \
            type=cyclic factor=SOFTMAX_VALUES_PER_CYCLE

        read_packed_softmax_row_loop:
        for (
            int pack_idx = 0;
            pack_idx < SEQ_LEN_PADDED / INT8_PACK;
            pack_idx++
        ) {
            #pragma HLS PIPELINE II=1

            int64_dt packed_word =
                FIFO_Softmax_packed.read();

            unpack_softmax_word_loop:
            for (int lane = 0; lane < INT8_PACK; lane++) {
                #pragma HLS UNROLL

                int idx =
                    pack_idx * INT8_PACK
                    + lane;

                line_buffer[idx] =
                    (int8_dt)packed_word.range(
                        lane * 8 + 7,
                        lane * 8
                    );
            }
        }

        float local_max =
            0.0f;

        float out_row_buffer[HEAD_DIM];

        #pragma HLS ARRAY_PARTITION variable=out_row_buffer \
            type=cyclic factor=16

        compute_v_channel_pair_loop:
        for (
            int c_base = 0;
            c_base < HEAD_DIM;
            c_base += SV_CHANNELS_PER_CYCLE
        ) {
            #pragma HLS LOOP_FLATTEN off

            int32_dt acc[SV_CHANNELS_PER_CYCLE];

            #pragma HLS ARRAY_PARTITION variable=acc \
                type=complete

            init_sv_acc_loop:
            for (
                int ch = 0;
                ch < SV_CHANNELS_PER_CYCLE;
                ch++
            ) {
                #pragma HLS UNROLL

                acc[ch] =
                    0;
            }

            dot_product_step:
            for (
                int step = 0;
                step <
                    (SEQ_LEN_PADDED / INT8_PACK)
                    / SV_PACKS_PER_CYCLE;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int32_dt partial_sum
                    [SV_CHANNELS_PER_CYCLE]
                    [SV_PACKS_PER_CYCLE];

                #pragma HLS ARRAY_PARTITION variable=partial_sum \
                    type=complete dim=0

                sv_channel_loop:
                for (
                    int ch = 0;
                    ch < SV_CHANNELS_PER_CYCLE;
                    ch++
                ) {
                    #pragma HLS UNROLL

                    mac_pack_loop:
                    for (
                        int p = 0;
                        p < SV_PACKS_PER_CYCLE;
                        p++
                    ) {
                        #pragma HLS UNROLL

                        int pack_idx =
                            step * SV_PACKS_PER_CYCLE
                            + p;

                        int64_dt v_pack =
                            BRAM_V_T[
                                c_base + ch
                            ][pack_idx];

                        int32_dt package_sum =
                            0;

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

                        partial_sum[ch][p] =
                            package_sum;
                    }
                }

                sv_accumulate_channel_loop:
                for (
                    int ch = 0;
                    ch < SV_CHANNELS_PER_CYCLE;
                    ch++
                ) {
                    #pragma HLS UNROLL

                    int32_dt sum_0 =
                        partial_sum[ch][0]
                        + partial_sum[ch][1];

                    int32_dt sum_1 =
                        partial_sum[ch][2]
                        + partial_sum[ch][3];

                    int32_dt cycle_acc =
                        sum_0
                        + sum_1
                        + partial_sum[ch][4];

                    acc[ch] +=
                        cycle_acc;
                }
            }

            float pair_abs[SV_CHANNELS_PER_CYCLE];

            #pragma HLS ARRAY_PARTITION variable=pair_abs \
                type=complete

            write_sv_channel_pair_loop:
            for (
                int ch = 0;
                ch < SV_CHANNELS_PER_CYCLE;
                ch++
            ) {
                #pragma HLS UNROLL

                int channel_idx =
                    c_base + ch;

                float out_val =
                    (float)acc[ch]
                    * softmax_scaler
                    * scale_V[channel_idx];

                out_row_buffer[channel_idx] =
                    out_val;

                pair_abs[ch] =
                    std::abs(out_val);
            }

            float pair_max =
                std::max(
                    pair_abs[0],
                    pair_abs[1]
                );

            local_max =
                std::max(
                    local_max,
                    pair_max
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

void produce_qkv_one_head_dataflow_to_fifo(
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
    int head_idx,
    hls::stream<int64_dt> &FIFO_Q_head,
    hls::stream<int64_dt> &FIFO_K_head,
    hls::stream<int64_dt> &FIFO_V_head,
    hls::stream<float> &FIFO_scale_Q_head,
    hls::stream<float> &FIFO_scale_K_head,
    hls::stream<float> &FIFO_scale_V_head
) {
    #pragma HLS DATAFLOW

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    hls::stream<uint512_dt>
        FIFO_Wq("FIFO_Wq_producer");

    hls::stream<uint512_dt>
        FIFO_Wk("FIFO_Wk_producer");

    hls::stream<uint512_dt>
        FIFO_Wv("FIFO_Wv_producer");

    hls::stream<float>
        FIFO_Q_out("FIFO_Q_out_producer");

    hls::stream<float>
        FIFO_K_out("FIFO_K_out_producer");

    hls::stream<float>
        FIFO_V_out("FIFO_V_out_producer");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wq depth=32
    #pragma HLS STREAM variable=FIFO_Wk depth=32
    #pragma HLS STREAM variable=FIFO_Wv depth=32

    #pragma HLS STREAM variable=FIFO_Q_out depth=32
    #pragma HLS STREAM variable=FIFO_K_out depth=32
    #pragma HLS STREAM variable=FIFO_V_out depth=32
#endif

    #pragma HLS stable variable=SRAM_bq
    #pragma HLS stable variable=SRAM_bk
    #pragma HLS stable variable=SRAM_bv
    #pragma HLS stable variable=SRAM_swq
    #pragma HLS stable variable=SRAM_swk
    #pragma HLS stable variable=SRAM_swv
    #pragma HLS stable variable=scaler_X

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

    qkv_sram_and_quantize_to_head_fifo(
        FIFO_Q_out,
        FIFO_K_out,
        FIFO_V_out,
        FIFO_Q_head,
        FIFO_K_head,
        FIFO_V_head,
        FIFO_scale_Q_head,
        FIFO_scale_K_head,
        FIFO_scale_V_head
    );
}

void consume_attention_one_head_dataflow_from_buffer(
    int64_dt BRAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    int64_dt BRAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK],
    int64_dt BRAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK],
    float scale_Q[SEQ_LEN],
    float scale_K[SEQ_LEN],
    float scale_V[HEAD_DIM],
    uint512_dt *HBM_Attention_Out,
    float global_max[SEQ_LEN],
    int head_idx
) {
    #pragma HLS DATAFLOW

    #pragma HLS ARRAY_PARTITION variable=BRAM_Q \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_K \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_V_T \
        type=cyclic factor=SV_PACKS_PER_CYCLE dim=2

    #pragma HLS ARRAY_PARTITION variable=scale_V \
        type=cyclic factor=SV_CHANNELS_PER_CYCLE

    hls::stream<int64_dt>
        FIFO_Softmax_packed(
            "FIFO_Softmax_packed_consumer"
        );

    hls::stream<float>
        FIFO_Softmax_scaler_out(
            "FIFO_Softmax_scaler_consumer"
        );

    #pragma HLS STREAM variable=FIFO_Softmax_packed \
        depth=SOFTMAX_PACKED_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_Softmax_scaler_out \
        depth=2

    #pragma HLS BIND_STORAGE variable=FIFO_Softmax_packed \
        type=fifo impl=bram

    compute_score_and_softmax_packed_fifo_buffer(
        BRAM_Q,
        BRAM_K,
        scale_Q,
        scale_K,
        FIFO_Softmax_packed,
        FIFO_Softmax_scaler_out
    );

    compute_attention_output_one_head_from_packed_fifo(
        FIFO_Softmax_packed,
        FIFO_Softmax_scaler_out,
        BRAM_V_T,
        scale_V,
        HBM_Attention_Out,
        global_max,
        head_idx
    );
}

void produce_all_heads_qkv_to_fifo(
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
    hls::stream<int64_dt> &FIFO_Q_head,
    hls::stream<int64_dt> &FIFO_K_head,
    hls::stream<int64_dt> &FIFO_V_head,
    hls::stream<float> &FIFO_scale_Q_head,
    hls::stream<float> &FIFO_scale_K_head,
    hls::stream<float> &FIFO_scale_V_head
) {
    #pragma HLS INLINE off

    producer_head_loop:
    for (
        int head_idx = 0;
        head_idx < NUM_HEADS;
        head_idx++
    ) {
        #pragma HLS LOOP_FLATTEN off

        produce_qkv_one_head_dataflow_to_fifo(
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
            head_idx,
            FIFO_Q_head,
            FIFO_K_head,
            FIFO_V_head,
            FIFO_scale_Q_head,
            FIFO_scale_K_head,
            FIFO_scale_V_head
        );
    }
}

void consume_all_heads_attention_from_fifo(
    hls::stream<int64_dt> &FIFO_Q_head,
    hls::stream<int64_dt> &FIFO_K_head,
    hls::stream<int64_dt> &FIFO_V_head,
    hls::stream<float> &FIFO_scale_Q_head,
    hls::stream<float> &FIFO_scale_K_head,
    hls::stream<float> &FIFO_scale_V_head,
    uint512_dt *HBM_Attention_Out,
    float global_max[SEQ_LEN]
) {
    #pragma HLS INLINE off

    int64_dt BRAM_Q
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK];

    int64_dt BRAM_K
        [SEQ_LEN]
        [HEAD_DIM / INT8_PACK];

    int64_dt BRAM_V_T
        [HEAD_DIM]
        [SEQ_LEN_PADDED / INT8_PACK];

    float scale_Q[SEQ_LEN];
    float scale_K[SEQ_LEN];
    float scale_V[HEAD_DIM];

    #pragma HLS ARRAY_PARTITION variable=BRAM_Q \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_K \
        type=complete dim=2

    #pragma HLS ARRAY_PARTITION variable=BRAM_V_T \
        type=cyclic factor=SV_PACKS_PER_CYCLE dim=2

    #pragma HLS ARRAY_PARTITION variable=scale_V \
        type=cyclic factor=SV_CHANNELS_PER_CYCLE

    #pragma HLS BIND_STORAGE variable=BRAM_Q \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=BRAM_K \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=BRAM_V_T \
        type=ram_t2p impl=bram

    #pragma HLS BIND_STORAGE variable=scale_Q \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=scale_K \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=scale_V \
        type=ram_2p impl=bram

    consumer_head_loop:
    for (
        int head_idx = 0;
        head_idx < NUM_HEADS;
        head_idx++
    ) {
        #pragma HLS LOOP_FLATTEN off

        load_one_head_from_fifo(
            FIFO_Q_head,
            FIFO_K_head,
            FIFO_V_head,
            FIFO_scale_Q_head,
            FIFO_scale_K_head,
            FIFO_scale_V_head,
            BRAM_Q,
            BRAM_K,
            BRAM_V_T,
            scale_Q,
            scale_K,
            scale_V
        );

        consume_attention_one_head_dataflow_from_buffer(
            BRAM_Q,
            BRAM_K,
            BRAM_V_T,
            scale_Q,
            scale_K,
            scale_V,
            HBM_Attention_Out,
            global_max,
            head_idx
        );
    }
}

void process_all_heads_fifo_dataflow(
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
    float global_max[SEQ_LEN]
) {
    #pragma HLS DATAFLOW

    #pragma HLS ARRAY_PARTITION variable=X_package \
        type=cyclic factor=6 dim=2

    hls::stream<int64_dt>
        FIFO_Q_head("FIFO_Q_head");

    hls::stream<int64_dt>
        FIFO_K_head("FIFO_K_head");

    hls::stream<int64_dt>
        FIFO_V_head("FIFO_V_head");

    hls::stream<float>
        FIFO_scale_Q_head("FIFO_scale_Q_head");

    hls::stream<float>
        FIFO_scale_K_head("FIFO_scale_K_head");

    hls::stream<float>
        FIFO_scale_V_head("FIFO_scale_V_head");

    #pragma HLS STREAM variable=FIFO_Q_head \
        depth=QK_HEAD_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_K_head \
        depth=QK_HEAD_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_V_head \
        depth=V_HEAD_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_scale_Q_head \
        depth=QK_SCALE_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_scale_K_head \
        depth=QK_SCALE_FIFO_DEPTH

    #pragma HLS STREAM variable=FIFO_scale_V_head \
        depth=V_SCALE_FIFO_DEPTH

    #pragma HLS BIND_STORAGE variable=FIFO_Q_head \
        type=fifo impl=uram

    #pragma HLS BIND_STORAGE variable=FIFO_K_head \
        type=fifo impl=uram

    #pragma HLS BIND_STORAGE variable=FIFO_V_head \
        type=fifo impl=uram

    #pragma HLS BIND_STORAGE variable=FIFO_scale_Q_head \
        type=fifo impl=bram

    #pragma HLS BIND_STORAGE variable=FIFO_scale_K_head \
        type=fifo impl=bram

    #pragma HLS BIND_STORAGE variable=FIFO_scale_V_head \
        type=fifo impl=bram

    produce_all_heads_qkv_to_fifo(
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
        FIFO_Q_head,
        FIFO_K_head,
        FIFO_V_head,
        FIFO_scale_Q_head,
        FIFO_scale_K_head,
        FIFO_scale_V_head
    );

    consume_all_heads_attention_from_fifo(
        FIFO_Q_head,
        FIFO_K_head,
        FIFO_V_head,
        FIFO_scale_Q_head,
        FIFO_scale_K_head,
        FIFO_scale_V_head,
        HBM_Attention_Out,
        global_max
    );
}

void store_attention_max_HBM(
    float global_max[SEQ_LEN],
    float *HBM_Attention_Max
) {
    #pragma HLS INLINE off

    store_attention_max_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        HBM_Attention_Max[i] =
            global_max[i];
    }
}
