#include "mha_output_projection.h"
// version: 20260721_four_kernel_v0

//========================================
//============INPUT=======================
// uint512_dt *HBM_Attention_Out,
// float *HBM_Attention_Max,
// uint512_dt *Wo,
// uint512_dt *x,
// float *bo,
// float *scale_wo,
//============OUTPUT======================
// uint512_dt *out
//========================================

extern "C" {

void mha_output_projection(
    uint512_dt *HBM_Attention_Out,
    float *HBM_Attention_Max,
    uint512_dt *Wo,
    uint512_dt *x,
    float *bo,
    float *scale_wo,
    uint512_dt *out
) {
    #pragma HLS INTERFACE m_axi \
        port=HBM_Attention_Out \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9456 \
        max_widen_bitwidth=512 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=HBM_Attention_Max \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=197 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=Wo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9216 \
        max_widen_bitwidth=512 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=x \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9456 \
        max_widen_bitwidth=512 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=bo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=768 \
        max_widen_bitwidth=512 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=scale_wo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=768 \
        max_widen_bitwidth=512 \
        max_read_burst_length=64 \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi \
        port=out \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9456 \
        max_widen_bitwidth=512 \
        max_write_burst_length=64 \
        num_write_outstanding=16

    #pragma HLS INTERFACE s_axilite port=HBM_Attention_Out bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_Attention_Max bundle=control
    #pragma HLS INTERFACE s_axilite port=Wo                bundle=control
    #pragma HLS INTERFACE s_axilite port=x                 bundle=control
    #pragma HLS INTERFACE s_axilite port=bo                bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_wo          bundle=control
    #pragma HLS INTERFACE s_axilite port=out               bundle=control
    #pragma HLS INTERFACE s_axilite port=return            bundle=control

    run_full_output_projection(
        HBM_Attention_Out,
        HBM_Attention_Max,
        Wo,
        x,
        out,
        bo,
        scale_wo
    );
}

}

void load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
) {
    #pragma HLS INLINE off

    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] =
            HBM_b[i];
    }
}

void load_attention_max_HBM2BRAM(
    const float *HBM_Attention_Max,
    float BRAM_Attention_Max[SEQ_LEN]
) {
    #pragma HLS INLINE off

    load_attention_max_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_Attention_Max[i] =
            HBM_Attention_Max[i];
    }
}

void prepare_attention_quant_params(
    float attention_max[SEQ_LEN],
    float quant_scale[SEQ_LEN],
    float out_scaler[SEQ_LEN]
) {
    #pragma HLS INLINE off

    prepare_scale_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        float max_val =
            std::max(
                attention_max[i],
                EPSILON
            );

        quant_scale[i] =
            127.0f /
            max_val;

        out_scaler[i] =
            max_val *
            (1.0f / 127.0f);
    }
}

void load_quantize_attention_HBM2URAM(
    const uint512_dt *HBM_Attention_Out,
    float quant_scale[SEQ_LEN],
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
    ]
) {
    #pragma HLS INLINE off

    constexpr int FP32_WORDS_PER_INT8_WORD =
        INT8_PER_512 /
        FP32_PER_512;

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
                type=complete dim=1

            source_word_loop:
            for (
                int source_word_idx = 0;
                source_word_idx < FP32_WORDS_PER_INT8_WORD;
                source_word_idx++
            ) {
                int hbm_word_idx =
                    token * FP32_WORDS_PER_ROW +
                    quant_word_idx *
                        FP32_WORDS_PER_INT8_WORD +
                    source_word_idx;

                uint512_dt fp32_chunk =
                    HBM_Attention_Out[hbm_word_idx];

                quantize_half_loop:
                for (int half = 0; half < 2; half++) {
                    #pragma HLS PIPELINE II=1

                    int64_dt packed_int8 =
                        0;

                    quantize_lane_loop:
                    for (
                        int lane = 0;
                        lane < INT8_PACK;
                        lane++
                    ) {
                        #pragma HLS UNROLL

                        int fp32_lane =
                            half * INT8_PACK +
                            lane;

                        int fp32_start =
                            fp32_lane * 32;

                        int fp32_end =
                            fp32_start + 31;

                        union {
                            float f;
                            uint32_t i;
                        } converter;

                        converter.i =
                            static_cast<uint32_t>(
                                fp32_chunk.range(
                                    fp32_end,
                                    fp32_start
                                )
                            );

                        float scaled_val =
                            converter.f *
                            token_quant_scale;

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
                                std::round(
                                    clamped_val
                                )
                            );

                        packed_int8.range(
                            lane * 8 + 7,
                            lane * 8
                        ) = q_val;
                    }

                    int package_idx =
                        source_word_idx * 2 +
                        half;

                    quant_packages[package_idx] =
                        packed_int8;
                }
            }

            uint512_dt quantized_word =
                0;

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
                token * OPROJ_INT8_WORDS_PER_ROW +
                quant_word_idx;

            URAM_Attention_Q[uram_addr] =
                quantized_word;
        }
    }
}

void load_Weight_tile_HBM2FIFO(
    const uint512_dt *HBM_Wo,
    hls::stream<uint512_dt> &FIFO_Wo,
    int iter
) {
    constexpr int CHUNKS_PER_ROW =
        EMBED_DIM /
        INT8_PER_512;

    int row_offset =
        iter *
        TILE_SIZE;

    int base_addr_idx =
        row_offset *
        CHUNKS_PER_ROW;

    load_tile_row:
    for (int i = 0; i < TILE_SIZE; i++) {
        int current_row_start =
            base_addr_idx +
            i * CHUNKS_PER_ROW;

        load_tile_col:
        for (int j = 0; j < CHUNKS_PER_ROW; j++) {
            #pragma HLS PIPELINE II=1

            FIFO_Wo.write(
                HBM_Wo[
                    current_row_start + j
                ]
            );
        }
    }
}

void load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X_Res,
    hls::stream<uint512_dt> &FIFO_Res,
    int iter
) {
    int tile_col_offset =
        iter *
        FP32_WORDS_PER_TILE;

    load_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start =
            i *
            FP32_WORDS_PER_ROW;

        load_residual_tile_loop:
        for (
            int j = 0;
            j < FP32_WORDS_PER_TILE;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            int target_idx =
                row_start +
                tile_col_offset +
                j;

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
    int tile_col_offset =
        iter *
        FP32_WORDS_PER_TILE;

    store_seq_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        int row_start =
            i *
            FP32_WORDS_PER_ROW;

        store_tile_loop:
        for (
            int j = 0;
            j < FP32_WORDS_PER_TILE;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            int target_idx =
                row_start +
                tile_col_offset +
                j;

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
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_1 \
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_2 \
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_3 \
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_4 \
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_5 \
        type=complete dim=1

    #pragma HLS ARRAY_PARTITION variable=level_6 \
        type=complete dim=1

    multiply_word_0_loop:
    for (int idx = 0; idx < INT8_PER_512; idx++) {
        #pragma HLS UNROLL

        int start_bit =
            idx * 8;

        int end_bit =
            start_bit + 7;

        int8_dt input_val =
            static_cast<int8_dt>(
                input_word_0.range(
                    end_bit,
                    start_bit
                )
            );

        int8_dt weight_val =
            static_cast<int8_dt>(
                weight_word_0.range(
                    end_bit,
                    start_bit
                )
            );

        product[idx] =
            input_val *
            weight_val;
    }

    multiply_word_1_loop:
    for (int idx = 0; idx < INT8_PER_512; idx++) {
        #pragma HLS UNROLL

        int start_bit =
            idx * 8;

        int end_bit =
            start_bit + 7;

        int8_dt input_val =
            static_cast<int8_dt>(
                input_word_1.range(
                    end_bit,
                    start_bit
                )
            );

        int8_dt weight_val =
            static_cast<int8_dt>(
                weight_word_1.range(
                    end_bit,
                    start_bit
                )
            );

        product[
            INT8_PER_512 + idx
        ] =
            input_val *
            weight_val;
    }

    reduce_level_1:
    for (int idx = 0; idx < 64; idx++) {
        #pragma HLS UNROLL

        level_1[idx] =
            product[idx * 2] +
            product[idx * 2 + 1];
    }

    reduce_level_2:
    for (int idx = 0; idx < 32; idx++) {
        #pragma HLS UNROLL

        level_2[idx] =
            level_1[idx * 2] +
            level_1[idx * 2 + 1];
    }

    reduce_level_3:
    for (int idx = 0; idx < 16; idx++) {
        #pragma HLS UNROLL

        level_3[idx] =
            level_2[idx * 2] +
            level_2[idx * 2 + 1];
    }

    reduce_level_4:
    for (int idx = 0; idx < 8; idx++) {
        #pragma HLS UNROLL

        level_4[idx] =
            level_3[idx * 2] +
            level_3[idx * 2 + 1];
    }

    reduce_level_5:
    for (int idx = 0; idx < 4; idx++) {
        #pragma HLS UNROLL

        level_5[idx] =
            level_4[idx * 2] +
            level_4[idx * 2 + 1];
    }

    reduce_level_6:
    for (int idx = 0; idx < 2; idx++) {
        #pragma HLS UNROLL

        level_6[idx] =
            level_5[idx * 2] +
            level_5[idx * 2 + 1];
    }

    return
        level_6[0] +
        level_6[1];
}

void compute_output_proj_and_residual(
    uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
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
#else
    static uint512_dt URAM_Wo
        [TILE_SIZE]
        [OPROJ_INT8_WORDS_PER_ROW];
#endif

    #pragma HLS BIND_STORAGE variable=URAM_Wo \
        type=ram_2p impl=uram

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
            i *
            OPROJ_INT8_WORDS_PER_ROW;

        compute_tile:
        for (int j = 0; j < TILE_SIZE; j++) {
            int32_dt acc =
                0;

            dot_product_step:
            for (
                int step = 0;
                step < OPROJ_DOT_STEPS;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int word_offset =
                    step *
                    OPROJ_WORDS_PER_STEP;

                int attention_addr_0 =
                    attention_row_base +
                    word_offset;

                int attention_addr_1 =
                    attention_addr_0 +
                    1;

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

                if (step == 0) {
                    acc =
                        cycle_acc;
                } else {
                    acc =
                        acc +
                        cycle_acc;
                }

                if (
                    step ==
                    OPROJ_DOT_STEPS - 1
                ) {
                    acc_buffer[j] =
                        acc;
                }
            }
        }

        float token_scaler =
            out_scaler[i];

        uint512_dt res_chunk =
            0;

        float pack_array[FP32_PER_512];

        #pragma HLS ARRAY_PARTITION variable=pack_array \
            type=complete dim=1

        post_process_loop:
        for (int j = 0; j < TILE_SIZE; j++) {
            #pragma HLS PIPELINE II=1

            int pack_lane =
                j %
                FP32_PER_512;

            if (pack_lane == 0) {
                res_chunk =
                    FIFO_Res.read();
            }

            int32_dt final_acc =
                acc_buffer[j];

            int weight_idx =
                iter * TILE_SIZE +
                j;

            float combined_scale =
                token_scaler *
                scale_wo[weight_idx];

            float proj_val =
                static_cast<float>(
                    final_acc
                ) *
                combined_scale +
                BRAM_bo[weight_idx];

            int res_start_bit =
                pack_lane * 32;

            int res_end_bit =
                res_start_bit + 31;

            union {
                float f;
                uint32_t i;
            } res_converter;

            res_converter.i =
                static_cast<uint32_t>(
                    res_chunk.range(
                        res_end_bit,
                        res_start_bit
                    )
                );

            pack_array[pack_lane] =
                proj_val +
                res_converter.f;

            if (
                pack_lane ==
                FP32_PER_512 - 1
            ) {
                uint512_dt out_chunk =
                    0;

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
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
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
        OPROJ_ATTENTION_TOTAL_WORDS
    ],
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
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
    const uint512_dt *HBM_Attention_Out,
    const float *HBM_Attention_Max,
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    const float *HBM_bo,
    const float *HBM_scale_wo
) {
    #pragma HLS INLINE off

    float SRAM_bo[EMBED_DIM];
    float SRAM_scale_wo[EMBED_DIM];
    float SRAM_Attention_Max[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=SRAM_bo \
        type=cyclic factor=8 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_wo \
        type=cyclic factor=8 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_bo \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_scale_wo \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_Attention_Max \
        type=ram_2p impl=bram

    float quant_scale[SEQ_LEN];
    float out_scaler[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=quant_scale \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=out_scaler \
        type=ram_2p impl=bram

    static uint512_dt URAM_Attention_Q[
        OPROJ_ATTENTION_TOTAL_WORDS
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

    load_attention_max_HBM2BRAM(
        HBM_Attention_Max,
        SRAM_Attention_Max
    );

    prepare_attention_quant_params(
        SRAM_Attention_Max,
        quant_scale,
        out_scaler
    );

    load_quantize_attention_HBM2URAM(
        HBM_Attention_Out,
        quant_scale,
        URAM_Attention_Q
    );

    constexpr int NUM_TILES =
        EMBED_DIM /
        TILE_SIZE;

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
