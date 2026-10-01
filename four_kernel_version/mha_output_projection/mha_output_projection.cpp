#include "mha_output_projection.h"
// version: 20260726_timing_fix_no_bundle_split

#ifndef OPROJ_REDUCED_OUTPUT_ENGINES
#define OPROJ_REDUCED_OUTPUT_ENGINES 2
#endif

// Force the 8-bit multipliers in the dot-product engines onto DSP blocks.
// Set to 0 if DSP usage is more important than LUT congestion/timing.
#ifndef OPROJ_FORCE_INT8_DSP
#define OPROJ_FORCE_INT8_DSP 1
#endif

static constexpr int OPROJ_REDUCED_MAC_LANES = 32;
static constexpr int OPROJ_REDUCED_VEC_BITS =
    OPROJ_REDUCED_MAC_LANES * 8;

static constexpr int OPROJ_REDUCED_WORDS_PER_ROW =
    EMBED_DIM / OPROJ_REDUCED_MAC_LANES;

static constexpr int OPROJ_REDUCED_WORDS_PER_512 =
    INT8_PER_512 / OPROJ_REDUCED_MAC_LANES;

static constexpr int OPROJ_REDUCED_ATTENTION_TOTAL_WORDS =
    SEQ_LEN * OPROJ_REDUCED_WORDS_PER_ROW;

typedef ap_uint<OPROJ_REDUCED_VEC_BITS> oproj_int8_vec_t;

static constexpr int OPROJ_OUTPUT_STREAMS = 4;
static constexpr int OPROJ_OUTPUT_STREAM_BITS = 128;
static constexpr int OPROJ_FP32_PER_OUTPUT_STREAM =
    OPROJ_OUTPUT_STREAM_BITS / 32;

typedef ap_uint<OPROJ_OUTPUT_STREAM_BITS> oproj_output_stream_t;

static_assert(
    EMBED_DIM % OPROJ_REDUCED_MAC_LANES == 0,
    "EMBED_DIM must be divisible by OPROJ_REDUCED_MAC_LANES"
);

static_assert(
    INT8_PER_512 % OPROJ_REDUCED_MAC_LANES == 0,
    "INT8_PER_512 must be divisible by OPROJ_REDUCED_MAC_LANES"
);

static_assert(
    TILE_SIZE % OPROJ_REDUCED_OUTPUT_ENGINES == 0,
    "TILE_SIZE must be divisible by OPROJ_REDUCED_OUTPUT_ENGINES"
);

static_assert(
    OPROJ_REDUCED_OUTPUT_ENGINES == 1 ||
    OPROJ_REDUCED_OUTPUT_ENGINES == 2,
    "This implementation supports one or two output engines"
);

static_assert(
    OPROJ_OUTPUT_STREAMS * OPROJ_OUTPUT_STREAM_BITS == 512,
    "Output streams must reconstruct one 512-bit AXI word"
);

static_assert(
    OPROJ_OUTPUT_STREAMS * OPROJ_FP32_PER_OUTPUT_STREAM ==
        FP32_PER_512,
    "Output stream packing does not match FP32_PER_512"
);

void run_full_output_projection(
    const uint512_dt *HBM_Attention_Out,
    const float *HBM_Attention_Max,
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    const float *HBM_bo,
    const float *HBM_scale_wo
);

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
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=HBM_Attention_Max \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=197 \
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=Wo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9216 \
        max_widen_bitwidth=512 \
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=x \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9456 \
        max_widen_bitwidth=512 \
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=bo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=768 \
        max_widen_bitwidth=512 \
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=scale_wo \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=768 \
        max_widen_bitwidth=512 \
        max_read_burst_length=16 \
        num_read_outstanding=2

    #pragma HLS INTERFACE m_axi \
        port=out \
        offset=slave \
        bundle=gmem_mha_output_projection \
        depth=9456 \
        max_widen_bitwidth=512 \
        max_write_burst_length=16 \
        num_write_outstanding=2

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
    // Inlined to remove a child-module handshake and parent-level
    // BRAM ownership multiplexer.
    #pragma HLS INLINE

    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] = HBM_b[i];
    }
}

void load_attention_max_HBM2BRAM(
    const float *HBM_Attention_Max,
    float BRAM_Attention_Max[SEQ_LEN]
) {
    // Inlined to remove a child-module handshake and parent-level
    // BRAM ownership multiplexer.
    #pragma HLS INLINE

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
    // Inlined because it is a small sequential initialization stage.
    #pragma HLS INLINE

    prepare_scale_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=4

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

static int8_dt quantize_fp32_to_int8_reduced(
    float input_val,
    float quant_scale
) {
    #pragma HLS INLINE

    float scaled_val =
        input_val *
        quant_scale;

    float clamped_val =
        std::max(
            -127.0f,
            std::min(
                127.0f,
                scaled_val
            )
        );

    float rounded_val =
        (clamped_val >= 0.0f)
            ? (clamped_val + 0.5f)
            : (clamped_val - 0.5f);

    int rounded_int =
        static_cast<int>(
            rounded_val
        );

    return
        static_cast<int8_dt>(
            rounded_int
        );
}

void load_quantize_attention_HBM2URAM(
    const uint512_dt *HBM_Attention_Out,
    float quant_scale[SEQ_LEN],
    oproj_int8_vec_t URAM_Attention_Q[
        OPROJ_REDUCED_ATTENTION_TOTAL_WORDS
    ]
) {
    #pragma HLS INLINE off

    constexpr int FP32_WORDS_PER_REDUCED_WORD =
        OPROJ_REDUCED_MAC_LANES /
        FP32_PER_512;

    token_loop:
    for (int token = 0; token < SEQ_LEN; token++) {
        float token_quant_scale =
            quant_scale[token];

        reduced_word_loop:
        for (
            int reduced_word_idx = 0;
            reduced_word_idx <
                OPROJ_REDUCED_WORDS_PER_ROW;
            reduced_word_idx++
        ) {
            int8_dt quantized_lane[
                OPROJ_REDUCED_MAC_LANES
            ];

            #pragma HLS ARRAY_PARTITION \
                variable=quantized_lane \
                type=complete dim=1

            source_word_loop:
            for (
                int source_word_idx = 0;
                source_word_idx <
                    FP32_WORDS_PER_REDUCED_WORD;
                source_word_idx++
            ) {
                int hbm_word_idx =
                    token *
                        FP32_WORDS_PER_ROW +
                    reduced_word_idx *
                        FP32_WORDS_PER_REDUCED_WORD +
                    source_word_idx;

                uint512_dt fp32_chunk =
                    HBM_Attention_Out[
                        hbm_word_idx
                    ];

                quantize_lane_loop:
                for (
                    int lane = 0;
                    lane < FP32_PER_512;
                    lane++
                ) {
                    #pragma HLS PIPELINE II=1

                    int fp32_start =
                        lane * 32;

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

                    int target_lane =
                        source_word_idx *
                            FP32_PER_512 +
                        lane;

                    quantized_lane[target_lane] =
                        quantize_fp32_to_int8_reduced(
                            converter.f,
                            token_quant_scale
                        );
                }
            }

            oproj_int8_vec_t quantized_word =
                0;

            pack_reduced_word_loop:
            for (
                int lane = 0;
                lane <
                    OPROJ_REDUCED_MAC_LANES;
                lane++
            ) {
                #pragma HLS UNROLL

                int start_bit =
                    lane * 8;

                int end_bit =
                    start_bit + 7;

                quantized_word.range(
                    end_bit,
                    start_bit
                ) =
                    quantized_lane[lane];
            }

            int uram_addr =
                token *
                    OPROJ_REDUCED_WORDS_PER_ROW +
                reduced_word_idx;

            URAM_Attention_Q[
                uram_addr
            ] =
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
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_0,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_1,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_2,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_3,
    uint512_dt *HBM_Out,
    int iter
) {
    #pragma HLS INLINE off

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

            oproj_output_stream_t output_slice_0 =
                FIFO_Final_Out_0.read();

            oproj_output_stream_t output_slice_1 =
                FIFO_Final_Out_1.read();

            oproj_output_stream_t output_slice_2 =
                FIFO_Final_Out_2.read();

            oproj_output_stream_t output_slice_3 =
                FIFO_Final_Out_3.read();

            uint512_dt output_word =
                0;

            output_word.range(127, 0) =
                output_slice_0;

            output_word.range(255, 128) =
                output_slice_1;

            output_word.range(383, 256) =
                output_slice_2;

            output_word.range(511, 384) =
                output_slice_3;

            int target_idx =
                row_start +
                tile_col_offset +
                j;

            HBM_Out[target_idx] =
                output_word;
        }
    }
}

static ap_int<16> multiply_int8_timing(
    int8_dt input_val,
    int8_dt weight_val
) {
    #pragma HLS INLINE

    ap_int<16> result =
        input_val *
        weight_val;

#if OPROJ_FORCE_INT8_DSP
    // A one-stage DSP multiply reduces LUT congestion without changing
    // the number of MAC lanes or the requested outer-loop II.
    #pragma HLS BIND_OP variable=result op=mul impl=dsp latency=1
#endif

    return result;
}

static int32_dt dot_product_32_int8(
    oproj_int8_vec_t input_word,
    oproj_int8_vec_t weight_word
) {
    #pragma HLS INLINE

    ap_int<16> product[32];

    ap_int<17> level_1[16];
    ap_int<18> level_2[8];
    ap_int<19> level_3[4];
    ap_int<20> level_4[2];
    ap_int<21> level_5;

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

    multiply_loop:
    for (
        int idx = 0;
        idx < OPROJ_REDUCED_MAC_LANES;
        idx++
    ) {
        #pragma HLS UNROLL

        int start_bit =
            idx * 8;

        int end_bit =
            start_bit + 7;

        int8_dt input_val =
            static_cast<int8_dt>(
                input_word.range(
                    end_bit,
                    start_bit
                )
            );

        int8_dt weight_val =
            static_cast<int8_dt>(
                weight_word.range(
                    end_bit,
                    start_bit
                )
            );

        product[idx] =
            multiply_int8_timing(
                input_val,
                weight_val
            );
    }

    reduce_level_1:
    for (int idx = 0; idx < 16; idx++) {
        #pragma HLS UNROLL

        level_1[idx] =
            product[idx * 2] +
            product[idx * 2 + 1];
    }

    reduce_level_2:
    for (int idx = 0; idx < 8; idx++) {
        #pragma HLS UNROLL

        level_2[idx] =
            level_1[idx * 2] +
            level_1[idx * 2 + 1];
    }

    reduce_level_3:
    for (int idx = 0; idx < 4; idx++) {
        #pragma HLS UNROLL

        level_3[idx] =
            level_2[idx * 2] +
            level_2[idx * 2 + 1];
    }

    reduce_level_4:
    for (int idx = 0; idx < 2; idx++) {
        #pragma HLS UNROLL

        level_4[idx] =
            level_3[idx * 2] +
            level_3[idx * 2 + 1];
    }

    level_5 =
        level_4[0] +
        level_4[1];

    return
        static_cast<int32_dt>(
            level_5
        );
}

void compute_output_proj_and_residual(
    oproj_int8_vec_t URAM_Attention_Q[
        OPROJ_REDUCED_ATTENTION_TOTAL_WORDS
    ],
    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_0,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_1,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_2,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_3,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM],
    int iter
) {
    #pragma HLS INLINE off

#ifdef __SYNTHESIS__
    oproj_int8_vec_t URAM_Wo
        [TILE_SIZE]
        [OPROJ_REDUCED_WORDS_PER_ROW];
#else
    static oproj_int8_vec_t URAM_Wo
        [TILE_SIZE]
        [OPROJ_REDUCED_WORDS_PER_ROW];
#endif

    #pragma HLS ARRAY_PARTITION variable=URAM_Wo \
        type=cyclic factor=OPROJ_REDUCED_OUTPUT_ENGINES dim=1

    #pragma HLS BIND_STORAGE variable=URAM_Wo \
        type=ram_1p impl=uram

    constexpr int HBM_WEIGHT_WORDS_PER_ROW =
        EMBED_DIM /
        INT8_PER_512;

    load_wo_row:
    for (int row = 0; row < TILE_SIZE; row++) {
        load_wo_col:
        for (
            int hbm_word_idx = 0;
            hbm_word_idx <
                HBM_WEIGHT_WORDS_PER_ROW;
            hbm_word_idx++
        ) {
            #pragma HLS PIPELINE II=2

            uint512_dt packed_weight =
                FIFO_Wo.read();

            int reduced_word_base =
                hbm_word_idx *
                OPROJ_REDUCED_WORDS_PER_512;

            URAM_Wo[row][
                reduced_word_base
            ] =
                packed_weight.range(
                    255,
                    0
                );

            URAM_Wo[row][
                reduced_word_base + 1
            ] =
                packed_weight.range(
                    511,
                    256
                );
        }
    }

    compute_seq:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS LOOP_FLATTEN off

        int32_dt acc_buffer[TILE_SIZE];

        #pragma HLS ARRAY_PARTITION variable=acc_buffer \
            type=cyclic factor=OPROJ_REDUCED_OUTPUT_ENGINES dim=1

        #pragma HLS BIND_STORAGE variable=acc_buffer \
            type=ram_1p impl=lutram

        int attention_row_base =
            i *
            OPROJ_REDUCED_WORDS_PER_ROW;

        compute_tile:
        for (
            int j_base = 0;
            j_base < TILE_SIZE;
            j_base += OPROJ_REDUCED_OUTPUT_ENGINES
        ) {
            int32_dt acc[
                OPROJ_REDUCED_OUTPUT_ENGINES
            ];

            #pragma HLS ARRAY_PARTITION variable=acc \
                type=complete dim=1

            initialize_acc_loop:
            for (
                int engine = 0;
                engine <
                    OPROJ_REDUCED_OUTPUT_ENGINES;
                engine++
            ) {
                #pragma HLS UNROLL

                acc[engine] =
                    0;
            }

            dot_product_step:
            for (
                int step = 0;
                step <
                    OPROJ_REDUCED_WORDS_PER_ROW;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                oproj_int8_vec_t input_word =
                    URAM_Attention_Q[
                        attention_row_base +
                        step
                    ];

                output_engine_loop:
                for (
                    int engine = 0;
                    engine <
                        OPROJ_REDUCED_OUTPUT_ENGINES;
                    engine++
                ) {
                    #pragma HLS UNROLL

                    oproj_int8_vec_t weight_word =
                        URAM_Wo[
                            j_base + engine
                        ][step];

                    int32_dt cycle_acc =
                        dot_product_32_int8(
                            input_word,
                            weight_word
                        );

                    acc[engine] +=
                        cycle_acc;
                }
            }

            store_acc_loop:
            for (
                int engine = 0;
                engine <
                    OPROJ_REDUCED_OUTPUT_ENGINES;
                engine++
            ) {
                #pragma HLS UNROLL

                acc_buffer[
                    j_base + engine
                ] =
                    acc[engine];
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
            #pragma HLS PIPELINE II=2

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
                iter *
                    TILE_SIZE +
                j;

            float combined_scale =
                token_scaler *
                scale_wo[weight_idx];

            float proj_mul =
                static_cast<float>(
                    final_acc
                ) *
                combined_scale;

            float proj_val =
                proj_mul +
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
                oproj_output_stream_t output_slice[
                    OPROJ_OUTPUT_STREAMS
                ];

                #pragma HLS ARRAY_PARTITION \
                    variable=output_slice \
                    type=complete dim=1

                initialize_output_slice_loop:
                for (
                    int stream_idx = 0;
                    stream_idx < OPROJ_OUTPUT_STREAMS;
                    stream_idx++
                ) {
                    #pragma HLS UNROLL

                    output_slice[stream_idx] =
                        0;
                }

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

                    int stream_idx =
                        lane /
                        OPROJ_FP32_PER_OUTPUT_STREAM;

                    int stream_lane =
                        lane %
                        OPROJ_FP32_PER_OUTPUT_STREAM;

                    int out_start =
                        stream_lane * 32;

                    int out_end =
                        out_start + 31;

                    output_slice[
                        stream_idx
                    ].range(
                        out_end,
                        out_start
                    ) =
                        out_converter.i;
                }

                FIFO_Final_Out_0.write(
                    output_slice[0]
                );

                FIFO_Final_Out_1.write(
                    output_slice[1]
                );

                FIFO_Final_Out_2.write(
                    output_slice[2]
                );

                FIFO_Final_Out_3.write(
                    output_slice[3]
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

void load_all_tiles_wo_residual_sequential(
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res
) {
    #pragma HLS INLINE off

    constexpr int NUM_TILES =
        EMBED_DIM /
        TILE_SIZE;

    load_all_tiles_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        #pragma HLS LOOP_FLATTEN off

        load_wo_residual_sequential(
            HBM_Wo,
            HBM_X_Res,
            FIFO_Wo,
            FIFO_Res,
            iter
        );
    }
}

void compute_all_tiles_output_proj_and_residual(
    oproj_int8_vec_t URAM_Attention_Q[
        OPROJ_REDUCED_ATTENTION_TOTAL_WORDS
    ],
    hls::stream<uint512_dt> &FIFO_Wo,
    hls::stream<uint512_dt> &FIFO_Res,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_0,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_1,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_2,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_3,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM]
) {
    #pragma HLS INLINE off

    constexpr int NUM_TILES =
        EMBED_DIM /
        TILE_SIZE;

    compute_all_tiles_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        #pragma HLS LOOP_FLATTEN off

        compute_output_proj_and_residual(
            URAM_Attention_Q,
            FIFO_Wo,
            FIFO_Res,
            FIFO_Final_Out_0,
            FIFO_Final_Out_1,
            FIFO_Final_Out_2,
            FIFO_Final_Out_3,
            out_scaler,
            BRAM_bo,
            scale_wo,
            iter
        );
    }
}

void store_all_tiles_Final_FIFO2HBM(
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_0,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_1,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_2,
    hls::stream<oproj_output_stream_t> &FIFO_Final_Out_3,
    uint512_dt *HBM_Out
) {
    #pragma HLS INLINE off

    constexpr int NUM_TILES =
        EMBED_DIM /
        TILE_SIZE;

    store_all_tiles_loop:
    for (int iter = 0; iter < NUM_TILES; iter++) {
        #pragma HLS LOOP_FLATTEN off

        store_Final_Tile_FIFO2HBM(
            FIFO_Final_Out_0,
            FIFO_Final_Out_1,
            FIFO_Final_Out_2,
            FIFO_Final_Out_3,
            HBM_Out,
            iter
        );
    }
}

void process_all_tiles_persistent_dataflow(
    oproj_int8_vec_t URAM_Attention_Q[
        OPROJ_REDUCED_ATTENTION_TOTAL_WORDS
    ],
    const uint512_dt *HBM_Wo,
    const uint512_dt *HBM_X_Res,
    uint512_dt *HBM_Out,
    float out_scaler[SEQ_LEN],
    float BRAM_bo[EMBED_DIM],
    float scale_wo[EMBED_DIM]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint512_dt>
        FIFO_Wo("FIFO_Wo_all_tiles");

    hls::stream<uint512_dt>
        FIFO_Res("FIFO_Res_all_tiles");

    hls::stream<oproj_output_stream_t>
        FIFO_Final_Out_0("FIFO_Final_Out_0_all_tiles");

    hls::stream<oproj_output_stream_t>
        FIFO_Final_Out_1("FIFO_Final_Out_1_all_tiles");

    hls::stream<oproj_output_stream_t>
        FIFO_Final_Out_2("FIFO_Final_Out_2_all_tiles");

    hls::stream<oproj_output_stream_t>
        FIFO_Final_Out_3("FIFO_Final_Out_3_all_tiles");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Wo depth=32
    #pragma HLS BIND_STORAGE variable=FIFO_Wo \
        type=fifo impl=bram

    #pragma HLS STREAM variable=FIFO_Res depth=32
    #pragma HLS BIND_STORAGE variable=FIFO_Res \
        type=fifo impl=bram

    #pragma HLS STREAM variable=FIFO_Final_Out_0 depth=4
    #pragma HLS BIND_STORAGE variable=FIFO_Final_Out_0 \
        type=fifo impl=bram

    #pragma HLS STREAM variable=FIFO_Final_Out_1 depth=4
    #pragma HLS BIND_STORAGE variable=FIFO_Final_Out_1 \
        type=fifo impl=bram

    #pragma HLS STREAM variable=FIFO_Final_Out_2 depth=4
    #pragma HLS BIND_STORAGE variable=FIFO_Final_Out_2 \
        type=fifo impl=bram

    #pragma HLS STREAM variable=FIFO_Final_Out_3 depth=4
    #pragma HLS BIND_STORAGE variable=FIFO_Final_Out_3 \
        type=fifo impl=bram
#endif

    #pragma HLS stable variable=URAM_Attention_Q
    #pragma HLS stable variable=out_scaler
    #pragma HLS stable variable=BRAM_bo
    #pragma HLS stable variable=scale_wo

    load_all_tiles_wo_residual_sequential(
        HBM_Wo,
        HBM_X_Res,
        FIFO_Wo,
        FIFO_Res
    );

    compute_all_tiles_output_proj_and_residual(
        URAM_Attention_Q,
        FIFO_Wo,
        FIFO_Res,
        FIFO_Final_Out_0,
        FIFO_Final_Out_1,
        FIFO_Final_Out_2,
        FIFO_Final_Out_3,
        out_scaler,
        BRAM_bo,
        scale_wo
    );

    store_all_tiles_Final_FIFO2HBM(
        FIFO_Final_Out_0,
        FIFO_Final_Out_1,
        FIFO_Final_Out_2,
        FIFO_Final_Out_3,
        HBM_Out
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
        type=cyclic factor=2 dim=1

    #pragma HLS ARRAY_PARTITION variable=SRAM_scale_wo \
        type=cyclic factor=2 dim=1

    #pragma HLS BIND_STORAGE variable=SRAM_bo \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_scale_wo \
        type=ram_2p impl=bram

    #pragma HLS BIND_STORAGE variable=SRAM_Attention_Max \
        type=ram_1p impl=bram

    float quant_scale[SEQ_LEN];
    float out_scaler[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=quant_scale \
        type=ram_1p impl=bram

    #pragma HLS BIND_STORAGE variable=out_scaler \
        type=ram_1p impl=bram

    static oproj_int8_vec_t URAM_Attention_Q[
        OPROJ_REDUCED_ATTENTION_TOTAL_WORDS
    ];

    // Dual-port URAM separates the initialization writer from the
    // output-projection reader and reduces parent-level port multiplexing.
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

    process_all_tiles_persistent_dataflow(
        URAM_Attention_Q,
        HBM_Wo,
        HBM_X_Res,
        HBM_Out,
        out_scaler,
        SRAM_bo,
        SRAM_scale_wo
    );
}
