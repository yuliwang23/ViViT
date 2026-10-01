#include "ffn2.h"
// version: 20260721_four_kernel_v0
//========================================
//============INPUT=======================
// uint512_dt *HBM_FP32_Intermediate,
// float *HBM_GELU_Max,
// uint512_dt *W2,
// uint512_dt *x,
// float *b2,
// float *scale_w2,
//============OUTPUT======================
// uint512_dt *out
//========================================
extern "C" {

void FFN2(
    uint512_dt *HBM_FP32_Intermediate,
    float *HBM_GELU_Max,
    uint512_dt *W2,
    uint512_dt *x,
    float *b2,
    float *scale_w2,
    uint512_dt *out
) {
    #pragma HLS INTERFACE m_axi port=HBM_FP32_Intermediate     \
        offset=slave bundle=gmem_ffn2                         \
        depth=37824 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=HBM_GELU_Max              \
        offset=slave bundle=gmem_ffn2                         \
        depth=197 max_widen_bitwidth=512                      \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=W2                        \
        offset=slave bundle=gmem_ffn2                         \
        depth=36864 max_widen_bitwidth=512                    \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=x                         \
        offset=slave bundle=gmem_ffn2                         \
        depth=9456 max_widen_bitwidth=512                     \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=b2                        \
        offset=slave bundle=gmem_ffn2                         \
        depth=768 max_widen_bitwidth=512                      \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=scale_w2                  \
        offset=slave bundle=gmem_ffn2                         \
        depth=768 max_widen_bitwidth=512                      \
        max_read_burst_length=64                              \
        num_read_outstanding=16

    #pragma HLS INTERFACE m_axi port=out                       \
        offset=slave bundle=gmem_ffn2                         \
        depth=9456 max_widen_bitwidth=512                     \
        max_write_burst_length=64                             \
        num_write_outstanding=16

    #pragma HLS INTERFACE s_axilite port=HBM_FP32_Intermediate bundle=control
    #pragma HLS INTERFACE s_axilite port=HBM_GELU_Max          bundle=control
    #pragma HLS INTERFACE s_axilite port=W2                    bundle=control
    #pragma HLS INTERFACE s_axilite port=x                     bundle=control
    #pragma HLS INTERFACE s_axilite port=b2                    bundle=control
    #pragma HLS INTERFACE s_axilite port=scale_w2              bundle=control
    #pragma HLS INTERFACE s_axilite port=out                   bundle=control
    #pragma HLS INTERFACE s_axilite port=return                bundle=control

    float global_max[SEQ_LEN];

    #pragma HLS BIND_STORAGE variable=global_max \
        type=ram_2p impl=bram

    static int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package];

    float scaler_MLP1[SEQ_LEN];

    #pragma HLS ARRAY_PARTITION variable=URAM_MLP1_Quant_Out \
        type=cyclic factor=8 dim=2

    #pragma HLS BIND_STORAGE variable=URAM_MLP1_Quant_Out \
        type=ram_2p impl=uram

    #pragma HLS BIND_STORAGE variable=scaler_MLP1 \
        type=ram_2p impl=bram

    ffn2_load_GELU_max_HBM2BRAM(
        HBM_GELU_Max,
        global_max
    );

    ffn2_run_quantize_mlp1_dataflow(
        HBM_FP32_Intermediate,
        global_max,
        URAM_MLP1_Quant_Out,
        scaler_MLP1
    );

    ffn2_run_full_mlp_layer2(
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

static void ffn2_load_GELU_max_HBM2BRAM(
    const float *HBM_GELU_Max,
    float global_max[SEQ_LEN]
) {
    #pragma HLS INLINE off

    load_gelu_max_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        #pragma HLS PIPELINE II=1

        global_max[i] =
            HBM_GELU_Max[i];
    }
}

static void ffn2_load_FP32_HBM2FIFO(
    const uint512_dt *HBM_FP32_Intermediate,
    hls::stream<uint256_dt> &FIFO_Quant_in
) {
    #pragma HLS INLINE off

    const int TOTAL_WORDS =
        (SEQ_LEN * MLP_DIM) /
        16;

    load_hbm_loop:
    for (
        int i = 0;
        i < TOTAL_WORDS;
        i++
    ) {
        #pragma HLS PIPELINE II=2

        uint512_dt hbm_word =
            HBM_FP32_Intermediate[i];

        FIFO_Quant_in.write(
            hbm_word.range(
                255,
                0
            )
        );

        FIFO_Quant_in.write(
            hbm_word.range(
                511,
                256
            )
        );
    }
}

static void ffn2_quantize_mlp1_output_from_FIFO(
    hls::stream<uint256_dt> &FIFO_Quant_in,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS INLINE off

    tile_quant_loop:
    for (
        int iter = 0;
        iter < MLP1_NUM_TILES;
        iter++
    ) {
        row_quant_loop:
        for (
            int i = 0;
            i < SEQ_LEN;
            i++
        ) {
            #pragma HLS LOOP_FLATTEN off

            float max_val =
                std::max(
                    global_max[i],
                    EPSILON
                );

            float scale =
                127.0f /
                max_val;

            if (iter == 0) {
                scaler_MLP1[i] =
                    max_val /
                    127.0f;
            }

            col_quant_loop:
            for (
                int j = 0;
                j < TILE_OUT_DIM / INT8_PACK;
                j++
            ) {
                #pragma HLS PIPELINE II=2

                uint256_dt fp32_pack =
                    FIFO_Quant_in.read();

                int64_dt packed_int8 =
                    0;

                unpack_and_quantize_loop:
                for (int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL factor=4

                    const int in_start_bit =
                        k * 32;

                    const int in_end_bit =
                        in_start_bit +
                        31;

                    uint32_t raw_bits =
                        fp32_pack.range(
                            in_end_bit,
                            in_start_bit
                        );

                    union {
                        uint32_t i;
                        float f;
                    } conv;

                    conv.i =
                        raw_bits;

                    float scaled_val =
                        conv.f *
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

                    const int out_start_bit =
                        k * 8;

                    const int out_end_bit =
                        out_start_bit +
                        7;

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

static void ffn2_run_quantize_mlp1_dataflow(
    const uint512_dt *HBM_FP32_Intermediate,
    float global_max[SEQ_LEN],
    int64_dt URAM_MLP1_Quant_Out
        [SEQ_LEN][MLP_DIM_package],
    float scaler_MLP1[SEQ_LEN]
) {
    #pragma HLS DATAFLOW
    #pragma HLS INLINE off

    hls::stream<uint256_dt>
        FIFO_Quant_in("FIFO_Quant_in");

#ifdef __SYNTHESIS__
    #pragma HLS STREAM variable=FIFO_Quant_in depth=32
#endif

    ffn2_load_FP32_HBM2FIFO(
        HBM_FP32_Intermediate,
        FIFO_Quant_in
    );

    ffn2_quantize_mlp1_output_from_FIFO(
        FIFO_Quant_in,
        global_max,
        URAM_MLP1_Quant_Out,
        scaler_MLP1
    );
}

static void ffn2_load_Bias_HBM2BRAM(
    const float *HBM_b,
    float BRAM_b[EMBED_DIM]
) {
    load_b_loop:
    for (int i = 0; i < EMBED_DIM; i++) {
        #pragma HLS PIPELINE II=1

        BRAM_b[i] =
            HBM_b[i];
    }
}

static void ffn2_load_W2_tile_HBM2FIFO(
    const uint512_dt *HBM_W2,
    hls::stream<uint512_dt> &FIFO_W2,
    int iter
) {
    const int row_offset =
        iter *
        TILE_SIZE;

    const int chunks_per_row =
        MLP_DIM /
        64;

    const int base_addr_idx =
        row_offset *
        chunks_per_row;

    load_w2_row:
    for (
        int i = 0;
        i < TILE_SIZE;
        i++
    ) {
        const int current_row_start =
            base_addr_idx +
            i *
            chunks_per_row;

        load_w2_col:
        for (
            int j = 0;
            j < chunks_per_row;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            FIFO_W2.write(
                HBM_W2[
                    current_row_start +
                    j
                ]
            );
        }
    }
}

static void ffn2_load_Residual_Tile_HBM2FIFO(
    const uint512_dt *HBM_X,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
) {
    const int words_per_row =
        EMBED_DIM /
        16;

    const int words_per_tile =
        TILE_SIZE /
        16;

    const int tile_col_offset =
        iter *
        words_per_tile;

    load_res_seq_loop:
    for (
        int i = 0;
        i < SEQ_LEN;
        i++
    ) {
        const int row_start_idx =
            i *
            words_per_row;

        load_res_tile_loop:
        for (
            int j = 0;
            j < words_per_tile;
            j++
        ) {
            #pragma HLS PIPELINE II=2

            const int target_idx =
                row_start_idx +
                tile_col_offset +
                j;

            uint512_dt hbm_word =
                HBM_X[target_idx];

            FIFO_Res.write(
                hbm_word.range(
                    255,
                    0
                )
            );

            FIFO_Res.write(
                hbm_word.range(
                    511,
                    256
                )
            );
        }
    }
}

static void ffn2_store_Final_Tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Final_Out,
    uint512_dt *HBM_Out,
    int iter
) {
    const int words_per_row =
        EMBED_DIM /
        16;

    const int words_per_tile =
        TILE_SIZE /
        16;

    const int tile_col_offset =
        iter *
        words_per_tile;

    store_res_seq_loop:
    for (
        int i = 0;
        i < SEQ_LEN;
        i++
    ) {
        const int row_start_idx =
            i *
            words_per_row;

        store_res_tile_loop:
        for (
            int j = 0;
            j < words_per_tile;
            j++
        ) {
            #pragma HLS PIPELINE II=2

            uint256_dt lower =
                FIFO_Final_Out.read();

            uint256_dt upper =
                FIFO_Final_Out.read();

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

            HBM_Out[
                row_start_idx +
                tile_col_offset +
                j
            ] = hbm_word;
        }
    }
}

static void ffn2_compute_mlp2_proj_and_residual(
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

    load_w2_row:
    for (
        int i = 0;
        i < TILE_SIZE;
        i++
    ) {
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

    compute_seq:
    for (
        int i = 0;
        i < SEQ_LEN;
        i++
    ) {
        #pragma HLS LOOP_FLATTEN off

        int32_dt acc_buffer[TILE_SIZE];

        #pragma HLS BIND_STORAGE variable=acc_buffer \
            type=ram_2p impl=lutram

        compute_tile:
        for (
            int j = 0;
            j < TILE_SIZE;
            j++
        ) {
            int32_dt acc =
                0;

            dot_product_step:
            for (
                int step = 0;
                step < MLP_DIM_package / 16;
                step++
            ) {
                #pragma HLS PIPELINE II=1

                int32_dt cycle_acc =
                    0;

                mac_pack_loop:
                for (int p = 0; p < 16; p++) {
                    #pragma HLS UNROLL

                    const int pack_idx =
                        step * 16 +
                        p;

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
                            start_bit +
                            7;

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
                            in_val *
                            w2_val;
                    }
                }

                acc +=
                    cycle_acc;
            }

            acc_buffer[j] =
                acc;
        }

        float token_scaler =
            scaler_MLP1[i];

        uint256_dt res_chunk =
            0;

        float pack_array[FP32_PACK];

        #pragma HLS ARRAY_PARTITION variable=pack_array \
            type=complete dim=1

        post_process_loop:
        for (
            int j = 0;
            j < TILE_SIZE;
            j++
        ) {
            #pragma HLS PIPELINE II=1

            if (
                (j % FP32_PACK) == 0
            ) {
                res_chunk =
                    FIFO_Res.read();
            }

            const int weight_idx =
                iter *
                TILE_SIZE +
                j;

            float combined_scale =
                token_scaler *
                scale_w2[weight_idx];

            float proj_val =
                static_cast<float>(
                    acc_buffer[j]
                ) *
                combined_scale +
                BRAM_b2[weight_idx];

            const int res_start_bit =
                (j % FP32_PACK) *
                32;

            const int res_end_bit =
                res_start_bit +
                31;

            union {
                float f;
                uint32_t i;
            } res_conv;

            res_conv.i =
                res_chunk.range(
                    res_end_bit,
                    res_start_bit
                );

            pack_array[
                j % FP32_PACK
            ] =
                proj_val +
                res_conv.f;

            if (
                (j % FP32_PACK) ==
                (FP32_PACK - 1)
            ) {
                uint256_dt out_chunk =
                    0;

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
                        out_start +
                        31;

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

static void ffn2_load_w2_and_res_sequential(
    const uint512_dt *HBM_W2,
    const uint512_dt *HBM_X_Res,
    hls::stream<uint512_dt> &FIFO_W2,
    hls::stream<uint256_dt> &FIFO_Res,
    int iter
) {
    #pragma HLS INLINE off

    ffn2_load_W2_tile_HBM2FIFO(
        HBM_W2,
        FIFO_W2,
        iter
    );

    ffn2_load_Residual_Tile_HBM2FIFO(
        HBM_X_Res,
        FIFO_Res,
        iter
    );
}

static void ffn2_process_mlp2_single_tile_dataflow(
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

    ffn2_load_w2_and_res_sequential(
        HBM_W2,
        HBM_X_Res,
        FIFO_W2,
        FIFO_Res,
        iter
    );

    ffn2_compute_mlp2_proj_and_residual(
        URAM_MLP1_Quant_Out,
        FIFO_W2,
        FIFO_Res,
        FIFO_Final_Out,
        scaler_MLP1,
        BRAM_b2,
        scale_w2,
        iter
    );

    ffn2_store_Final_Tile_FIFO2HBM(
        FIFO_Final_Out,
        HBM_Out,
        iter
    );
}

static void ffn2_run_full_mlp_layer2(
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

    ffn2_load_Bias_HBM2BRAM(
        HBM_b2,
        SRAM_b2
    );

    ffn2_load_Bias_HBM2BRAM(
        HBM_scale_w2,
        SRAM_scale_w2
    );

    const int NUM_TILES =
        EMBED_DIM /
        TILE_SIZE;

    full_mlp2_loop:
    for (
        int iter = 0;
        iter < NUM_TILES;
        iter++
    ) {
        ffn2_process_mlp2_single_tile_dataflow(
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
