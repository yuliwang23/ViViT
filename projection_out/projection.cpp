#include "projection.h"

static int8_t clamp_int8(int value)
{
#pragma HLS INLINE
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<int8_t>(value);
}

extern "C" {
void projection_out(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{
#pragma HLS INTERFACE m_axi port=attention_in offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=bias offset=slave bundle=gmem3 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=scale_weight offset=slave bundle=gmem3 max_widen_bitwidth=256

#pragma HLS INTERFACE s_axilite port=attention_in bundle=control
#pragma HLS INTERFACE s_axilite port=weight bundle=control
#pragma HLS INTERFACE s_axilite port=bias bundle=control
#pragma HLS INTERFACE s_axilite port=scale_weight bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t q_token[EMBED_DIM];
    float out_token[EMBED_DIM];

#pragma HLS ARRAY_PARTITION variable=q_token type=cyclic factor=32 dim=1
#pragma HLS ARRAY_PARTITION variable=out_token type=cyclic factor=8 dim=1

token_loop:
    for (int token = 0; token < SEQ_LEN; token++) {
        float scale_input = 1.0f;
        quantize_attention_token(attention_in, token, q_token, &scale_input);
        project_token(q_token, scale_input, weight, bias, scale_weight, out_token);

    store_out:
        for (int o = 0; o < EMBED_DIM; o++) {
#pragma HLS PIPELINE II=1
            out[token][o] = out_token[o];
        }
    }
}
}

void quantize_attention_token(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    int8_t q_token[EMBED_DIM],
    float *scale)
{
#pragma HLS INLINE off
    float max_abs = 0.0f;

find_max_head:
    for (int h = 0; h < NUM_HEADS; h++) {
    find_max_dim:
        for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
            float value = fabsf(attention_in[h][token][d]);
            if (value > max_abs) {
                max_abs = value;
            }
        }
    }

    const float local_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    *scale = local_scale;

quant_head:
    for (int h = 0; h < NUM_HEADS; h++) {
    quant_dim:
        for (int d = 0; d < HEAD_DIM; d++) {
#pragma HLS PIPELINE II=1
            const int flat = h * HEAD_DIM + d;
            const float scaled = attention_in[h][token][d] / local_scale;
            const int rounded = static_cast<int>(roundf(scaled));
            q_token[flat] = clamp_int8(rounded);
        }
    }
}

void project_token(
    int8_t q_token[EMBED_DIM],
    float scale_input,
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out_token[EMBED_DIM])
{
#pragma HLS INLINE off

out_tile_loop:
    for (int o_tile = 0; o_tile < EMBED_DIM; o_tile += OUT_TILE) {
    out_col_loop:
        for (int o = 0; o < OUT_TILE; o++) {
            const int out_col = o_tile + o;
            int32_t acc = 0;

        dot_loop:
            for (int d = 0; d < EMBED_DIM; d += 16) {
#pragma HLS PIPELINE II=1
                int32_t local_acc = 0;
            dot_unroll:
                for (int r = 0; r < 16; r++) {
#pragma HLS UNROLL
                    const int in_col = d + r;
                    local_acc += static_cast<int32_t>(q_token[in_col]) *
                                 static_cast<int32_t>(weight[in_col][out_col]);
                }
                acc += local_acc;
            }

            out_token[out_col] =
                static_cast<float>(acc) * scale_input * scale_weight[out_col] +
                bias[out_col];
        }
    }
}
