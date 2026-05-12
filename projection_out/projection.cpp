#include "projection.h"

// Clamp a value to the int8 range [-128, 127]
static int8_t clamp_int8(int value)
{
#pragma HLS INLINE // prevent function call overhead for this small utility function
    if (value > 127)
    {
        return 127;
    }
    if (value < -128)
    {
        return -128;
    }
    return static_cast<int8_t>(value);
}

extern "C"
{
    void projection_out(
        float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
        int8_t weight[EMBED_DIM][EMBED_DIM],
        float bias[EMBED_DIM],
        float scale_weight[EMBED_DIM],
        int out[SEQ_LEN][EMBED_DIM]) // 197 x 768
    {
#pragma HLS INTERFACE m_axi port = attention_in offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem1 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem2 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem3 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = scale_weight offset = slave bundle = gmem3 max_widen_bitwidth = 256

#pragma HLS INTERFACE s_axilite port = attention_in bundle = control
#pragma HLS INTERFACE s_axilite port = weight bundle = control
#pragma HLS INTERFACE s_axilite port = bias bundle = control
#pragma HLS INTERFACE s_axilite port = scale_weight bundle = control
#pragma HLS INTERFACE s_axilite port = out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

        static int8_t q_tokens[SEQ_LEN][EMBED_DIM];
        static float scale_input[SEQ_LEN];
        static int8_t weight_tile[EMBED_DIM][OUT_TILE];

#pragma HLS ARRAY_PARTITION variable = q_tokens type = cyclic factor = 32 dim = 2
#pragma HLS ARRAY_PARTITION variable = weight_tile type = cyclic factor = DOT_UNROLL dim = 1
#pragma HLS BIND_STORAGE variable = q_tokens type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = weight_tile type = ram_2p impl = bram

        quantize_attention_all(attention_in, q_tokens, scale_input);

    out_tile_loop:
        for (int o_tile = 0; o_tile < EMBED_DIM; o_tile += OUT_TILE)
        {
            load_weight_tile(weight, o_tile, weight_tile);
            project_tile(q_tokens, scale_input, weight_tile, o_tile, bias, scale_weight, out);
        }
    }
}

void quantize_attention_all(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t q_tokens[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN])
{
#pragma HLS INLINE off

quant_token_loop:
    for (int token = 0; token < SEQ_LEN; token++)
    {
        quantize_attention_token(attention_in, token, q_tokens[token], &scale_input[token]);
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
    for (int h = 0; h < NUM_HEADS; h++)
    {
    find_max_dim:
        for (int d = 0; d < HEAD_DIM; d++)
        {
#pragma HLS PIPELINE II = 1
            float value = fabsf(attention_in[h][token][d]);
            if (value > max_abs)
            {
                max_abs = value;
            }
        }
    }

    const float local_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    *scale = local_scale;

quant_head:
    for (int h = 0; h < NUM_HEADS; h++)
    {
    quant_dim:
        for (int d = 0; d < HEAD_DIM; d++)
        {
#pragma HLS PIPELINE II = 1
            const int flat = h * HEAD_DIM + d;
            const float scaled = attention_in[h][token][d] / local_scale;
            const int rounded = static_cast<int>(roundf(scaled));
            q_token[flat] = clamp_int8(rounded);
        }
    }
}

void load_weight_tile(
    int8_t weight[EMBED_DIM][EMBED_DIM],
    int o_tile,
    int8_t weight_tile[EMBED_DIM][OUT_TILE])
{
#pragma HLS INLINE off

load_w_row:
    for (int d = 0; d < EMBED_DIM; d++)
    {
    load_w_col:
        for (int o = 0; o < OUT_TILE; o++)
        {
#pragma HLS PIPELINE II = 1
            weight_tile[d][o] = weight[d][o_tile + o];
        }
    }
}

void project_tile(
    int8_t q_tokens[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight_tile[EMBED_DIM][OUT_TILE],
    int o_tile,
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
#pragma HLS INLINE off

token_loop:
    for (int token = 0; token < SEQ_LEN; token++)
    {
    out_col_loop:
        for (int o = 0; o < OUT_TILE; o++)
        {
            const int out_col = o_tile + o;
            int32_t acc = 0;

        dot_loop:
            for (int d = 0; d < EMBED_DIM; d += DOT_UNROLL)
            {
#pragma HLS PIPELINE II = 1
                int32_t local_acc = 0;
            dot_unroll:
                for (int r = 0; r < DOT_UNROLL; r++)
                {
#pragma HLS UNROLL
                    const int in_col = d + r;
                    local_acc += static_cast<int32_t>(q_tokens[token][in_col]) *
                                 static_cast<int32_t>(weight_tile[in_col][o]);
                }
                acc += local_acc;
            }

            const float dequant =
                static_cast<float>(acc) * scale_input[token] * scale_weight[out_col] +
                bias[out_col];
            out[token][out_col] = static_cast<int>(roundf(dequant));
        }
    }
}
