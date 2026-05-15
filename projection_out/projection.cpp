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
#pragma HLS INTERFACE m_axi port = attention_in offset = slave bundle = gmem0 max_widen_bitwidth = 256 // memory access 256 bits at a time (32 int8 values)
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

        static int8_t q_tokens[SEQ_LEN][EMBED_DIM];     // Quantized attention concat for each token
        static float scale_input[SEQ_LEN];              // Scale factor for each token's quantized attention concat
        static int8_t weight_tile[EMBED_DIM][OUT_TILE]; // Tile of the weight matrix

#pragma HLS ARRAY_PARTITION variable = q_tokens type = cyclic factor = 32 dim = 2 // Partition the quantized tokens by columns (embedding dimension) with a factor of 32 to enable parallel access to 32 int8 values for the dot product
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

void quantize_attention_all( // Quantize the attention output for all tokens
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

void quantize_attention_token( // Quantize the attention output for a single token across all heads
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    int8_t q_token[EMBED_DIM],
    float *scale)
{
#pragma HLS INLINE off
    uint32_t max_abs_bits = 0;

find_max_head:
    for (int h = 0; h < NUM_HEADS; h++)
    {
    find_max_dim:
        for (int d = 0; d < HEAD_DIM; d++)
        {
#pragma HLS PIPELINE II = 1
            union
            {
                float f;
                uint32_t u;
            } converter;
            converter.f = attention_in[h][token][d];
            const uint32_t abs_bits = converter.u & 0x7fffffffu;
            if (abs_bits > max_abs_bits)
            {
                max_abs_bits = abs_bits;
            }
        }
    }

    union
    {
        float f;
        uint32_t u;
    } max_converter;
    max_converter.u = max_abs_bits;
    const float max_abs = max_converter.f;
    const float local_scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
    const float inv_scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 1.0f;
    *scale = local_scale;

quant_head:
    for (int h = 0; h < NUM_HEADS; h++)
    {
    quant_dim:
        for (int d = 0; d < HEAD_DIM; d++)
        {
#pragma HLS PIPELINE II = 1
            const int flat = h * HEAD_DIM + d;
            const float scaled = attention_in[h][token][d] * inv_scale;
            const int rounded = static_cast<int>(roundf(scaled));
            q_token[flat] = clamp_int8(rounded); // concatenate the quantized heads for this token into one vector of size EMBED_DIM (NUM_HEADS * HEAD_DIM)
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
