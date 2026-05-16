#include "projection.h"

#include <ap_int.h>

typedef ap_uint<64> qpack_t;

void quantize_attention_token(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    qpack_t q_token_pack[EMBED_DIM_PACKED],
    float *scale);

void quantize_attention_all(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    qpack_t q_tokens_pack[SEQ_LEN][EMBED_DIM_PACKED],
    float scale_input[SEQ_LEN]);

void load_weight_tile(
    int8_t weight[EMBED_DIM][EMBED_DIM],
    int o_tile,
    int8_t weight_tile[EMBED_DIM][OUT_TILE]);

void project_tile(
    qpack_t q_tokens_pack[SEQ_LEN][EMBED_DIM_PACKED],
    float scale_input[SEQ_LEN],
    int8_t weight_tile[EMBED_DIM][OUT_TILE],
    int o_tile,
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM]);

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

        static qpack_t q_tokens_pack[SEQ_LEN][EMBED_DIM_PACKED]; // 8 packed int8 values per 64-bit word
        static float scale_input[SEQ_LEN];                       // Scale factor for each token's quantized attention concat
        static int8_t weight_tile[EMBED_DIM][OUT_TILE];          // Tile of the weight matrix

#pragma HLS ARRAY_PARTITION variable = q_tokens_pack type = cyclic factor = PACKS_PER_DOT dim = 2
#pragma HLS ARRAY_PARTITION variable = weight_tile type = cyclic factor = DOT_UNROLL dim = 1
#pragma HLS BIND_STORAGE variable = q_tokens_pack type = ram_2p impl = uram
#pragma HLS BIND_STORAGE variable = weight_tile type = ram_2p impl = bram

        quantize_attention_all(attention_in, q_tokens_pack, scale_input);

    out_tile_loop:
        for (int o_tile = 0; o_tile < EMBED_DIM; o_tile += OUT_TILE)
        {
            load_weight_tile(weight, o_tile, weight_tile);
            project_tile(q_tokens_pack, scale_input, weight_tile, o_tile, bias, scale_weight, out);
        }
    }
}

void quantize_attention_all( // Quantize the attention output for all tokens
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    qpack_t q_tokens_pack[SEQ_LEN][EMBED_DIM_PACKED],
    float scale_input[SEQ_LEN])
{
#pragma HLS INLINE off

quant_token_loop:
    for (int token = 0; token < SEQ_LEN; token++)
    {
        quantize_attention_token(attention_in, token, q_tokens_pack[token], &scale_input[token]);
    }
}

void quantize_attention_token( // Quantize the attention output for a single token across all heads
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int token,
    qpack_t q_token_pack[EMBED_DIM_PACKED],
    float *scale)
{
#pragma HLS INLINE off
    float token_buf[EMBED_DIM];
#pragma HLS ARRAY_PARTITION variable = token_buf type = cyclic factor = PACK_FACTOR dim = 1

    uint32_t max_abs_bits = 0;

load_token:
    for (int flat = 0; flat < EMBED_DIM; flat++)
    {
#pragma HLS PIPELINE II = 1
        const int h = flat / HEAD_DIM;
        const int d = flat - h * HEAD_DIM;
        const float value = attention_in[h][token][d];
        token_buf[flat] = value;

        union
        {
            float f;
            uint32_t u;
        } converter;
        converter.f = value;
        const uint32_t abs_bits = converter.u & 0x7fffffffu;
        if (abs_bits > max_abs_bits)
        {
            max_abs_bits = abs_bits;
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

quant_pack:
    for (int pack_idx = 0; pack_idx < EMBED_DIM_PACKED; pack_idx++)
    {
#pragma HLS PIPELINE II = 1
        qpack_t pack = 0;

    pack_lane:
        for (int lane = 0; lane < PACK_FACTOR; lane++)
        {
#pragma HLS UNROLL
            const int flat = pack_idx * PACK_FACTOR + lane;
            const float scaled = token_buf[flat] * inv_scale;
            const int rounded = static_cast<int>(roundf(scaled));
            const int bit_offset = lane * 8;
            const ap_uint<8> q_bits = static_cast<unsigned char>(clamp_int8(rounded));
            pack.range(bit_offset + 7, bit_offset) = q_bits;
        }

        q_token_pack[pack_idx] = pack;
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
    qpack_t q_tokens_pack[SEQ_LEN][EMBED_DIM_PACKED],
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
                qpack_t q_pack[PACKS_PER_DOT];
#pragma HLS ARRAY_PARTITION variable = q_pack complete dim = 1

            load_q_pack:
                for (int p = 0; p < PACKS_PER_DOT; p++)
                {
#pragma HLS UNROLL
                    q_pack[p] = q_tokens_pack[token][(d / PACK_FACTOR) + p];
                }

            dot_unpack:
                for (int p = 0; p < PACKS_PER_DOT; p++)
                {
#pragma HLS UNROLL
                dot_unroll:
                    for (int r = 0; r < PACK_FACTOR; r++)
                    {
#pragma HLS UNROLL
                        const int in_col = d + p * PACK_FACTOR + r;
                        const int bit_offset = r * 8;
                        const ap_int<8> q_val = q_pack[p].range(bit_offset + 7, bit_offset);
                        local_acc += static_cast<int32_t>(q_val) *
                                     static_cast<int32_t>(weight_tile[in_col][o]);
                    }
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
