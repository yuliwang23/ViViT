#include "projection.h"

#include <cmath>
#include <cstdio>

static float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM];
static int8_t weight[EMBED_DIM][EMBED_DIM];
static float bias[EMBED_DIM];
static float scale_weight[EMBED_DIM];
static float out_hw[SEQ_LEN][EMBED_DIM];
static float out_ref[SEQ_LEN][EMBED_DIM];

static int8_t clamp_ref(int value)
{
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<int8_t>(value);
}

static void projection_ref()
{
    for (int token = 0; token < SEQ_LEN; token++) {
        float max_abs = 0.0f;
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const float value = std::fabs(attention_in[h][token][d]);
                if (value > max_abs) {
                    max_abs = value;
                }
            }
        }

        const float scale_input = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        int8_t q_token[EMBED_DIM];

        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int flat = h * HEAD_DIM + d;
                q_token[flat] = clamp_ref(static_cast<int>(std::round(
                    attention_in[h][token][d] / scale_input)));
            }
        }

        for (int o = 0; o < EMBED_DIM; o++) {
            int32_t acc = 0;
            for (int d = 0; d < EMBED_DIM; d++) {
                acc += static_cast<int32_t>(q_token[d]) *
                       static_cast<int32_t>(weight[d][o]);
            }
            const float dequant =
                static_cast<float>(acc) * scale_input * scale_weight[o] + bias[o];
            out_ref[token][o] = dequant;
        }
    }
}

int main()
{
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int v = (h * 17 + i * 5 + d * 3) % 251;
                attention_in[h][i][d] = (static_cast<float>(v) - 125.0f) * 0.003f;
            }
        }
    }

    for (int d = 0; d < EMBED_DIM; d++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            weight[d][o] = static_cast<int8_t>(((d * 11 + o * 7) % 255) - 127);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        bias[o] = static_cast<float>((o % 19) - 9) * 0.001f;
        scale_weight[o] = 0.0004f + static_cast<float>(o % 13) * 0.00001f;
    }

    projection_ref();
    projection_out(attention_in, weight, bias, scale_weight, out_hw);

    float max_err = 0.0f;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            const float err = std::fabs(out_hw[i][o] - out_ref[i][o]);
            if (err > max_err) {
                max_err = err;
            }
        }
    }

    std::printf("projection_out max fp32 error = %.8f\n", max_err);
    if (max_err > 1.0e-5f) {
        std::printf("FAIL\n");
        return 1;
    }

    std::printf("PASS\n");
    return 0;
}
