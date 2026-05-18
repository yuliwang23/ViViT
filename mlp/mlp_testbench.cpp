//==============================================================
// File: tb_mlp.cpp
// Testbench for the quantized MLP HLS kernel.
//
// Flow:
//   1. Generate random FP32 input + weights + biases
//   2. Compute FP32 reference (golden) using the standard GELU formula
//   3. Offline-quantize weights per-output-channel to int8, pack to uint256
//   4. Pack FP32 input to uint256
//   5. Call mlp_q() kernel
//   6. Unpack output and compare against the golden FP32 result
//
// Compile (C-sim from command line, outside Vitis):
//   g++ -std=c++14 -I<HLS_INC> tb_mlp.cpp mlp_hls.cpp -o tb_mlp -lm
//
// Or in Vitis HLS:
//   add_files       mlp_hls.cpp
//   add_files -tb   tb_mlp.cpp
//==============================================================
#include "mlp.h"
#include "gelu_lut.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

//==============================================================
// FP32 reference GELU (exact formula, not LUT)
//==============================================================
static float gelu_ref(float x) {
    const float k0 = 0.7978845608f;   // sqrt(2/pi)
    const float k1 = 0.044715f;
    float t = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(t));
}

//==============================================================
// FP32 reference MLP (golden model)
//   h[i][n] = GELU( sum_k input[i][k] * W1[n][k]  + b1[n] )
//   out[i][m] = sum_n h[i][n] * W2[m][n] + b2[m]
//
//   W1 shape: [HIDDEN_DIM][EMBED_DIM]  (output-channel major)
//   W2 shape: [EMBED_DIM ][HIDDEN_DIM]
//==============================================================
static void mlp_reference(
    const std::vector<float> &input_fp,    // [SEQ_LEN * EMBED_DIM]
    const std::vector<float> &W1_fp,       // [HIDDEN_DIM * EMBED_DIM]
    const std::vector<float> &W2_fp,       // [EMBED_DIM  * HIDDEN_DIM]
    const std::vector<float> &b1,          // [HIDDEN_DIM]
    const std::vector<float> &b2,          // [EMBED_DIM]
    std::vector<float>       &output_ref)  // [SEQ_LEN * EMBED_DIM]
{
    std::vector<float> hidden(SEQ_LEN * HIDDEN_DIM, 0.0f);

    // Dense_0 + bias + GELU
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int n = 0; n < HIDDEN_DIM; n++) {
            float acc = b1[n];
            for (int k = 0; k < EMBED_DIM; k++) {
                acc += input_fp[i * EMBED_DIM + k] * W1_fp[n * EMBED_DIM + k];
            }
            hidden[i * HIDDEN_DIM + n] = gelu_ref(acc);
        }
    }

    // Dense_1 + bias
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int m = 0; m < EMBED_DIM; m++) {
            float acc = b2[m];
            for (int n = 0; n < HIDDEN_DIM; n++) {
                acc += hidden[i * HIDDEN_DIM + n] * W2_fp[m * HIDDEN_DIM + n];
            }
            output_ref[i * EMBED_DIM + m] = acc;
        }
    }
}

//==============================================================
// Offline weight quantization, per-output-channel (symmetric int8)
//   For each row of W, find max-abs, set scale = max_abs / 127.0,
//   quantize all elements of that row by that scale.
//
//   Stored as packed uint256_dt where 32 int8s share one chunk.
//   For one row of length K_dim, that's K_dim/32 chunks.
//==============================================================
static void quantize_weight_offline(
    const std::vector<float> &W_fp,        // [N_out][K_dim] row-major
    int                       N_out,
    int                       K_dim,
    std::vector<uint256_dt>  &W_q_packed,   // size = N_out * (K_dim/32)
    std::vector<float>       &scale_W)      // size = N_out
{
    int chunks_per_row = K_dim / 32;
    W_q_packed.assign(N_out * chunks_per_row, 0);
    scale_W.assign(N_out, 0.0f);

    for (int n = 0; n < N_out; n++) {
        // find max-abs of this row
        float max_abs = 0.0f;
        for (int k = 0; k < K_dim; k++) {
            float a = std::fabs(W_fp[n * K_dim + k]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs < 1e-8f) max_abs = 1e-8f;
        float scale = max_abs / 127.0f;       // dequant scaler
        float inv   = 1.0f / scale;
        scale_W[n]  = scale;

        // quantize and pack
        for (int j = 0; j < chunks_per_row; j++) {
            uint256_dt chunk = 0;
            for (int r = 0; r < 32; r++) {
                int   k = j * 32 + r;
                float f = W_fp[n * K_dim + k] * inv;
                int   qi = (int)std::round(f);
                if (qi >  127) qi =  127;
                if (qi < -128) qi = -128;
                ap_int<8> qv = (ap_int<8>)qi;
                int s = r * 8;
                // place int8 into chunk[s+7 : s]
                chunk.range(s + 7, s) = (ap_uint<8>)(uint8_t)(int8_t)qv;
            }
            W_q_packed[n * chunks_per_row + j] = chunk;
        }
    }
}

//==============================================================
// Pack FP32 array into uint256_dt chunks (8 floats per chunk)
//==============================================================
static void pack_fp32(
    const std::vector<float> &fp,
    std::vector<uint256_dt>  &packed)
{
    int N = fp.size();
    packed.assign(N / 8, 0);
    for (int i = 0; i < N / 8; i++) {
        uint256_dt chunk = 0;
        for (int k = 0; k < 8; k++) {
            union { float f; uint32_t u; } cv;
            cv.f = fp[i * 8 + k];
            int s = k * 32;
            chunk.range(s + 31, s) = cv.u;
        }
        packed[i] = chunk;
    }
}

//==============================================================
// Unpack uint256_dt chunks back to FP32
//==============================================================
static void unpack_fp32(
    const std::vector<uint256_dt> &packed,
    std::vector<float>            &fp)
{
    int N = packed.size() * 8;
    fp.assign(N, 0.0f);
    for (size_t i = 0; i < packed.size(); i++) {
        uint256_dt chunk = packed[i];
        for (int k = 0; k < 8; k++) {
            int s = k * 32;
            uint32_t raw = (uint32_t)chunk.range(s + 31, s);
            union { uint32_t u; float f; } cv;
            cv.u = raw;
            fp[i * 8 + k] = cv.f;
        }
    }
}

//==============================================================
// Error metrics
//==============================================================
struct ErrorReport {
    float max_abs_err;
    float mean_abs_err;
    float rmse;
    float max_ref_abs;
    float rel_max;     // max_abs_err / max_ref_abs
};

static ErrorReport compute_error(
    const std::vector<float> &ref,
    const std::vector<float> &got)
{
    ErrorReport e{};
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    int    N       = ref.size();
    for (int i = 0; i < N; i++) {
        float d = got[i] - ref[i];
        float a = std::fabs(d);
        if (a > e.max_abs_err) e.max_abs_err = a;
        sum_abs += a;
        sum_sq  += (double)d * (double)d;
        float r = std::fabs(ref[i]);
        if (r > e.max_ref_abs) e.max_ref_abs = r;
    }
    e.mean_abs_err = (float)(sum_abs / N);
    e.rmse         = (float)std::sqrt(sum_sq / N);
    e.rel_max      = (e.max_ref_abs > 0) ? (e.max_abs_err / e.max_ref_abs) : 0.0f;
    return e;
}

//==============================================================
// main
//==============================================================
int main()
{
    printf("======================================================\n");
    printf("  MLP-Q Testbench\n");
    printf("    SEQ_LEN    = %d\n", SEQ_LEN);
    printf("    EMBED_DIM  = %d\n", EMBED_DIM);
    printf("    HIDDEN_DIM = %d\n", HIDDEN_DIM);
    printf("    TILE_SIZE  = %d\n", TILE_SIZE);
    printf("======================================================\n");

    //----------------------------------------------------------
    // 1. Random data generation
    //----------------------------------------------------------
    std::mt19937 rng(42);
    std::normal_distribution<float> dist_x(0.0f, 1.0f);
    std::normal_distribution<float> dist_w(0.0f, 0.02f);   // small init
    std::normal_distribution<float> dist_b(0.0f, 0.05f);

    std::vector<float> input_fp(SEQ_LEN * EMBED_DIM);
    std::vector<float> W1_fp   (HIDDEN_DIM * EMBED_DIM);
    std::vector<float> W2_fp   (EMBED_DIM  * HIDDEN_DIM);
    std::vector<float> b1      (HIDDEN_DIM);
    std::vector<float> b2      (EMBED_DIM);

    for (auto &v : input_fp) v = dist_x(rng);
    for (auto &v : W1_fp)    v = dist_w(rng);
    for (auto &v : W2_fp)    v = dist_w(rng);
    for (auto &v : b1)       v = dist_b(rng);
    for (auto &v : b2)       v = dist_b(rng);

    //----------------------------------------------------------
    // 2. FP32 golden reference
    //----------------------------------------------------------
    printf("[ref] Computing FP32 reference ...\n");
    std::vector<float> output_ref(SEQ_LEN * EMBED_DIM);
    mlp_reference(input_fp, W1_fp, W2_fp, b1, b2, output_ref);
    printf("[ref] Done.\n");

    //----------------------------------------------------------
    // 3. Offline quantize weights
    //----------------------------------------------------------
    printf("[q]   Offline-quantizing W1, W2 per output channel ...\n");
    std::vector<uint256_dt> W1_q;
    std::vector<uint256_dt> W2_q;
    std::vector<float>      scale_w1, scale_w2;

    quantize_weight_offline(W1_fp, HIDDEN_DIM, EMBED_DIM,  W1_q, scale_w1);
    quantize_weight_offline(W2_fp, EMBED_DIM,  HIDDEN_DIM, W2_q, scale_w2);
    printf("[q]   W1 scale range: [%.6f, %.6f]\n",
        *std::min_element(scale_w1.begin(), scale_w1.end()),
        *std::max_element(scale_w1.begin(), scale_w1.end()));
    printf("[q]   W2 scale range: [%.6f, %.6f]\n",
        *std::min_element(scale_w2.begin(), scale_w2.end()),
        *std::max_element(scale_w2.begin(), scale_w2.end()));

    //----------------------------------------------------------
    // 4. Pack FP32 input
    //----------------------------------------------------------
    std::vector<uint256_dt> input_packed;
    pack_fp32(input_fp, input_packed);

    //----------------------------------------------------------
    // 5. Allocate output buffer & call kernel
    //----------------------------------------------------------
    std::vector<uint256_dt> output_packed(SEQ_LEN * EMBED_DIM / 8, 0);

    printf("[run] Calling mlp_q kernel ...\n");
    mlp_q(
        input_packed.data(),
        W1_q.data(),
        W2_q.data(),
        b1.data(),
        b2.data(),
        scale_w1.data(),
        scale_w2.data(),
        output_packed.data()
    );
    printf("[run] Kernel returned.\n");

    //----------------------------------------------------------
    // 6. Unpack output and compute error
    //----------------------------------------------------------
    std::vector<float> output_got(SEQ_LEN * EMBED_DIM);
    unpack_fp32(output_packed, output_got);

    ErrorReport e = compute_error(output_ref, output_got);

    printf("\n======================================================\n");
    printf("  Error Report\n");
    printf("------------------------------------------------------\n");
    printf("    max |ref|         : %.6e\n", e.max_ref_abs);
    printf("    max abs error     : %.6e\n", e.max_abs_err);
    printf("    mean abs error    : %.6e\n", e.mean_abs_err);
    printf("    rmse              : %.6e\n", e.rmse);
    printf("    relative max err  : %.4f %%\n", e.rel_max * 100.0f);
    printf("======================================================\n\n");

    //----------------------------------------------------------
    // 7. Show a few sample values
    //----------------------------------------------------------
    printf("  Sample [token 0, first 8 outputs]:\n");
    printf("    %-4s   %-12s   %-12s   %-12s\n", "idx", "ref", "got", "err");
    for (int i = 0; i < 8; i++) {
        printf("    %-4d   % 8.6f   % 8.6f   % 8.6f\n",
               i, output_ref[i], output_got[i],
               output_got[i] - output_ref[i]);
    }
    printf("\n");

    //----------------------------------------------------------
    // 8. Pass / fail
    //----------------------------------------------------------
    //  Tolerance picked for int8 activation + int8 weight + LUT GELU
    //  with EMBED_DIM=768, HIDDEN_DIM=3072.  Errors scale with sqrt(K).
    const float REL_TOL = 0.10f;   // 10% relative max error
    if (e.rel_max > REL_TOL) {
        printf("[FAIL] relative max error %.4f%% > %.2f%%\n",
               e.rel_max * 100.0f, REL_TOL * 100.0f);
        return 1;
    }
    printf("[PASS] within tolerance.\n");
    return 0;
}