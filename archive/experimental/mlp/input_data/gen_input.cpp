//==============================================================
// File: gen_input.cpp
// Description:
//   Stand-alone host utility (NOT a Vitis kernel).
//   Generates the input tensors needed to test the mlp_q kernel,
//   and writes them out as plain-text files:
//
//     input.txt        FP32, SEQ_LEN * EMBED_DIM   values
//     W1_q.txt         int8, HIDDEN_DIM * EMBED_DIM values
//     W2_q.txt         int8, EMBED_DIM * HIDDEN_DIM values
//     b1.txt           FP32, HIDDEN_DIM            values
//     b2.txt           FP32, EMBED_DIM             values
//     scale_w1.txt     FP32, HIDDEN_DIM            values
//     scale_w2.txt     FP32, EMBED_DIM             values
//     golden_output.txt FP32, SEQ_LEN * EMBED_DIM  values
//
//   Each file has one value per line, in row-major / linearized
//   order matching how the kernel expects them packed.
//
//   Build (host PC, no Vitis needed):
//     g++ -std=c++14 gen_input.cpp -o gen_input -lm
//   Run:
//     ./gen_input
//==============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <fstream>
#include <iomanip>
#include <algorithm>

// Match the kernel's dimensions exactly
#define SEQ_LEN     197
#define EMBED_DIM   768
#define HIDDEN_DIM  3072

//==============================================================
// Reference math (FP32 golden)
//==============================================================
static float gelu_ref(float x) {
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float t = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(t));
}

static void mlp_reference(
    const std::vector<float> &input_fp,
    const std::vector<float> &W1_fp,
    const std::vector<float> &W2_fp,
    const std::vector<float> &b1,
    const std::vector<float> &b2,
    std::vector<float>       &output_ref)
{
    std::vector<float> hidden(SEQ_LEN * HIDDEN_DIM, 0.0f);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int n = 0; n < HIDDEN_DIM; n++) {
            float acc = b1[n];
            for (int k = 0; k < EMBED_DIM; k++) {
                acc += input_fp[i * EMBED_DIM + k] * W1_fp[n * EMBED_DIM + k];
            }
            hidden[i * HIDDEN_DIM + n] = gelu_ref(acc);
        }
    }
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
// Offline per-output-channel weight quantization to int8
//==============================================================
static void quantize_weight(
    const std::vector<float> &W_fp,
    int                       N_out,
    int                       K_dim,
    std::vector<int>         &W_q,
    std::vector<float>       &scale_W)
{
    W_q.assign(N_out * K_dim, 0);
    scale_W.assign(N_out, 0.0f);

    for (int n = 0; n < N_out; n++) {
        float max_abs = 0.0f;
        for (int k = 0; k < K_dim; k++) {
            float a = std::fabs(W_fp[n * K_dim + k]);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs < 1e-8f) max_abs = 1e-8f;
        float scale = max_abs / 127.0f;
        float inv   = 1.0f / scale;
        scale_W[n]  = scale;

        for (int k = 0; k < K_dim; k++) {
            float f  = W_fp[n * K_dim + k] * inv;
            int   qi = (int)std::round(f);
            if (qi >  127) qi =  127;
            if (qi < -128) qi = -128;
            W_q[n * K_dim + k] = qi;
        }
    }
}

//==============================================================
// File writers
//==============================================================
static void write_fp32(const std::string &path, const std::vector<float> &v) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s for write\n", path.c_str());
        std::exit(1);
    }
    // %.9e gives full single-precision precision (~9 significant digits)
    f << std::scientific << std::setprecision(9);
    for (float x : v) f << x << "\n";
    printf("  wrote %s  (%zu floats)\n", path.c_str(), v.size());
}

static void write_int8(const std::string &path, const std::vector<int> &v) {
    std::ofstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open %s for write\n", path.c_str());
        std::exit(1);
    }
    for (int x : v) f << x << "\n";
    printf("  wrote %s  (%zu ints)\n", path.c_str(), v.size());
}

//==============================================================
// main
//==============================================================
int main(int argc, char **argv)
{
    unsigned seed = 42;
    if (argc >= 2) seed = (unsigned)std::atoi(argv[1]);

    printf("======================================================\n");
    printf("  gen_input  (seed = %u)\n", seed);
    printf("    SEQ_LEN    = %d\n", SEQ_LEN);
    printf("    EMBED_DIM  = %d\n", EMBED_DIM);
    printf("    HIDDEN_DIM = %d\n", HIDDEN_DIM);
    printf("======================================================\n");

    //----------------------------------------------------------
    // 1. Random data
    //----------------------------------------------------------
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist_x(0.0f, 1.0f);
    std::normal_distribution<float> dist_w(0.0f, 0.02f);
    std::normal_distribution<float> dist_b(0.0f, 0.05f);

    std::vector<float> input_fp(SEQ_LEN * EMBED_DIM);
    std::vector<float> W1_fp   (HIDDEN_DIM * EMBED_DIM);
    std::vector<float> W2_fp   (EMBED_DIM  * HIDDEN_DIM);
    std::vector<float> b1      (HIDDEN_DIM);
    std::vector<float> b2      (EMBED_DIM);

    printf("[1] Generating random tensors ...\n");
    for (auto &v : input_fp) v = dist_x(rng);
    for (auto &v : W1_fp)    v = dist_w(rng);
    for (auto &v : W2_fp)    v = dist_w(rng);
    for (auto &v : b1)       v = dist_b(rng);
    for (auto &v : b2)       v = dist_b(rng);

    //----------------------------------------------------------
    // 2. FP32 reference (golden output)
    //----------------------------------------------------------
    // SEQ_LEN=197 makes this ~3x slower than at SEQ_LEN=64.
    // Expect ~30s on a modern desktop CPU.
    printf("[2] Computing FP32 reference (may take ~30s at SEQ_LEN=197) ...\n");
    std::vector<float> output_ref(SEQ_LEN * EMBED_DIM);
    mlp_reference(input_fp, W1_fp, W2_fp, b1, b2, output_ref);

    //----------------------------------------------------------
    // 3. Quantize weights per output channel
    //----------------------------------------------------------
    printf("[3] Quantizing weights ...\n");
    std::vector<int>   W1_q, W2_q;
    std::vector<float> scale_w1, scale_w2;
    quantize_weight(W1_fp, HIDDEN_DIM, EMBED_DIM,  W1_q, scale_w1);
    quantize_weight(W2_fp, EMBED_DIM,  HIDDEN_DIM, W2_q, scale_w2);

    //----------------------------------------------------------
    // 4. Dump everything to files
    //----------------------------------------------------------
    printf("[4] Writing files ...\n");
    write_fp32("input.txt",         input_fp);
    write_int8("W1_q.txt",          W1_q);
    write_int8("W2_q.txt",          W2_q);
    write_fp32("b1.txt",            b1);
    write_fp32("b2.txt",            b2);
    write_fp32("scale_w1.txt",      scale_w1);
    write_fp32("scale_w2.txt",      scale_w2);
    write_fp32("golden_output.txt", output_ref);

    //----------------------------------------------------------
    // 5. Print summary stats so user can sanity-check values
    //----------------------------------------------------------
    float in_min  =  1e30f, in_max  = -1e30f;
    float out_min =  1e30f, out_max = -1e30f;
    for (float v : input_fp) {
        if (v < in_min)  in_min  = v;
        if (v > in_max)  in_max  = v;
    }
    for (float v : output_ref) {
        if (v < out_min) out_min = v;
        if (v > out_max) out_max = v;
    }
    printf("\n  Summary:\n");
    printf("    input  range : [%+.3f, %+.3f]\n", in_min,  in_max);
    printf("    output range : [%+.3f, %+.3f]\n", out_min, out_max);
    printf("    scale_w1 range: [%.6f, %.6f]\n",
        *std::min_element(scale_w1.begin(), scale_w1.end()),
        *std::max_element(scale_w1.begin(), scale_w1.end()));
    printf("    scale_w2 range: [%.6f, %.6f]\n",
        *std::min_element(scale_w2.begin(), scale_w2.end()),
        *std::max_element(scale_w2.begin(), scale_w2.end()));

    printf("\nDone.\n");
    return 0;
}