//==============================================================
// File: tb_mlp_from_files.cpp
// Description:
//   Testbench that reads all kernel inputs from text files
//   (produced by gen_input), calls the mlp_q kernel, and compares
//   the kernel output against a host reference that uses the same
//   quantized LayerNorm input format as mlp_q.
//
//   Expected files in the current working directory:
//     input.txt          FP32, SEQ_LEN * EMBED_DIM
//     W1_q.txt           int8, HIDDEN_DIM * EMBED_DIM   (already quantized)
//     W2_q.txt           int8, EMBED_DIM * HIDDEN_DIM   (already quantized)
//     b1.txt             FP32, HIDDEN_DIM
//     b2.txt             FP32, EMBED_DIM
//     scale_w1.txt       FP32, HIDDEN_DIM
//     scale_w2.txt       FP32, EMBED_DIM
//
//   In Vitis HLS:
//     add_files       mlp_hls.cpp
//     add_files -tb   tb_mlp_from_files.cpp
//     # Make sure the txt files are in the project run directory.
//
//   The testbench also writes kernel_output.txt for downstream
//   inspection / plotting.
//==============================================================
#include "mlp.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

//==============================================================
// File readers
//==============================================================
static void read_fp32(const std::string &path,
                      std::vector<float> &out,
                      size_t expected_count)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", path.c_str());
        std::exit(1);
    }
    out.clear();
    out.reserve(expected_count);
    float v;
    while (f >> v) out.push_back(v);
    if (out.size() != expected_count) {
        fprintf(stderr,
            "[ERROR] %s: expected %zu values, got %zu\n",
            path.c_str(), expected_count, out.size());
        std::exit(1);
    }
    printf("  read %-22s (%zu floats)\n", path.c_str(), out.size());
}

static void read_int8(const std::string &path,
                      std::vector<int> &out,
                      size_t expected_count)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", path.c_str());
        std::exit(1);
    }
    out.clear();
    out.reserve(expected_count);
    int v;
    while (f >> v) out.push_back(v);
    if (out.size() != expected_count) {
        fprintf(stderr,
            "[ERROR] %s: expected %zu values, got %zu\n",
            path.c_str(), expected_count, out.size());
        std::exit(1);
    }
    printf("  read %-22s (%zu ints)\n", path.c_str(), out.size());
}

//==============================================================
// Packers (int8 array -> uint256_dt chunks)
//==============================================================
static void pack_int8(const std::vector<int>  &q,
                      std::vector<uint256_dt> &packed)
{
    int N = (int)q.size();
    packed.assign(N / 32, 0);
    for (int i = 0; i < N / 32; i++) {
        uint256_dt chunk = 0;
        for (int r = 0; r < 32; r++) {
            int8_dt v = (int8_dt)q[i * 32 + r];
            int s = r * 8;
            chunk.range(s + 7, s) = v;
        }
        packed[i] = chunk;
    }
}

static void quantize_input_for_mlp(const std::vector<float> &fp,
                                   std::vector<uint256_dt> &packed,
                                   std::vector<float> &scaler,
                                   std::vector<float> &dequant)
{
    packed.assign((SEQ_LEN * EMBED_DIM) / 32, 0);
    scaler.assign(SEQ_LEN, 1.0f);
    dequant.assign(SEQ_LEN * EMBED_DIM, 0.0f);

    for (int i = 0; i < SEQ_LEN; i++) {
        float max_val = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++) {
            max_val = std::max(max_val, std::fabs(fp[i * EMBED_DIM + d]));
        }
        if (max_val < EPSILON) max_val = EPSILON;
        float scale = 127.0f / max_val;
        scaler[i] = 1.0f / scale;

        for (int j = 0; j < EMBED_DIM / 32; j++) {
            uint256_dt chunk = 0;
            for (int k = 0; k < 32; k++) {
                int d = j * 32 + k;
                int qi = (int)(fp[i * EMBED_DIM + d] * scale);
                if (qi >  127) qi =  127;
                if (qi < -128) qi = -128;
                int s = k * 8;
                chunk.range(s + 7, s) = (ap_uint<8>)(uint8_t)(int8_t)qi;
                dequant[i * EMBED_DIM + d] = (float)qi * scaler[i];
            }
            packed[i * (EMBED_DIM / 32) + j] = chunk;
        }
    }
}

static float gelu_ref(float x)
{
    const float k0 = 0.7978845608f;
    const float k1 = 0.044715f;
    float t = k0 * (x + k1 * x * x * x);
    return 0.5f * x * (1.0f + std::tanh(t));
}

static void mlp_quant_reference(const std::vector<float> &input_dequant,
                                const std::vector<int> &W1_q,
                                const std::vector<int> &W2_q,
                                const std::vector<float> &b1,
                                const std::vector<float> &b2,
                                const std::vector<float> &scale_w1,
                                const std::vector<float> &scale_w2,
                                std::vector<float> &output_ref)
{
    std::vector<float> hidden(SEQ_LEN * HIDDEN_DIM, 0.0f);
    output_ref.assign(SEQ_LEN * EMBED_DIM, 0.0f);

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int n = 0; n < HIDDEN_DIM; n++) {
            float acc = b1[n];
            for (int k = 0; k < EMBED_DIM; k++) {
                acc += input_dequant[i * EMBED_DIM + k] *
                       (float)W1_q[n * EMBED_DIM + k] * scale_w1[n];
            }
            hidden[i * HIDDEN_DIM + n] = gelu_ref(acc);
        }
    }

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int m = 0; m < EMBED_DIM; m++) {
            float acc = b2[m];
            for (int n = 0; n < HIDDEN_DIM; n++) {
                acc += hidden[i * HIDDEN_DIM + n] *
                       (float)W2_q[m * HIDDEN_DIM + n] * scale_w2[m];
            }
            output_ref[i * EMBED_DIM + m] = acc;
        }
    }
}

//==============================================================
// Unpack output (uint256_dt chunks -> FP32 array)
//==============================================================
static void unpack_fp32(const std::vector<uint256_dt> &packed,
                       std::vector<float>            &fp)
{
    int N = (int)packed.size() * 8;
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
// Error report
//==============================================================
struct ErrorReport {
    float max_abs_err;
    float mean_abs_err;
    float rmse;
    float max_ref_abs;
    float rel_max;
    int   worst_idx;
};

static ErrorReport compute_error(const std::vector<float> &ref,
                                 const std::vector<float> &got)
{
    ErrorReport e{};
    e.max_abs_err = 0;
    e.max_ref_abs = 0;
    e.worst_idx   = -1;
    double sum_abs = 0.0, sum_sq = 0.0;
    int N = (int)ref.size();
    for (int i = 0; i < N; i++) {
        float d = got[i] - ref[i];
        float a = std::fabs(d);
        if (a > e.max_abs_err) {
            e.max_abs_err = a;
            e.worst_idx   = i;
        }
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
    printf("  tb_mlp_from_files\n");
    printf("    SEQ_LEN    = %d\n", SEQ_LEN);
    printf("    EMBED_DIM  = %d\n", EMBED_DIM);
    printf("    HIDDEN_DIM = %d\n", HIDDEN_DIM);
    printf("======================================================\n");

    //----------------------------------------------------------
    // 1. Read all input files
    //----------------------------------------------------------
    printf("[1] Reading input files ...\n");
    std::vector<float> input_fp, b1, b2, scale_w1, scale_w2;
    std::vector<int>   W1_q, W2_q;

    read_fp32("input.txt",         input_fp, (size_t)SEQ_LEN * EMBED_DIM);
    read_int8("W1_q.txt",          W1_q,     (size_t)HIDDEN_DIM * EMBED_DIM);
    read_int8("W2_q.txt",          W2_q,     (size_t)EMBED_DIM  * HIDDEN_DIM);
    read_fp32("b1.txt",            b1,       HIDDEN_DIM);
    read_fp32("b2.txt",            b2,       EMBED_DIM);
    read_fp32("scale_w1.txt",      scale_w1, HIDDEN_DIM);
    read_fp32("scale_w2.txt",      scale_w2, EMBED_DIM);

    //----------------------------------------------------------
    // 2. Pack inputs into uint256_dt chunks
    //----------------------------------------------------------
    printf("[2] Packing into uint256 chunks ...\n");
    std::vector<uint256_dt> input_packed, W1_packed, W2_packed;
    std::vector<float> input_scaler, input_dequant;
    quantize_input_for_mlp(input_fp, input_packed, input_scaler, input_dequant);
    pack_int8(W1_q,     W1_packed);
    pack_int8(W2_q,     W2_packed);

    std::vector<uint256_dt> output_packed(SEQ_LEN * EMBED_DIM / 8, 0);

    //----------------------------------------------------------
    // 2b. Host reference from the same quantized input/weights
    //----------------------------------------------------------
    printf("[2b] Computing quantized host reference ...\n");
    std::vector<float> output_ref;
    mlp_quant_reference(input_dequant, W1_q, W2_q, b1, b2, scale_w1, scale_w2, output_ref);

    //----------------------------------------------------------
    // 3. Call kernel
    //----------------------------------------------------------
    printf("[3] Calling mlp_q kernel ...\n");
    mlp_q(
        input_packed.data(),
        input_scaler.data(),
        W1_packed.data(),
        W2_packed.data(),
        b1.data(),
        b2.data(),
        scale_w1.data(),
        scale_w2.data(),
        output_packed.data(),
        0,
        SEQ_LEN
    );
    printf("[3] Kernel returned.\n");

    //----------------------------------------------------------
    // 4. Unpack output and write to file
    //----------------------------------------------------------
    std::vector<float> output_got;
    unpack_fp32(output_packed, output_got);

    {
        std::ofstream f("output.txt");
        if (f.is_open()) {
            f.precision(9);
            f.setf(std::ios::scientific);
            for (float v : output_got) f << v << "\n";
            printf("[4] Wrote output.txt\n");
        }
    }

    //----------------------------------------------------------
    // 5. Compare against host reference
    //----------------------------------------------------------
    ErrorReport e = compute_error(output_ref, output_got);

    printf("\n======================================================\n");
    printf("  Error Report (kernel vs quantized host reference)\n");
    printf("------------------------------------------------------\n");
    printf("    max |ref|         : %.6e\n", e.max_ref_abs);
    printf("    max abs error     : %.6e  (at idx %d)\n",
           e.max_abs_err, e.worst_idx);
    printf("    mean abs error    : %.6e\n", e.mean_abs_err);
    printf("    rmse              : %.6e\n", e.rmse);
    printf("    relative max err  : %.4f %%\n", e.rel_max * 100.0f);
    printf("======================================================\n\n");

    //----------------------------------------------------------
    // 6. Show a few samples
    //----------------------------------------------------------
    printf("  Sample [token 0, first 8 outputs]:\n");
    printf("    %-4s   %-13s   %-13s   %-13s\n",
           "idx", "ref", "got", "err");
    for (int i = 0; i < 8; i++) {
        printf("    %-4d   % 9.6f   % 9.6f   % 9.6f\n",
               i, output_ref[i], output_got[i],
               output_got[i] - output_ref[i]);
    }
    printf("\n");

    //----------------------------------------------------------
    // 7. Pass / fail
    //----------------------------------------------------------
    const float REL_TOL = 0.10f;
    if (e.rel_max > REL_TOL) {
        printf("[FAIL] relative max error %.4f%% > %.2f%%\n",
               e.rel_max * 100.0f, REL_TOL * 100.0f);
        return 1;
    }
    printf("[PASS] within tolerance.\n");
    return 0;
}
