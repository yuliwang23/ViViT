//==============================================================
// File: tb_mlp_from_files.cpp
// Description:
//   Testbench that reads all kernel inputs from text files
//   (produced by gen_input), calls the mlp_q kernel, and compares
//   the kernel output against the FP32 golden in golden_output.txt.
//
//   Expected files in the current working directory:
//     input.txt          FP32, SEQ_LEN * EMBED_DIM
//     W1_q.txt           int8, HIDDEN_DIM * EMBED_DIM   (already quantized)
//     W2_q.txt           int8, EMBED_DIM * HIDDEN_DIM   (already quantized)
//     b1.txt             FP32, HIDDEN_DIM
//     b2.txt             FP32, EMBED_DIM
//     scale_w1.txt       FP32, HIDDEN_DIM
//     scale_w2.txt       FP32, EMBED_DIM
//     golden_output.txt  FP32, SEQ_LEN * EMBED_DIM
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
// Packers (FP32 / int8 array -> uint256_dt chunks)
//==============================================================
static void pack_fp32(const std::vector<float> &fp,
                      std::vector<uint256_dt>  &packed)
{
    int N = (int)fp.size();
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
    std::vector<float> input_fp, b1, b2, scale_w1, scale_w2, golden;
    std::vector<int>   W1_q, W2_q;

    read_fp32("input.txt",         input_fp, (size_t)SEQ_LEN * EMBED_DIM);
    read_int8("W1_q.txt",          W1_q,     (size_t)HIDDEN_DIM * EMBED_DIM);
    read_int8("W2_q.txt",          W2_q,     (size_t)EMBED_DIM  * HIDDEN_DIM);
    read_fp32("b1.txt",            b1,       HIDDEN_DIM);
    read_fp32("b2.txt",            b2,       EMBED_DIM);
    read_fp32("scale_w1.txt",      scale_w1, HIDDEN_DIM);
    read_fp32("scale_w2.txt",      scale_w2, EMBED_DIM);
    read_fp32("golden_output.txt", golden,   (size_t)SEQ_LEN * EMBED_DIM);

    //----------------------------------------------------------
    // 2. Pack inputs into uint256_dt chunks
    //----------------------------------------------------------
    printf("[2] Packing into uint256 chunks ...\n");
    std::vector<uint256_dt> input_packed, W1_packed, W2_packed;
    pack_fp32(input_fp, input_packed);
    pack_int8(W1_q,     W1_packed);
    pack_int8(W2_q,     W2_packed);

    std::vector<uint256_dt> output_packed(SEQ_LEN * EMBED_DIM / 8, 0);

    //----------------------------------------------------------
    // 3. Call kernel
    //----------------------------------------------------------
    printf("[3] Calling mlp_q kernel ...\n");
    mlp_q(
        input_packed.data(),
        W1_packed.data(),
        W2_packed.data(),
        b1.data(),
        b2.data(),
        scale_w1.data(),
        scale_w2.data(),
        output_packed.data()
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
    // 5. Compare against golden
    //----------------------------------------------------------
    ErrorReport e = compute_error(golden, output_got);

    printf("\n======================================================\n");
    printf("  Error Report (kernel vs FP32 golden)\n");
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
               i, golden[i], output_got[i],
               output_got[i] - golden[i]);
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