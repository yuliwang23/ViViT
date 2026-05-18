//==============================================================
// File: compare_output.cpp
// Description:
//   Stand-alone comparison utility — NO HLS headers required.
//   Compares the hardware kernel output against the FP32 software
//   golden reference.
//
//   Inputs (plain text, space- or newline-separated):
//     output.txt         <- produced by the FPGA host (hardware result)
//     golden_output.txt  <- produced by gen_input (software FP32 reference)
//
//   Both files must contain SEQ_LEN * EMBED_DIM float values.
//
//   Build (any platform, no Vitis needed):
//     g++ -std=c++14 compare_output.cpp -o compare
//   Run:
//     ./compare
//   or specify custom filenames:
//     ./compare myout.txt mygolden.txt
//==============================================================
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

// Must match the kernel dimensions
#define SEQ_LEN     64
#define EMBED_DIM   768

static bool read_floats(const std::string &path,
                        std::vector<float> &out,
                        size_t expected)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "[ERROR] Cannot open %s\n", path.c_str());
        return false;
    }
    out.clear();
    out.reserve(expected);
    float v;
    while (f >> v) out.push_back(v);
    f.close();
    printf("  read %-22s : %zu values\n", path.c_str(), out.size());
    if (out.size() != expected) {
        fprintf(stderr,
            "[WARN] %s has %zu values, expected %zu (will compare min length)\n",
            path.c_str(), out.size(), expected);
    }
    return true;
}

int main(int argc, char **argv)
{
    const char *out_path    = (argc >= 2) ? argv[1] : "output.txt";
    const char *golden_path = (argc >= 3) ? argv[2] : "golden_output.txt";

    const size_t EXPECTED = (size_t)SEQ_LEN * EMBED_DIM;

    printf("======================================================\n");
    printf("  Hardware vs Software Output Comparison\n");
    printf("    hardware : %s\n", out_path);
    printf("    golden   : %s\n", golden_path);
    printf("    expected : %d x %d = %zu values\n",
           SEQ_LEN, EMBED_DIM, EXPECTED);
    printf("======================================================\n");

    std::vector<float> hw, ref;
    if (!read_floats(out_path,    hw,  EXPECTED)) return 1;
    if (!read_floats(golden_path, ref, EXPECTED)) return 1;

    size_t N = hw.size() < ref.size() ? hw.size() : ref.size();
    if (N == 0) {
        fprintf(stderr, "[ERROR] One of the files is empty.\n");
        return 1;
    }

    //----------------------------------------------------------
    // Compute error statistics
    //----------------------------------------------------------
    double sum_abs = 0.0, sum_sq = 0.0;
    float  max_abs_err = 0.0f, max_ref_abs = 0.0f;
    int    worst_idx = -1;
    int    nan_count = 0;

    for (size_t i = 0; i < N; i++) {
        float g = ref[i];
        float h = hw[i];
        if (std::isnan(h) || std::isinf(h)) { nan_count++; continue; }
        float d = h - g;
        float a = std::fabs(d);
        if (a > max_abs_err) { max_abs_err = a; worst_idx = (int)i; }
        sum_abs += a;
        sum_sq  += (double)d * (double)d;
        float r = std::fabs(g);
        if (r > max_ref_abs) max_ref_abs = r;
    }

    float mean_abs = (float)(sum_abs / N);
    float rmse     = (float)std::sqrt(sum_sq / N);
    float rel_max  = (max_ref_abs > 0) ? (max_abs_err / max_ref_abs) : 0.0f;

    printf("\n------------------------------------------------------\n");
    printf("  Error Report\n");
    printf("------------------------------------------------------\n");
    printf("    compared values  : %zu\n", N);
    printf("    NaN/Inf in hw    : %d\n", nan_count);
    printf("    max |ref|        : %.6e\n", max_ref_abs);
    printf("    max abs error    : %.6e  (at idx %d)\n",
           max_abs_err, worst_idx);
    printf("    mean abs error   : %.6e\n", mean_abs);
    printf("    rmse             : %.6e\n", rmse);
    printf("    relative max err : %.4f %%\n", rel_max * 100.0f);
    printf("------------------------------------------------------\n");

    //----------------------------------------------------------
    // Show worst-case sample neighborhood
    //----------------------------------------------------------
    if (worst_idx >= 0) {
        int start = worst_idx - 2; if (start < 0) start = 0;
        int end   = worst_idx + 3; if (end > (int)N) end = (int)N;
        printf("\n  Worst-case neighborhood (idx %d):\n", worst_idx);
        printf("    %-6s  %-14s  %-14s  %-14s\n",
               "idx", "hardware", "golden", "error");
        for (int i = start; i < end; i++) {
            printf("    %-6d  % 12.6f  % 12.6f  % 12.6f%s\n",
                   i, hw[i], ref[i], hw[i] - ref[i],
                   (i == worst_idx) ? "  <-- worst" : "");
        }
    }

    //----------------------------------------------------------
    // First 8 values of token 0
    //----------------------------------------------------------
    printf("\n  Sample [first 8 values]:\n");
    printf("    %-6s  %-14s  %-14s  %-14s\n",
           "idx", "hardware", "golden", "error");
    for (int i = 0; i < 8 && i < (int)N; i++) {
        printf("    %-6d  % 12.6f  % 12.6f  % 12.6f\n",
               i, hw[i], ref[i], hw[i] - ref[i]);
    }

    //----------------------------------------------------------
    // Pass / fail
    //----------------------------------------------------------
    const float REL_TOL = 0.10f;   // 10% — int8 quantization tolerance
    printf("\n======================================================\n");
    if (nan_count > 0) {
        printf("[FAIL] hardware output contains %d NaN/Inf values\n",
               nan_count);
        return 1;
    }
    if (rel_max > REL_TOL) {
        printf("[FAIL] relative max error %.4f%% > %.2f%%\n",
               rel_max * 100.0f, REL_TOL * 100.0f);
        return 1;
    }
    printf("[PASS] hardware matches software within %.0f%% tolerance\n",
           REL_TOL * 100.0f);
    printf("======================================================\n");
    return 0;
}