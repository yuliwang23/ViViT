// verify_output_layernorm.cpp
//
// 1. 讀入 input_x.txt / input_gamma.txt / input_beta.txt
// 2. 用軟體黃金模型（完全模擬硬體量化流程）計算 out (int8) 與 scaler (float)
// 3. 讀入硬體輸出 output_out.txt / output_scaler.txt
// 4. 逐元素比對，報告誤差統計與 pass/fail
//
// 編譯: g++ -O2 -o verify verify_output_layernorm.cpp -lm
// 執行: ./verify

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <vector>
#include <algorithm>

constexpr int   SEQ_LEN   = 197;
constexpr int   EMBED_DIM = 768;
constexpr float EPSILON   = 1e-5f;

// ----------------------------------------------------------------
// 讀 float txt（一行一值）
// ----------------------------------------------------------------
static bool read_float(const char *fname, float *dst, int n)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "[ERR] Cannot open %s\n", fname); return false; }
    for (int i = 0; i < n; i++) {
        if (fscanf(fp, "%f", &dst[i]) != 1) {
            fprintf(stderr, "[ERR] %s: EOF at index %d\n", fname, i);
            fclose(fp); return false;
        }
    }
    fclose(fp);
    printf("[load] %-30s  (%d values)\n", fname, n);
    return true;
}

// ----------------------------------------------------------------
// 讀 int8 txt（一行一值，十進位整數）
// ----------------------------------------------------------------
static bool read_int8(const char *fname, int8_t *dst, int n)
{
    FILE *fp = fopen(fname, "r");
    if (!fp) { fprintf(stderr, "[ERR] Cannot open %s\n", fname); return false; }
    for (int i = 0; i < n; i++) {
        int v;
        if (fscanf(fp, "%d", &v) != 1) {
            fprintf(stderr, "[ERR] %s: EOF at index %d\n", fname, i);
            fclose(fp); return false;
        }
        dst[i] = (int8_t)v;
    }
    fclose(fp);
    printf("[load] %-30s  (%d values)\n", fname, n);
    return true;
}

// ----------------------------------------------------------------
// 軟體黃金模型
// 完全模擬硬體量化流程：
//   1. 計算 mean / variance
//   2. normalize + gamma*x+beta → float norm
//   3. 找 max_abs → scale = 127 / max_abs
//   4. out[i][j] = (int8)(norm * scale)        ← truncate（與硬體一致）
//   5. scaler[i] = 1 / scale                   ← 反量化 scale 回傳給下一層
//
// 注意：硬體用 static_cast<int8_dt>(scaled_val)，是 truncate 不是 round
//       tb.cpp 的 golden 用 std::round，兩者差 1 LSB 是正常的
// ----------------------------------------------------------------
static void golden_layernorm(
    const float  x    [SEQ_LEN][EMBED_DIM],
    const float  gamma[EMBED_DIM],
    const float  beta [EMBED_DIM],
    int8_t       out  [SEQ_LEN][EMBED_DIM],
    float        scaler[SEQ_LEN])
{
    static float norm_buf[EMBED_DIM]; // 避免 stack 過大

    for (int i = 0; i < SEQ_LEN; i++) {
        // --- mean ---
        float mean = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++)
            mean += x[i][j];
        mean /= (float)EMBED_DIM;

        // --- variance ---
        float var = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) {
            float d = x[i][j] - mean;
            var += d * d;
        }
        var /= (float)EMBED_DIM;

        float inv_std = 1.0f / sqrtf(var + EPSILON);

        // --- normalize + affine, 找 max_abs ---
        float max_abs = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) {
            norm_buf[j] = gamma[j] * ((x[i][j] - mean) * inv_std) + beta[j];
            float a = fabsf(norm_buf[j]);
            if (a > max_abs) max_abs = a;
        }

        // --- quantize (truncate，與硬體 static_cast 行為一致）---
        float scale = (max_abs > 0.0f) ? (127.0f / max_abs) : 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) {
            float sv = norm_buf[j] * scale;
            // clamp to [-128, 127]
            if (sv >  127.0f) sv =  127.0f;
            if (sv < -128.0f) sv = -128.0f;
            out[i][j] = (int8_t)(int)sv;  // truncate toward zero，與 HLS static_cast 一致
        }

        // --- scaler 回傳給下一層 ---
        scaler[i] = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    }
}

// ----------------------------------------------------------------
// 比對 int8 陣列
// ----------------------------------------------------------------
struct Int8Stat {
    int max_err;
    int err_count;   // 超過 tolerance 的點數
    int total;
};

static Int8Stat compare_int8(
    const char    *label,
    const int8_t  *golden,
    const int8_t  *hw,
    int            n,
    int            tolerance = 1)   // ±1 LSB 容忍（float truncate vs 硬體 cast）
{
    Int8Stat s{0, 0, n};
    for (int i = 0; i < n; i++) {
        int err = abs((int)golden[i] - (int)hw[i]);
        if (err > s.max_err) s.max_err = err;
        if (err > tolerance) {
            s.err_count++;
            if (s.err_count <= 5)
                printf("  [%s] idx=%d  golden=%d  hw=%d  diff=%d\n",
                       label, i, (int)golden[i], (int)hw[i], err);
        }
    }
    return s;
}

// ----------------------------------------------------------------
// 比對 float 陣列（scaler）
// ----------------------------------------------------------------
struct FloatStat {
    float max_abs_err;
    float max_rel_err;
    int   err_count;
    int   total;
};

static FloatStat compare_float(
    const char  *label,
    const float *golden,
    const float *hw,
    int          n,
    float        abs_tol = 1e-4f,
    float        rel_tol = 1e-3f)
{
    FloatStat s{0.0f, 0.0f, 0, n};
    for (int i = 0; i < n; i++) {
        float diff = fabsf(golden[i] - hw[i]);
        float rel  = (fabsf(golden[i]) > 1e-8f) ? diff / fabsf(golden[i]) : diff;
        if (diff > s.max_abs_err) s.max_abs_err = diff;
        if (rel  > s.max_rel_err) s.max_rel_err = rel;
        if (diff > abs_tol && rel > rel_tol) {
            s.err_count++;
            if (s.err_count <= 5)
                printf("  [%s] idx=%d  golden=%.6e  hw=%.6e  abs=%.3e  rel=%.3e\n",
                       label, i, golden[i], hw[i], diff, rel);
        }
    }
    return s;
}

// ================================================================
int main()
{
    // ---- 靜態分配避免 stack overflow ----
    static float   x    [SEQ_LEN][EMBED_DIM];
    static float   gamma[EMBED_DIM];
    static float   beta [EMBED_DIM];

    // 黃金輸出
    static int8_t  g_out   [SEQ_LEN][EMBED_DIM];
    static float   g_scaler[SEQ_LEN];

    // 硬體輸出
    static int8_t  h_out   [SEQ_LEN][EMBED_DIM];
    static float   h_scaler[SEQ_LEN];

    // ---- 讀 inputs ----
    printf("=== Loading inputs ===\n");
    if (!read_float("input_x.txt",     &x[0][0], SEQ_LEN * EMBED_DIM)) return 1;
    if (!read_float("input_gamma.txt",  gamma,    EMBED_DIM))           return 1;
    if (!read_float("input_beta.txt",   beta,     EMBED_DIM))           return 1;

    // ---- 軟體黃金模型 ----
    printf("\n=== Running software golden model ===\n");
    golden_layernorm(x, gamma, beta, g_out, g_scaler);
    printf("Golden model done.\n");

    // ---- 讀硬體輸出 ----
    // 硬體輸出格式：
    //   output_out.txt    : 一行一個 int8（十進位），展開順序 out[i][j]（i 為外層）
    //   output_scaler.txt : 一行一個 float，共 SEQ_LEN=197 個
    printf("\n=== Loading hardware outputs ===\n");
    if (!read_int8 ("output_out.txt",    &h_out[0][0], SEQ_LEN * EMBED_DIM)) return 1;
    if (!read_float("output_scaler.txt",  h_scaler,    SEQ_LEN))             return 1;

    // ---- 比對 ----
    printf("\n=== Comparing outputs ===\n");

    // out 比對：容忍 ±1 LSB（truncate vs round 造成的差異）
    Int8Stat  so = compare_int8  ("out",    &g_out[0][0], &h_out[0][0],
                                  SEQ_LEN * EMBED_DIM, /*tolerance=*/1);
    // scaler 比對
    FloatStat ss = compare_float ("scaler",  g_scaler,     h_scaler,
                                  SEQ_LEN, /*abs_tol=*/1e-4f, /*rel_tol=*/1e-3f);

    printf("\n=== Summary ===\n");
    printf("  %-8s  max_err=%d         mismatch=%d/%d  %s\n",
           "out", so.max_err, so.err_count, so.total,
           so.err_count == 0 ? "PASS" : "FAIL");
    printf("  %-8s  max_abs=%.3e  max_rel=%.3e  mismatch=%d/%d  %s\n",
           "scaler", ss.max_abs_err, ss.max_rel_err, ss.err_count, ss.total,
           ss.err_count == 0 ? "PASS" : "FAIL");

    bool all_pass = (so.err_count == 0) && (ss.err_count == 0);
    printf("\n%s\n", all_pass ? ">>> ALL PASS <<<" : ">>> FAIL <<<");
    return all_pass ? 0 : 1;
}