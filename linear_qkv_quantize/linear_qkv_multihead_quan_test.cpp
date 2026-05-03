// 1. 讀入所有 input .txt（與 gen_input.cpp 生成的相同）
// 2. 用軟體黃金模型（依照硬體量化流程）計算 Q/K/V
// 3. 讀入硬體跑完產生的 output_Q.txt / output_K.txt / output_V.txt
// 4. 逐元素比對，報告最大誤差與 pass/fail

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cfloat>

// ----------------------------------------------------------------
// 維度（與硬體 header 一致）
// ----------------------------------------------------------------
constexpr int SEQ_LEN   = 14 * 14 + 1;        // 197
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM  = EMBED_DIM / NUM_HEADS; // 64

// ----------------------------------------------------------------
// 讀 int8 txt
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
    printf("[load] %s  (%d values)\n", fname, n);
    return true;
}

// ----------------------------------------------------------------
// 讀 float txt
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
    printf("[load] %s  (%d values)\n", fname, n);
    return true;
}

// ----------------------------------------------------------------
// 軟體黃金模型
//   完全模擬硬體量化流程：
//     acc (int32) = sum_e { X_q[i][e] * W_q[e][col] }
//     output (float) = acc * scale_input[i] * scale_w[col] + bias[col]
//   其中 X_q, W_q 已為 int8（由 gen_input.cpp 量化完畢後讀入）
// ----------------------------------------------------------------
static void golden_qkv(
    const int8_t  X [SEQ_LEN ][EMBED_DIM],
    const int8_t  Wq[EMBED_DIM][EMBED_DIM],
    const int8_t  Wk[EMBED_DIM][EMBED_DIM],
    const int8_t  Wv[EMBED_DIM][EMBED_DIM],
    const float   bq[EMBED_DIM],
    const float   bk[EMBED_DIM],
    const float   bv[EMBED_DIM],
    const float   scale_input[SEQ_LEN],
    const float   scale_wq   [EMBED_DIM],
    const float   scale_wk   [EMBED_DIM],
    const float   scale_wv   [EMBED_DIM],
    float         Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float         K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float         V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {
        const float si = scale_input[i];
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int col = h * HEAD_DIM + d;

                int32_t acc_q = 0, acc_k = 0, acc_v = 0;
                for (int e = 0; e < EMBED_DIM; e++) {
                    acc_q += (int32_t)X[i][e] * (int32_t)Wq[e][col];
                    acc_k += (int32_t)X[i][e] * (int32_t)Wk[e][col];
                    acc_v += (int32_t)X[i][e] * (int32_t)Wv[e][col];
                }

                Q[h][i][d] = (float)acc_q * si * scale_wq[col] + bq[col];
                K[h][i][d] = (float)acc_k * si * scale_wk[col] + bk[col];
                V[h][i][d] = (float)acc_v * si * scale_wv[col] + bv[col];
            }
        }
    }
}

// ----------------------------------------------------------------
// 比對函式：回傳 false 表示超過容忍誤差
// ----------------------------------------------------------------
struct DiffStat {
    float max_abs_err;
    float max_rel_err;
    int   err_count;     // 超過 abs_tol 的點數
    int   total;
};

static DiffStat compare(
    const char  *label,
    const float *golden,
    const float *hw,
    int          n,
    float        abs_tol = 1e-3f,   // 絕對誤差容忍
    float        rel_tol = 1e-2f)   // 相對誤差容忍
{
    DiffStat s{0.0f, 0.0f, 0, n};
    for (int i = 0; i < n; i++) {
        float diff = fabsf(golden[i] - hw[i]);
        float rel  = (fabsf(golden[i]) > 1e-8f) ?
                     diff / fabsf(golden[i]) : diff;
        if (diff > s.max_abs_err) s.max_abs_err = diff;
        if (rel  > s.max_rel_err) s.max_rel_err = rel;
        if (diff > abs_tol && rel > rel_tol) {
            s.err_count++;
            // 只印前 5 個異常點
            if (s.err_count <= 5)
                printf("  [%s] idx=%d  golden=%.6e  hw=%.6e  abs_err=%.3e  rel_err=%.3e\n",
                       label, i, golden[i], hw[i], diff, rel);
        }
    }
    return s;
}

// ================================================================
int main()
{
    // ---- 動態分配避免 stack overflow ----
    static int8_t  X [SEQ_LEN ][EMBED_DIM];
    static int8_t  Wq[EMBED_DIM][EMBED_DIM];
    static int8_t  Wk[EMBED_DIM][EMBED_DIM];
    static int8_t  Wv[EMBED_DIM][EMBED_DIM];
    static float   bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM];
    static float   scale_input[SEQ_LEN];
    static float   scale_wq[EMBED_DIM], scale_wk[EMBED_DIM], scale_wv[EMBED_DIM];

    // 黃金輸出
    static float gQ[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float gK[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float gV[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    // 硬體輸出
    static float hQ[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float hK[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float hV[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    // ---- 讀 input ----
    printf("=== Loading inputs ===\n");
    if (!read_int8 ("X.txt",           &X[0][0],   SEQ_LEN * EMBED_DIM))    return 1;
    if (!read_int8 ("Wq.txt",          &Wq[0][0],  EMBED_DIM * EMBED_DIM))  return 1;
    if (!read_int8 ("Wk.txt",          &Wk[0][0],  EMBED_DIM * EMBED_DIM))  return 1;
    if (!read_int8 ("Wv.txt",          &Wv[0][0],  EMBED_DIM * EMBED_DIM))  return 1;
    if (!read_float("bq.txt",          bq,          EMBED_DIM))             return 1;
    if (!read_float("bk.txt",          bk,          EMBED_DIM))             return 1;
    if (!read_float("bv.txt",          bv,          EMBED_DIM))             return 1;
    if (!read_float("scale_input.txt", scale_input, SEQ_LEN))               return 1;
    if (!read_float("scale_wq.txt",    scale_wq,    EMBED_DIM))             return 1;
    if (!read_float("scale_wk.txt",    scale_wk,    EMBED_DIM))             return 1;
    if (!read_float("scale_wv.txt",    scale_wv,    EMBED_DIM))             return 1;

    // ---- 軟體黃金模型 ----
    printf("\n=== Running software golden model ===\n");
    golden_qkv(X, Wq, Wk, Wv,
               bq, bk, bv,
               scale_input, scale_wq, scale_wk, scale_wv,
               gQ, gK, gV);
    printf("Golden model done.\n");

    // ---- 讀硬體輸出 ----
    // 硬體輸出格式：output_Q.txt / output_K.txt / output_V.txt
    // 儲存順序與 float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM] 的線性展開相同
    // (即 Q[h][i][d]，h 為最外層)
    printf("\n=== Loading hardware outputs ===\n");
    const int OUT_SIZE = NUM_HEADS * SEQ_LEN * HEAD_DIM;
    if (!read_float("output_Q.txt", &hQ[0][0][0], OUT_SIZE)) return 1;
    if (!read_float("output_K.txt", &hK[0][0][0], OUT_SIZE)) return 1;
    if (!read_float("output_V.txt", &hV[0][0][0], OUT_SIZE)) return 1;

    // ---- 比對 ----
    printf("\n=== Comparing Q/K/V ===\n");

    // 可依需要調整容忍度
    // int8 量化本身有精度損失，abs_tol 設稍寬
    const float ABS_TOL = 5e-3f;
    const float REL_TOL = 1e-2f;

    DiffStat sq = compare("Q", &gQ[0][0][0], &hQ[0][0][0], OUT_SIZE, ABS_TOL, REL_TOL);
    DiffStat sk = compare("K", &gK[0][0][0], &hK[0][0][0], OUT_SIZE, ABS_TOL, REL_TOL);
    DiffStat sv = compare("V", &gV[0][0][0], &hV[0][0][0], OUT_SIZE, ABS_TOL, REL_TOL);

    printf("\n=== Summary ===\n");
    printf("  %-4s  max_abs_err=%.3e  max_rel_err=%.3e  mismatch=%d/%d  %s\n",
           "Q", sq.max_abs_err, sq.max_rel_err, sq.err_count, sq.total,
           sq.err_count == 0 ? "PASS" : "FAIL");
    printf("  %-4s  max_abs_err=%.3e  max_rel_err=%.3e  mismatch=%d/%d  %s\n",
           "K", sk.max_abs_err, sk.max_rel_err, sk.err_count, sk.total,
           sk.err_count == 0 ? "PASS" : "FAIL");
    printf("  %-4s  max_abs_err=%.3e  max_rel_err=%.3e  mismatch=%d/%d  %s\n",
           "V", sv.max_abs_err, sv.max_rel_err, sv.err_count, sv.total,
           sv.err_count == 0 ? "PASS" : "FAIL");

    bool all_pass = (sq.err_count == 0) && (sk.err_count == 0) && (sv.err_count == 0);
    printf("\n%s\n", all_pass ? ">>> ALL PASS <<<" : ">>> FAIL <<<");
    return all_pass ? 0 : 1;
}