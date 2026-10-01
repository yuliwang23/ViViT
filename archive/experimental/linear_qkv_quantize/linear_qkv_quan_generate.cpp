// 產生所有硬體 kernel 所需的 input .txt 檔案
// 矩陣格式：每行對應一個 row，數值以空白分隔
// 向量格式：一行，數值以空白分隔

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <ctime>

constexpr int SEQ_LEN   = 14 * 14 + 1;       // 197
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM  = EMBED_DIM / NUM_HEADS; // 64

// ----------------------------------------------------------------
// 量化：float → int8，回傳 scale
// ----------------------------------------------------------------
static void quantize_int8(const float *src, int8_t *dst, float *scale_out,
                           int rows, int cols, bool per_row_scale)
{
    if (per_row_scale) {
        // per-token (per-row)，用於 X
        for (int i = 0; i < rows; i++) {
            float max_abs = 1e-9f;
            for (int j = 0; j < cols; j++) {
                float v = fabsf(src[i * cols + j]);
                if (v > max_abs) max_abs = v;
            }
            float s = max_abs / 127.0f;
            scale_out[i] = s;
            for (int j = 0; j < cols; j++) {
                float q = src[i * cols + j] / s;
                if (q >  127.0f) q =  127.0f;
                if (q < -128.0f) q = -128.0f;
                dst[i * cols + j] = (int8_t)(int)roundf(q);
            }
        }
    } else {
        // per-output-channel (per-col)，用於 W
        for (int j = 0; j < cols; j++) {
            float max_abs = 1e-9f;
            for (int i = 0; i < rows; i++) {
                float v = fabsf(src[i * cols + j]);
                if (v > max_abs) max_abs = v;
            }
            float s = max_abs / 127.0f;
            scale_out[j] = s;
            for (int i = 0; i < rows; i++) {
                float q = src[i * cols + j] / s;
                if (q >  127.0f) q =  127.0f;
                if (q < -128.0f) q = -128.0f;
                dst[i * cols + j] = (int8_t)(int)roundf(q);
            }
        }
    }
}

// ----------------------------------------------------------------
// 寫 int8 矩陣：每行對應一個 row，數值空白分隔
// host 的 load_int8_matrix 用 getline + stringstream 讀，
// 所以必須是這個格式。
// ----------------------------------------------------------------
static void write_int8_matrix(const char *fname, const int8_t *data,
                               int rows, int cols)
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", fname); return; }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (j > 0) fprintf(fp, " ");
            fprintf(fp, "%d", (int)data[i * cols + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("[gen] %s  (%d x %d)\n", fname, rows, cols);
}

// ----------------------------------------------------------------
// 寫 float 向量：一行，數值空白分隔
// host 的 load_float_vector 用 f >> 讀，格式不限，
// 但一行比較清楚。
// ----------------------------------------------------------------
static void write_float_vector(const char *fname, const float *data, int n)
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", fname); return; }
    for (int i = 0; i < n; i++) {
        if (i > 0) fprintf(fp, " ");
        fprintf(fp, "%.9e", data[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
    printf("[gen] %s  (%d values)\n", fname, n);
}

// ----------------------------------------------------------------
// 隨機 float，範圍 [lo, hi]
// ----------------------------------------------------------------
static float randf(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

// ================================================================
int main(int argc, char *argv[])
{
    unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : 42;
    srand(seed);
    printf("Seed = %u\n", seed);

    // ---- fp32 工作陣列 ----
    static float  fX [SEQ_LEN  ][EMBED_DIM];
    static float  fWq[EMBED_DIM][EMBED_DIM];
    static float  fWk[EMBED_DIM][EMBED_DIM];
    static float  fWv[EMBED_DIM][EMBED_DIM];
    static float  fbq[EMBED_DIM];
    static float  fbk[EMBED_DIM];
    static float  fbv[EMBED_DIM];

    // ---- 量化陣列 ----
    static int8_t qX [SEQ_LEN  ][EMBED_DIM];
    static int8_t qWq[EMBED_DIM][EMBED_DIM];
    static int8_t qWk[EMBED_DIM][EMBED_DIM];
    static int8_t qWv[EMBED_DIM][EMBED_DIM];
    static float  scale_input[SEQ_LEN];
    static float  scale_wq   [EMBED_DIM];
    static float  scale_wk   [EMBED_DIM];
    static float  scale_wv   [EMBED_DIM];

    // ---- 生成隨機資料 ----
    for (int i = 0; i < SEQ_LEN;   i++)
        for (int j = 0; j < EMBED_DIM; j++)
            fX[i][j] = randf(-1.0f, 1.0f);

    for (int i = 0; i < EMBED_DIM; i++)
        for (int j = 0; j < EMBED_DIM; j++) {
            fWq[i][j] = randf(-0.1f, 0.1f);
            fWk[i][j] = randf(-0.1f, 0.1f);
            fWv[i][j] = randf(-0.1f, 0.1f);
        }

    for (int j = 0; j < EMBED_DIM; j++) {
        fbq[j] = randf(-0.01f, 0.01f);
        fbk[j] = randf(-0.01f, 0.01f);
        fbv[j] = randf(-0.01f, 0.01f);
    }

    // ---- 量化 ----
    quantize_int8(&fX[0][0],  &qX[0][0],  scale_input,
                  SEQ_LEN,   EMBED_DIM, /*per_row=*/true);
    quantize_int8(&fWq[0][0], &qWq[0][0], scale_wq,
                  EMBED_DIM, EMBED_DIM, /*per_row=*/false);
    quantize_int8(&fWk[0][0], &qWk[0][0], scale_wk,
                  EMBED_DIM, EMBED_DIM, /*per_row=*/false);
    quantize_int8(&fWv[0][0], &qWv[0][0], scale_wv,
                  EMBED_DIM, EMBED_DIM, /*per_row=*/false);

    // ---- 寫出 ----
    // 矩陣：每行一個 row（對應 host load_int8_matrix 的讀法）
    write_int8_matrix("X.txt",  &qX[0][0],  SEQ_LEN,   EMBED_DIM);
    write_int8_matrix("Wq.txt", &qWq[0][0], EMBED_DIM, EMBED_DIM);
    write_int8_matrix("Wk.txt", &qWk[0][0], EMBED_DIM, EMBED_DIM);
    write_int8_matrix("Wv.txt", &qWv[0][0], EMBED_DIM, EMBED_DIM);

    // 向量：一行（對應 host load_float_vector 的讀法）
    write_float_vector("bq.txt",          fbq,         EMBED_DIM);
    write_float_vector("bk.txt",          fbk,         EMBED_DIM);
    write_float_vector("bv.txt",          fbv,         EMBED_DIM);
    write_float_vector("scale_input.txt", scale_input, SEQ_LEN);
    write_float_vector("scale_wq.txt",    scale_wq,    EMBED_DIM);
    write_float_vector("scale_wk.txt",    scale_wk,    EMBED_DIM);
    write_float_vector("scale_wv.txt",    scale_wv,    EMBED_DIM);

    printf("\nAll input files generated (seed=%u).\n", seed);
    return 0;
}