// gen_input_layernorm.cpp
// 產生 layernorm kernel 所需的所有 input .txt 檔案
// 編譯: g++ -O2 -o gen_input gen_input_layernorm.cpp
// 執行: ./gen_input [seed]  (seed 預設 42)

#include <cstdio>
#include <cstdlib>
#include <cmath>

constexpr int SEQ_LEN   = 197;
constexpr int EMBED_DIM = 768;

// ----------------------------------------------------------------
// 工具函式
// ----------------------------------------------------------------
static float randf(float lo, float hi)
{
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void write_float(const char *fname, const float *data, int n)
{
    FILE *fp = fopen(fname, "w");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", fname); return; }
    for (int i = 0; i < n; i++)
        fprintf(fp, "%.9e\n", data[i]);
    fclose(fp);
    printf("[gen] %-30s  (%d values)\n", fname, n);
}

// ================================================================
int main(int argc, char *argv[])
{
    unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : 42;
    srand(seed);
    printf("Seed = %u\n\n", seed);

    // ---- 靜態分配避免 stack overflow ----
    static float X    [SEQ_LEN][EMBED_DIM];
    static float gamma[EMBED_DIM];
    static float beta [EMBED_DIM];

    // ---- 生成隨機資料（與 tb.cpp init_test_data 相同的分佈）----
    // X: 模擬 raw feature map [-15, 15]
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[i][j] = randf(-15.0f, 15.0f);

    // gamma: 訓練後通常接近 1.0 [0.5, 2.0]
    for (int j = 0; j < EMBED_DIM; j++)
        gamma[j] = randf(0.5f, 2.0f);

    // beta: 訓練後通常接近 0.0 [-1.0, 1.0]
    for (int j = 0; j < EMBED_DIM; j++)
        beta[j] = randf(-1.0f, 1.0f);

    // ---- 寫出 .txt ----
    write_float("input_x.txt",     &X[0][0], SEQ_LEN * EMBED_DIM);
    write_float("input_gamma.txt",  gamma,    EMBED_DIM);
    write_float("input_beta.txt",   beta,     EMBED_DIM);

    printf("\nAll input files generated (seed=%u).\n", seed);
    printf("Files: input_x.txt  input_gamma.txt  input_beta.txt\n");
    return 0;
}