// ============================================================
// C-sim Testbench for linear_qkv_multihead
//
// 放在 Vitis HLS 專案的 tb/ 資料夾
// Project Settings -> Simulation -> TestBench Files 加入此檔
//
// 編譯方式（C-sim 時 Vitis HLS 會自動處理）
// 手動測試: g++ -O2 -std=c++17 -I../src tb_linear_qkv_multihead.cpp -o tb
// ============================================================

#include "linear_qkv_multihead.h"  // 你的 HLS header
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================
// 靜態陣列（C-sim 裡 stack 很小，一律用 static）
// ============================================================
static int X    [SEQ_LEN][EMBED_DIM];
static int Wq   [EMBED_DIM][EMBED_DIM];
static int Wk   [EMBED_DIM][EMBED_DIM];
static int Wv   [EMBED_DIM][EMBED_DIM];
static int bq   [EMBED_DIM];
static int bk   [EMBED_DIM];
static int bv   [EMBED_DIM];
static int Q_out[NUM_HEADS][SEQ_LEN][HEAD_DIM];
static int K_out[NUM_HEADS][SEQ_LEN][HEAD_DIM];
static int V_out[NUM_HEADS][SEQ_LEN][HEAD_DIM];

// golden reference（未 reshape，SEQ_LEN x EMBED_DIM）
static int Q_ref[SEQ_LEN][EMBED_DIM];
static int K_ref[SEQ_LEN][EMBED_DIM];
static int V_ref[SEQ_LEN][EMBED_DIM];

// ============================================================
// 工具：重置所有陣列
// ============================================================
void reset_all() {
    memset(X,     0, sizeof(X));
    memset(Wq,    0, sizeof(Wq));
    memset(Wk,    0, sizeof(Wk));
    memset(Wv,    0, sizeof(Wv));
    memset(bq,    0, sizeof(bq));
    memset(bk,    0, sizeof(bk));
    memset(bv,    0, sizeof(bv));
    memset(Q_out, 0, sizeof(Q_out));
    memset(K_out, 0, sizeof(K_out));
    memset(V_out, 0, sizeof(V_out));
    memset(Q_ref, 0, sizeof(Q_ref));
    memset(K_ref, 0, sizeof(K_ref));
    memset(V_ref, 0, sizeof(V_ref));
}

// ============================================================
// 工具：software golden reference
// out[s][j] = bias[j] + sum_k( X[s][k] * W[k][j] )
// ============================================================
void matmul_ref(
    int X_in [SEQ_LEN][EMBED_DIM],
    int W_in [EMBED_DIM][EMBED_DIM],
    int bias [EMBED_DIM],
    int out  [SEQ_LEN][EMBED_DIM])
{
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++) {
            int acc = bias[j];
            for (int k = 0; k < EMBED_DIM; k++)
                acc += X_in[s][k] * W_in[k][j];
            out[s][j] = acc;
        }
}

// ============================================================
// 工具：把 HLS 輸出 [NUM_HEADS][SEQ_LEN][HEAD_DIM]
//        攤平成   [SEQ_LEN][EMBED_DIM]
// 假設排列：Q_flat[s][h*HEAD_DIM + d] = Q_out[h][s][d]
// ============================================================
void flatten(
    int src[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int dst[SEQ_LEN][EMBED_DIM])
{
    for (int h = 0; h < NUM_HEADS; h++)
        for (int s = 0; s < SEQ_LEN; s++)
            for (int d = 0; d < HEAD_DIM; d++)
                dst[s][h * HEAD_DIM + d] = src[h][s][d];
}

// ============================================================
// 工具：比較 ref vs got，回傳不符合的數量
// ============================================================
struct Stat { int max_err; long long sum_err; int mismatch; };

Stat compare(int ref[SEQ_LEN][EMBED_DIM], int got[SEQ_LEN][EMBED_DIM]) {
    Stat s = {0, 0, 0};
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++) {
            int diff = abs(ref[i][j] - got[i][j]);
            if (diff > s.max_err) s.max_err = diff;
            s.sum_err += diff;
            if (diff != 0) s.mismatch++;
        }
    return s;
}

void print_stat(const char* name, Stat& s) {
    bool pass = (s.mismatch == 0);
    printf("[%s] %s  max_err=%d  total_mismatch=%d/%d\n",
           pass ? "PASS" : "FAIL", name,
           s.max_err, s.mismatch, SEQ_LEN * EMBED_DIM);
}

// 印前幾個元素對照（debug 用）
void dump(const char* label,
          int ref[SEQ_LEN][EMBED_DIM],
          int got[SEQ_LEN][EMBED_DIM],
          int n = 8)
{
    printf("  %s ref[0][0..%d]:", label, n-1);
    for (int j = 0; j < n; j++) printf(" %6d", ref[0][j]);
    printf("\n");
    printf("  %s got[0][0..%d]:", label, n-1);
    for (int j = 0; j < n; j++) printf(" %6d", got[0][j]);
    printf("\n");
}

// ============================================================
// 呼叫 DUT（HLS top function）
// ============================================================
void run_dut() {
    linear_qkv_multihead(
        X, Wq, bq, Wk, bk, Wv, bv,
        Q_out, K_out, V_out
    );
}

// ============================================================
// TEST 1：Wq=Wk=Wv=Identity, bias=0
//   期望：Q_flat == K_flat == V_flat == X
// ============================================================
void test_identity() {
    printf("\n=== TEST 1: W=Identity, bias=0  (expect output == X) ===\n");
    reset_all();

    // X = 小整數（固定 seed，確保可重現）
    srand(42);
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[s][j] = rand() % 21 - 10;  // -10 ~ 10

    // W = Identity
    for (int i = 0; i < EMBED_DIM; i++)
        Wq[i][i] = Wk[i][i] = Wv[i][i] = 1;

    // golden
    matmul_ref(X, Wq, bq, Q_ref);
    matmul_ref(X, Wk, bk, K_ref);
    matmul_ref(X, Wv, bv, V_ref);

    run_dut();

    int Q_flat[SEQ_LEN][EMBED_DIM];
    int K_flat[SEQ_LEN][EMBED_DIM];
    int V_flat[SEQ_LEN][EMBED_DIM];
    flatten(Q_out, Q_flat);
    flatten(K_out, K_flat);
    flatten(V_out, V_flat);

    Stat sq = compare(Q_ref, Q_flat);
    Stat sk = compare(K_ref, K_flat);
    Stat sv = compare(V_ref, V_flat);

    print_stat("Q Identity", sq); dump("Q", Q_ref, Q_flat);
    print_stat("K Identity", sk);
    print_stat("V Identity", sv);

    if (sq.mismatch > 0) {
        // 判斷是轉置還是亂值
        // 若是轉置，got[0][j] 應等於 X[j][0]（對小矩陣適用）
        printf("  HINT: 若 got 看起來像轉置 => row/col 方向錯誤\n");
        printf("  HINT: 若 got 是亂值       => 地址計算根本錯誤\n");
    }
}

// ============================================================
// TEST 2：只有 Wq[0][:]=1，其餘全 0，bias=0
//   期望：Q_flat[s][j] = X[s][0]  for all j
// ============================================================
void test_single_row() {
    printf("\n=== TEST 2: Wq[0][:]=1  (expect Q[s][j] = X[s][0]) ===\n");
    reset_all();

    srand(42);
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[s][j] = rand() % 21 - 10;

    for (int j = 0; j < EMBED_DIM; j++)
        Wq[0][j] = 1;
    // Wk, Wv 維持 0（輸出應該全 0 + bias）

    matmul_ref(X, Wq, bq, Q_ref);

    run_dut();

    int Q_flat[SEQ_LEN][EMBED_DIM];
    flatten(Q_out, Q_flat);
    Stat sq = compare(Q_ref, Q_flat);
    print_stat("Q SingleRow", sq);
    dump("Q SingleRow", Q_ref, Q_flat);

    if (sq.mismatch > 0) {
        // 若 got[s][j] = X[s][j]（沒有把所有 col 統一成同一值）
        // 代表 Wq 被當成 Identity 讀，即 row/col 搞反
        printf("  HINT: 若 got[s][j] != X[s][0] 但 got[s][j] == X[s][j]\n");
        printf("        => FIFO2BRAM 的 outer/inner loop 方向錯誤（轉置問題）\n");
    }
}

// ============================================================
// TEST 3：TILE 邊界，只有 Wq[TILE_SIZE][:]=1，bias=0
//   期望：Q_flat[s][j] = X[s][TILE_SIZE]  for all j
//   這測試跨越第一個 k_tile 邊界
// ============================================================
void test_tile_boundary() {
    printf("\n=== TEST 3: Wq[TILE_SIZE][:]=1  (second k_tile, row 0) ===\n");
    reset_all();

    srand(42);
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[s][j] = rand() % 21 - 10;

    for (int j = 0; j < EMBED_DIM; j++)
        Wq[TILE_SIZE][j] = 1;

    matmul_ref(X, Wq, bq, Q_ref);

    run_dut();

    int Q_flat[SEQ_LEN][EMBED_DIM];
    flatten(Q_out, Q_flat);
    Stat sq = compare(Q_ref, Q_flat);
    print_stat("Q TileBoundary", sq);
    dump("Q TileBoundary", Q_ref, Q_flat);

    if (sq.mismatch > 0) {
        printf("  HINT: TEST 1 PASS 但這裡 FAIL => 跨 tile 的 base_addr_idx 計算錯誤\n");
        printf("        檢查: base_addr_idx = row_offset * global_stride + col_offset/8\n");
    }
}

// ============================================================
// TEST 4：bias 測試，W=0，bias 有值
//   期望：Q_flat[s][j] = bq[j]  for all s
// ============================================================
void test_bias_only() {
    printf("\n=== TEST 4: W=0, bias=nonzero  (expect output == broadcast bias) ===\n");
    reset_all();

    srand(42);
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[s][j] = rand() % 21 - 10;

    for (int j = 0; j < EMBED_DIM; j++) {
        bq[j] = j + 1;   // 1 ~ 768
        bk[j] = j + 1;
        bv[j] = j + 1;
    }

    matmul_ref(X, Wq, bq, Q_ref);  // W=0, 所以 Q_ref[s][j] = bq[j]

    run_dut();

    int Q_flat[SEQ_LEN][EMBED_DIM];
    flatten(Q_out, Q_flat);
    Stat sq = compare(Q_ref, Q_flat);
    print_stat("Q BiasOnly", sq);
    dump("Q BiasOnly", Q_ref, Q_flat);

    if (sq.mismatch > 0) {
        printf("  HINT: bias 初始化（init_acc_loop）或 j_tile offset 有誤\n");
        printf("        確認 BRAM_Acc[i][j] 初始化為 BRAM_bq[j_tile + j]\n");
    }
}

// ============================================================
// TEST 5：逐 j_tile 隔離（找出哪個輸出 tile 開始出錯）
//   每次只讓 Wq 的對應 col range 有值
// ============================================================
void test_each_jtile() {
    printf("\n=== TEST 5: Per j_tile isolation ===\n");
    int num_tiles = EMBED_DIM / TILE_SIZE;

    srand(42);
    // 先產生固定 X
    int X_fixed[SEQ_LEN][EMBED_DIM];
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X_fixed[s][j] = rand() % 21 - 10;

    for (int t = 0; t < num_tiles; t++) {
        reset_all();
        memcpy(X, X_fixed, sizeof(X));

        // Wq: 只有 col [t*TILE_SIZE .. (t+1)*TILE_SIZE-1] 有值
        int col_start = t * TILE_SIZE;
        for (int i = 0; i < EMBED_DIM; i++)
            for (int j = col_start; j < col_start + TILE_SIZE; j++)
                Wq[i][j] = 1;

        matmul_ref(X, Wq, bq, Q_ref);
        run_dut();

        int Q_flat[SEQ_LEN][EMBED_DIM];
        flatten(Q_out, Q_flat);
        Stat sq = compare(Q_ref, Q_flat);

        char name[64];
        snprintf(name, sizeof(name), "Q j_tile=%d (col %d..%d)",
                 t, col_start, col_start + TILE_SIZE - 1);
        print_stat(name, sq);
    }
}

// ============================================================
// TEST 6：逐 k_tile 隔離（找出哪個輸入 tile 讀錯）
//   每次只讓 Wq 的對應 row range 有值
// ============================================================
void test_each_ktile() {
    printf("\n=== TEST 6: Per k_tile isolation (weight row direction) ===\n");
    int num_tiles = EMBED_DIM / TILE_SIZE;

    srand(42);
    int X_fixed[SEQ_LEN][EMBED_DIM];
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X_fixed[s][j] = rand() % 21 - 10;

    for (int t = 0; t < num_tiles; t++) {
        reset_all();
        memcpy(X, X_fixed, sizeof(X));

        // Wq: 只有 row [t*TILE_SIZE .. (t+1)*TILE_SIZE-1] 有值
        int row_start = t * TILE_SIZE;
        for (int i = row_start; i < row_start + TILE_SIZE; i++)
            for (int j = 0; j < EMBED_DIM; j++)
                Wq[i][j] = 1;

        matmul_ref(X, Wq, bq, Q_ref);
        run_dut();

        int Q_flat[SEQ_LEN][EMBED_DIM];
        flatten(Q_out, Q_flat);
        Stat sq = compare(Q_ref, Q_flat);

        char name[64];
        snprintf(name, sizeof(name), "Q k_tile=%d (row %d..%d)",
                 t, row_start, row_start + TILE_SIZE - 1);
        print_stat(name, sq);
    }
}

// ============================================================
// TEST 7：完整隨機測試（最終驗證）
// ============================================================
void test_random() {
    printf("\n=== TEST 7: Full random test ===\n");
    reset_all();

    srand(123);
    for (int s = 0; s < SEQ_LEN; s++)
        for (int j = 0; j < EMBED_DIM; j++)
            X[s][j] = rand() % 11 - 5;     // 小值避免 int 溢位

    for (int i = 0; i < EMBED_DIM; i++)
        for (int j = 0; j < EMBED_DIM; j++) {
            Wq[i][j] = rand() % 7 - 3;
            Wk[i][j] = rand() % 7 - 3;
            Wv[i][j] = rand() % 7 - 3;
        }

    for (int j = 0; j < EMBED_DIM; j++) {
        bq[j] = rand() % 21 - 10;
        bk[j] = rand() % 21 - 10;
        bv[j] = rand() % 21 - 10;
    }

    matmul_ref(X, Wq, bq, Q_ref);
    matmul_ref(X, Wk, bk, K_ref);
    matmul_ref(X, Wv, bv, V_ref);

    run_dut();

    int Q_flat[SEQ_LEN][EMBED_DIM];
    int K_flat[SEQ_LEN][EMBED_DIM];
    int V_flat[SEQ_LEN][EMBED_DIM];
    flatten(Q_out, Q_flat);
    flatten(K_out, K_flat);
    flatten(V_out, V_flat);

    Stat sq = compare(Q_ref, Q_flat);
    Stat sk = compare(K_ref, K_flat);
    Stat sv = compare(V_ref, V_flat);

    print_stat("Q Random", sq);
    print_stat("K Random", sk);
    print_stat("V Random", sv);
}

// ============================================================
// MAIN
// ============================================================
int main() {
    printf("========================================\n");
    printf("  C-sim Testbench: linear_qkv_multihead\n");
    printf("  EMBED_DIM=%d  SEQ_LEN=%d  TILE_SIZE=%d\n",
           EMBED_DIM, SEQ_LEN, TILE_SIZE);
    printf("  NUM_HEADS=%d  HEAD_DIM=%d\n", NUM_HEADS, HEAD_DIM);
    printf("========================================\n");

    test_identity();       // 最基本：W=I 時 output==input
    test_single_row();     // 診斷 row/col 方向
    test_tile_boundary();  // 診斷跨 tile 地址
    test_bias_only();      // 診斷 bias 初始化
    test_each_jtile();     // 逐 output tile 隔離
    test_each_ktile();     // 逐 input tile 隔離
    test_random();         // 完整隨機驗證

    printf("\n========================================\n");
    printf("  Done.\n");
    printf("========================================\n");
    return 0;
}