#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include "linear_qkv_multihead_quan.h"

using namespace std;

// 初始化 INT8 矩陣 (模擬量化後的權重與輸入)
void init_matrix_int8(int8_t mat[SEQ_LEN][EMBED_DIM]) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            mat[i][j] = (int8_t)(rand() % 255 - 128); // -128 ~ 127
        }
    }
}

void init_weight_int8(int8_t mat[EMBED_DIM][EMBED_DIM]) {
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            mat[i][j] = (int8_t)(rand() % 41 - 20); // 權重通常較小
        }
    }
}

// 初始化 Float 向量 (Bias 與 Scales)
void init_vector_float(float vec[], int size, float min_val, float max_val) {
    for (int i = 0; i < size; i++) {
        vec[i] = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
    }
}

// ─── Golden Reference (模擬硬體量化邏輯) ───
void linear_qkv_golden(
    int8_t X[SEQ_LEN][EMBED_DIM],
    int8_t Wq[EMBED_DIM][EMBED_DIM], float bq[EMBED_DIM], float sq[EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM], float bk[EMBED_DIM], float sk[EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM], float bv[EMBED_DIM], float sv[EMBED_DIM],
    float si[SEQ_LEN],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            int32_t acc_q = 0, acc_k = 0, acc_v = 0;
            
            // 1. 執行 INT8 矩陣乘法 (Accumulate in INT32)
            for (int k = 0; k < EMBED_DIM; k++) {
                acc_q += (int32_t)X[i][k] * (int32_t)Wq[k][j];
                acc_k += (int32_t)X[i][k] * (int32_t)Wk[k][j];
                acc_v += (int32_t)X[i][k] * (int32_t)Wv[k][j];
            }

            // 2. 反量化與加 Bias (Dequant: ACC * scale_in * scale_w + bias)
            float res_q = (float)acc_q * si[i] * sq[j] + bq[j];
            float res_k = (float)acc_k * si[i] * sk[j] + bk[j];
            float res_v = (float)acc_v * si[i] * sv[j] + bv[j];

            // 3. 映射至 Multi-head 輸出格式
            int head_idx = j / HEAD_DIM;
            int head_dim = j % HEAD_DIM;
            Q[head_idx][i][head_dim] = res_q;
            K[head_idx][i][head_dim] = res_k;
            V[head_idx][i][head_dim] = res_v;
        }
    }
}

int main() {
    srand(42); // 固定 Seed 方便除錯

    // 使用 static 避免 Stack Overflow (資料量約數 MB)
    static int8_t h_X[SEQ_LEN][EMBED_DIM];
    static int8_t h_Wq[EMBED_DIM][EMBED_DIM], h_Wk[EMBED_DIM][EMBED_DIM], h_Wv[EMBED_DIM][EMBED_DIM];
    static int8_t h_Wq_T[EMBED_DIM][EMBED_DIM], h_Wk_T[EMBED_DIM][EMBED_DIM], h_Wv_T[EMBED_DIM][EMBED_DIM];
    static float h_bq[EMBED_DIM], h_bk[EMBED_DIM], h_bv[EMBED_DIM];
    static float h_si[SEQ_LEN];
    static float h_sq[EMBED_DIM], h_sk[EMBED_DIM], h_sv[EMBED_DIM];

    static float h_Q_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float h_K_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float h_V_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    
    static float h_Q_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float h_K_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float h_V_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    cout << "Initializing data..." << endl;
    init_matrix_int8(h_X);
    init_weight_int8(h_Wq); init_weight_int8(h_Wk); init_weight_int8(h_Wv);
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_Wq_T[j][i] = h_Wq[i][j];
            h_Wk_T[j][i] = h_Wk[i][j];
            h_Wv_T[j][i] = h_Wv[i][j];
        }
    }
    
    init_vector_float(h_bq, EMBED_DIM, -1.0f, 1.0f);
    init_vector_float(h_bk, EMBED_DIM, -1.0f, 1.0f);
    init_vector_float(h_bv, EMBED_DIM, -1.0f, 1.0f);
    
    init_vector_float(h_si, SEQ_LEN, 0.001f, 0.05f); // Scale Input
    init_vector_float(h_sq, EMBED_DIM, 0.001f, 0.02f); // Scale Weight
    init_vector_float(h_sk, EMBED_DIM, 0.001f, 0.02f);
    init_vector_float(h_sv, EMBED_DIM, 0.001f, 0.02f);

    // 2. 計算 Golden Reference
    cout << "Calculating Golden Reference..." << endl;
    linear_qkv_golden(h_X, h_Wq, h_bq, h_sq, h_Wk, h_bk, h_sk, h_Wv, h_bv, h_sv, h_si, h_Q_ref, h_K_ref, h_V_ref);

    // 3. 執行 HLS Kernel
    cout << "Running HLS Kernel with Transposed Weights..." << endl;
    linear_qkv_multihead(
        h_X, 
        h_Wq_T, h_bq, 
        h_Wk_T, h_bk, 
        h_Wv_T, h_bv,
        h_si, h_sq, h_sk, h_sv,
        h_Q_opt, h_K_opt, h_V_opt
    );

    // 4. 比對結果
    float max_err = 0.0f;
    int err_cnt = 0;
    const float TOLERANCE = 1e-3; // FP32 運算允許微小誤差

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float diff = fabs(h_Q_ref[h][i][d] - h_Q_opt[h][i][d]);
                if (diff > max_err) max_err = diff;
                if (diff > TOLERANCE) {
                    if (err_cnt < 5) // 只印前幾個錯誤
                        printf("Mismatch at Q[%d][%d][%d]: Ref=%f, Opt=%f (diff=%f)\n", h, i, d, h_Q_ref[h][i][d], h_Q_opt[h][i][d], diff);
                    err_cnt++;
                }
            }
        }
    }

    printf("\nVerification Result:\n");
    printf("Max Error: %f\n", max_err);
    if (err_cnt == 0) printf("PASSED!\n");
    else printf("FAILED with %d errors.\n", err_cnt);

    return (err_cnt == 0) ? 0 : 1;
}