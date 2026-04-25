#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cfloat>
#include "layernorm.h"

using namespace std;

// 初始化輸入資料 (Float)
void init_matrix_float(float mat[SEQ_LEN][EMBED_DIM]) {
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            // 產生一些隨機浮點數，模擬 Transformer 的輸出
            mat[i][j] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 2.0f - 1.0f;
        }
    }
}

void init_vector_float(float vec[], int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

// ─── Golden Reference ───
// 模擬 LayerNorm 的計算過程並提取 Golden Scaler
void layernorm_golden(
    float x[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    int8_dt out[SEQ_LEN][EMBED_DIM],
    float scaler_ref[SEQ_LEN]) 
{
    for (int i = 0; i < SEQ_LEN; i++) {
        float mean = 0.0f;
        float var = 0.0f;

        // 1. 計算 Mean
        for (int j = 0; j < EMBED_DIM; j++) {
            mean += x[i][j];
        }
        mean /= EMBED_DIM;

        // 2. 計算 Variance
        for (int j = 0; j < EMBED_DIM; j++) {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= EMBED_DIM;

        // 3. Normalize + Affine (Gamma/Beta) + 找出 Max Abs Val 用於量化
        float inv_std = 1.0f / sqrt(var + 1e-5f); // 假設 EPSILON = 1e-5
        float temp_norm[EMBED_DIM];
        float max_abs = 0.0f;

        for (int j = 0; j < EMBED_DIM; j++) {
            float val = (x[i][j] - mean) * inv_std * gamma[j] + beta[j];
            temp_norm[j] = val;
            if (fabs(val) > max_abs) max_abs = fabs(val);
        }

        // 4. 計算 Scaler (根據你的 HLS: scale = 127/max, scaler = 1/scale)
        // 所以 scaler = max_abs / 127.0
        if (max_abs == 0) scaler_ref[i] = 0.0f;
        else scaler_ref[i] = max_abs / 127.0f;

        // 5. 量化輸出
        float scale = 127.0f / max_abs;
        for (int j = 0; j < EMBED_DIM; j++) {
            out[i][j] = (int8_dt)(temp_norm[j] * scale);
        }
    }
}

int main() {
    srand(42);

    // 分配記憶體 (使用 static 避免 stack overflow)
    static float h_x[SEQ_LEN][EMBED_DIM];
    static float h_gamma[EMBED_DIM];
    static float h_beta[EMBED_DIM];
    static int8_dt h_out_hls[SEQ_LEN][EMBED_DIM];
    static int8_dt h_out_ref[SEQ_LEN][EMBED_DIM];
    static float h_scaler_hls[SEQ_LEN];
    static float h_scaler_ref[SEQ_LEN];

    cout << "--- LayerNorm Scaler Verification ---" << endl;
    cout << "Initializing data..." << endl;
    init_matrix_float(h_x);
    init_vector_float(h_gamma, EMBED_DIM);
    init_vector_float(h_beta, EMBED_DIM);

    // 1. 執行 Golden Reference
    cout << "Calculating Golden Reference..." << endl;
    layernorm_golden(h_x, h_gamma, h_beta, h_out_ref, h_scaler_ref);

    // 2. 執行 HLS Kernel
    cout << "Running HLS Kernel..." << endl;
    layernorm(h_x, h_gamma, h_beta, h_out_hls, h_scaler_hls);

    // 3. 比對 Scaler 結果
    float max_scaler_err = 0.0f;
    int scaler_err_cnt = 0;
    const float TOLERANCE = 1e-4;

    cout << "\nChecking Scaler (1/scale) results:" << endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        float diff = fabs(h_scaler_ref[i] - h_scaler_hls[i]);
        if (diff > max_scaler_err) max_scaler_err = diff;
        
        if (diff > TOLERANCE) {
            if (scaler_err_cnt < 5) {
                printf("  Mismatch at scaler[%d]: Ref=%f, HLS=%f (diff=%f)\n", 
                        i, h_scaler_ref[i], h_scaler_hls[i], diff);
            }
            scaler_err_cnt++;
        }
    }

    // 4. 比對量化後的輸出 (int8)
    int out_err_cnt = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            if (abs(h_out_ref[i][j] - h_out_hls[i][j]) > 1) { // 允許 +/- 1 的捨入誤差
                out_err_cnt++;
            }
        }
    }

    printf("\nVerification Result:\n");
    printf("Max Scaler Error: %e\n", max_scaler_err);
    printf("Scaler Mismatches: %d / %d\n", scaler_err_cnt, SEQ_LEN);
    printf("Output INT8 Mismatches: %d\n", out_err_cnt);

    if (scaler_err_cnt == 0 && out_err_cnt < (SEQ_LEN * EMBED_DIM * 0.01)) {
        printf("PASSED!\n");
    } else {
        printf("FAILED!\n");
    }

    return (scaler_err_cnt == 0) ? 0 : 1;
}