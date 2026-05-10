#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

// 假設的硬體模組標頭檔
#include "linear_qkv_multihead.h" 

using namespace std;

// ============================================================================
// Model Configuration Macros
// ============================================================================
#define SEQ_LEN     128
#define EMBED_DIM   768
#define NUM_HEADS   12
#define HEAD_DIM    (EMBED_DIM / NUM_HEADS) // 64

// ============================================================================
// Initialization & Quantization Helpers
// ============================================================================

// Initialize float matrix with random values
void init_matrix_float(float mat[EMBED_DIM][EMBED_DIM], float min_val, float max_val) {
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            mat[i][j] = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
        }
    }
}

// Initialize Float vector (for Bias and Input Scales)
void init_vector_float(float vec[], int size, float min_val, float max_val) {
    for (int i = 0; i < size; i++) {
        vec[i] = min_val + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max_val - min_val)));
    }
}

// Per-channel Quantization for Weights
// Calculates the scaler = 127.5 / max_abs, quantizes the weight, 
// and outputs the hardware scaler.
void quantize_weight_per_channel(
    float W_fp32[EMBED_DIM][EMBED_DIM],
    int8_t W_int8[EMBED_DIM][EMBED_DIM],
    float hw_scale[EMBED_DIM]
) {
    for (int j = 0; j < EMBED_DIM; j++) {
        float max_abs = 1e-5f; // Epsilon to prevent division by zero

        // 1. Find the maximum absolute value in the current column (channel)
        for (int i = 0; i < EMBED_DIM; i++) {
            max_abs = std::max(max_abs, std::abs(W_fp32[i][j]));
        }

        // 2. Calculate the quantizer scaler based on the requested formula
        // Formula: Reciprocal of (abs(max) / 127.5) -> 127.5 / abs(max)
        float quant_scaler = 127.5f / max_abs;

        // 3. Quantize the weights to INT8
        for (int i = 0; i < EMBED_DIM; i++) {
            float scaled_val = W_fp32[i][j] * quant_scaler;
            // Round to nearest integer and clip to INT8 range (-128 to 127)
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            W_int8[i][j] = static_cast<int8_t>(scaled_val);
        }

        // 4. Output the scale for hardware dequantization.
        // NOTE: If hardware dequantizes using multiplication (MAC_output * scale), 
        // this needs to be the inverse of the quant_scaler (i.e., max_abs / 127.5).
        hw_scale[j] = 1.0f / quant_scaler; 
    }
}

// ============================================================================
// Golden Reference 
// ============================================================================
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
            
            // 1. Perform INT8 Matrix Multiplication (Accumulate in INT32)
            for (int k = 0; k < EMBED_DIM; k++) {
                acc_q += (int32_t)X[i][k] * (int32_t)Wq[k][j];
                acc_k += (int32_t)X[i][k] * (int32_t)Wk[k][j];
                acc_v += (int32_t)X[i][k] * (int32_t)Wv[k][j];
            }

            // 2. Dequantize and add Bias (Dequant: ACC * scale_in * scale_w + bias)
            float res_q = (float)acc_q * si[i] * sq[j] + bq[j];
            float res_k = (float)acc_k * si[i] * sk[j] + bk[j];
            float res_v = (float)acc_v * si[i] * sv[j] + bv[j];

            // 3. Map to Multi-head layout format
            int head_idx = j / HEAD_DIM;
            int head_dim = j % HEAD_DIM;
            Q[head_idx][i][head_dim] = res_q;
            K[head_idx][i][head_dim] = res_k;
            V[head_idx][i][head_dim] = res_v;
        }
    }
}

// 模擬硬體介面的 Dummy Function (實際測試時請替換為你的硬體模組呼叫)
void linear_qkv_multihead_dummy(
    int8_t X[SEQ_LEN][EMBED_DIM],
    int8_t Wq_T[EMBED_DIM][EMBED_DIM], float bq[EMBED_DIM], float sq[EMBED_DIM],
    int8_t Wk_T[EMBED_DIM][EMBED_DIM], float bk[EMBED_DIM], float sk[EMBED_DIM],
    int8_t Wv_T[EMBED_DIM][EMBED_DIM], float bv[EMBED_DIM], float sv[EMBED_DIM],
    float si[SEQ_LEN],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM]) 
{
    // C-Sim 環境下，如果尚未打包成 uint256_dt，可在此處反轉置並呼叫 Golden 測試
    // 在實際環境中，這將會是呼叫 linear_qkv_multihead 的地方
    int8_t Wq_orig[EMBED_DIM][EMBED_DIM];
    int8_t Wk_orig[EMBED_DIM][EMBED_DIM];
    int8_t Wv_orig[EMBED_DIM][EMBED_DIM];

    for(int i=0; i<EMBED_DIM; i++){
        for(int j=0; j<EMBED_DIM; j++){
            Wq_orig[i][j] = Wq_T[j][i];
            Wk_orig[i][j] = Wk_T[j][i];
            Wv_orig[i][j] = Wv_T[j][i];
        }
    }
    linear_qkv_golden(X, Wq_orig, bq, sq, Wk_orig, bk, sk, Wv_orig, bv, sv, si, Q, K, V);
}

// ============================================================================
// Main Simulation Testbench
// ============================================================================
int main() {
    srand(42); // Fixed seed for reproducible debugging

    cout << "Allocating memory..." << endl;
    
    // Using 'static' to place large arrays in the data segment instead of the stack 
    // to prevent Stack Overflow during C-Simulation.
    static float h_Wq_fp32[EMBED_DIM][EMBED_DIM], h_Wk_fp32[EMBED_DIM][EMBED_DIM], h_Wv_fp32[EMBED_DIM][EMBED_DIM];
    
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

    cout << "Initializing random data..." << endl;
    
    // Initialize X with pseudo-random INT8 directly (Assuming LayerNorm outputs this)
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_X[i][j] = (int8_t)(rand() % 255 - 128); 
        }
    }
    
    // Initialize raw FP32 weights (Typically loaded from PyTorch models)
    init_matrix_float(h_Wq_fp32, -0.5f, 0.5f);
    init_matrix_float(h_Wk_fp32, -0.5f, 0.5f);
    init_matrix_float(h_Wv_fp32, -0.5f, 0.5f);
    
    cout << "Quantizing weights and calculating scalers..." << endl;
    quantize_weight_per_channel(h_Wq_fp32, h_Wq, h_sq);
    quantize_weight_per_channel(h_Wk_fp32, h_Wk, h_sk);
    quantize_weight_per_channel(h_Wv_fp32, h_Wv, h_sv);

    // Transpose quantized weights for hardware friendly memory access
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
    
    init_vector_float(h_si, SEQ_LEN, 0.001f, 0.05f); // Input Activation Scales

    cout << "Calculating Golden Reference..." << endl;
    linear_qkv_golden(h_X, h_Wq, h_bq, h_sq, h_Wk, h_bk, h_sk, h_Wv, h_bv, h_sv, h_si, h_Q_ref, h_K_ref, h_V_ref);

    cout << "Running HLS Kernel Simulation..." << endl;
    // NOTE: Replace this dummy call with your actual hardware function signature
    // Make sure to pack the INT8 arrays into uint256_dt if your top-level requires it.
    linear_qkv_multihead_dummy(
        h_X, 
        h_Wq_T, h_bq, h_sq, 
        h_Wk_T, h_bk, h_sk, 
        h_Wv_T, h_bv, h_sv,
        h_si,
        h_Q_opt, h_K_opt, h_V_opt
    );

    cout << "Comparing results..." << endl;
    float max_err = 0.0f;
    int err_cnt = 0;
    const float TOLERANCE = 1e-3; // Acceptable FP32 calculation noise

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float diff = fabs(h_Q_ref[h][i][d] - h_Q_opt[h][i][d]);
                if (diff > max_err) max_err = diff;
                if (diff > TOLERANCE) {
                    if (err_cnt < 5) 
                        printf("Mismatch at Q[%d][%d][%d]: Ref=%f, Opt=%f (diff=%f)\n", h, i, d, h_Q_ref[h][i][d], h_Q_opt[h][i][d], diff);
                    err_cnt++;
                }
            }
        }
    }

    printf("\nVerification Result:\n");
    printf("Max Error: %f\n", max_err);
    if (err_cnt == 0) printf(">> C-Simulation PASSED!\n");
    else printf(">> C-Simulation FAILED with %d errors.\n", err_cnt);

    return (err_cnt == 0) ? 0 : 1;
}