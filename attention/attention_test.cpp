#include "attention.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>

void softmax_software(float in[VEC_MAX],
                      float out[VEC_MAX],
                      int length)
{
    float max_val = -1e38f;
    float sum_exp = 0.0f;

    // Safe Softmax implementation for software
    for (int i = 0; i < length; i++)
    {
        float x = in[i];
        if (x > max_val)
        {
            float diff = max_val - x;
            // Update sum_exp with the new max_val scaling
            sum_exp = sum_exp * std::exp(diff) + 1.0f; 
            max_val = x;
        }
        else
        {
            float diff = x - max_val;
            sum_exp += std::exp(diff);
        }
    }

    for (int i = 0; i < length; i++)
    {
        float diff = in[i] - max_val;
        out[i] = std::exp(diff) / sum_exp;
    }
}

void attention_software(
    float Q[SEQ_LEN][HEAD_DIM],
    float K[SEQ_LEN][HEAD_DIM],
    float V[SEQ_LEN][HEAD_DIM],
    float Out[SEQ_LEN][HEAD_DIM])
{
    static float score_float[SEQ_LEN][SEQ_LEN];
    static float softmax_out[SEQ_LEN];
    static float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            // FIXED: Changed int to float to prevent truncation of floating-point products
            float sum = 0.0f; 
            for (int d = 0; d < HEAD_DIM; d++)
                sum += Q[i][d] * K[j][d];
            score_float[i][j] = sum * scale;
        }
    }

    for (int i = 0; i < SEQ_LEN; i++)
    {
        softmax_software(score_float[i], score_float[i], SEQ_LEN);
    }

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int d = 0; d < HEAD_DIM; d++)
        {
            // Changed to float and 0.0f for consistency
            float sum = 0.0f; 
            for (int j = 0; j < SEQ_LEN; j++)
                sum += score_float[i][j] * V[j][d];
            Out[i][d] = sum;
        }
    }
}

int main() {
    static float Q_test[SEQ_LEN][HEAD_DIM];
    static float K_test[SEQ_LEN][HEAD_DIM];
    static float V_test[SEQ_LEN][HEAD_DIM];

    static float Out_ref[SEQ_LEN][HEAD_DIM];
    static float Out_opt[SEQ_LEN][HEAD_DIM];

    srand(0);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            // Generates random numbers between -10.0 and 10.0
            Q_test[i][d] = (float)((rand() % 21) - 10);
            K_test[i][d] = (float)((rand() % 21) - 10);
            V_test[i][d] = (float)((rand() % 21) - 10);
        }
    }

    // Run Software Reference
    attention_software(Q_test, K_test, V_test, Out_ref);
    
    // Run Hardware Optimized Version (Make sure attention_opt is linked properly)
    attention_opt(Q_test, K_test, V_test, Out_opt);

    // Error Tracking variables changed to float to support precise evaluation
    float max_err = 0.0f;
    int total_elements = SEQ_LEN * HEAD_DIM;
    
    // Buckets for floating point tolerances
    int err_below_1e4 = 0;
    int err_below_1e3 = 0;
    int err_below_1e2 = 0;
    int err_above_1e2 = 0;

    printf("\n=== Detailed Comparison ===\n");
    printf("Index [i, d]     |  Ref Value  |  Opt Value  |  Diff (Abs)\n");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            float err = std::abs(Out_ref[i][d] - Out_opt[i][d]);
            max_err = std::max(max_err, err);
            
            // Un-comment the line below if you want to print every single value
            // printf("Out  [%d, %d]     | %11.6f | %11.6f | %11.6f\n", i, d, Out_ref[i][d], Out_opt[i][d], err);
            
            // Floating point tolerance bucketing
            if (err < 1e-4f) err_below_1e4++;
            else if (err < 1e-3f) err_below_1e3++;
            else if (err < 1e-2f) err_below_1e2++;
            else err_above_1e2++;
        }
    }

    printf("----------------------------------------------------------\n");
    printf("Total Max error = %11.6f\n", max_err);
    printf("Diff < 1e-4: %d (%.2f%%)\n", err_below_1e4, (float)err_below_1e4 / total_elements * 100);
    printf("Diff < 1e-3: %d (%.2f%%)\n", err_below_1e3, (float)err_below_1e3 / total_elements * 100);
    printf("Diff < 1e-2: %d (%.2f%%)\n", err_below_1e2, (float)err_below_1e2 / total_elements * 100);
    printf("Diff > 1e-2: %d (%.2f%%)\n", err_above_1e2, (float)err_above_1e2 / total_elements * 100);

    // For HLS vs CPU float comparison, a tolerance of 1e-2 or 1e-3 is generally acceptable 
    // due to different accumulation orders (adder trees) and math library implementations.
    if (max_err > 1e-2f) {
        printf("Test FAILED! (Max error exceeds 0.01)\n");
        return 1; // Return non-zero to indicate failure to the OS/Scripts
    } else {
        printf("Test PASSED!\n");
        return 0;
    }
}