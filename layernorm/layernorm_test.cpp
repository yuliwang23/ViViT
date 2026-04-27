#include "layernorm.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// Golden model for software reference
void layernorm_golden(float x[SEQ_LEN][EMBED_DIM],
                      float gamma[EMBED_DIM],
                      float beta[EMBED_DIM],
                      int8_t out[SEQ_LEN][EMBED_DIM],
                      int M, int N,
                      float eps)
{
    for (int i = 0; i < M; i++)
    {
        float mean = 0.0f;
        float var = 0.0f;
        
        for (int j = 0; j < N; j++)
            mean += x[i][j];
        mean /= N;
        
        for (int j = 0; j < N; j++)
        {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= N;
        
        float inv_std = 1.0f / std::sqrt(var + eps);
        float max_abs_val = 0.0f;
        std::vector<float> temp_norm(N);
        
        for (int j = 0; j < N; j++)
        {
            temp_norm[j] = gamma[j] * ((x[i][j] - mean) * inv_std) + beta[j];
            max_abs_val = std::max(max_abs_val, std::abs(temp_norm[j]));
        }

        float scale = (max_abs_val == 0.0f) ? 0.0f : (127.0f / max_abs_val);

        for (int j = 0; j < N; j++)
        {
            float scaled_val = temp_norm[j] * scale;
            out[i][j] = static_cast<int8_t>(std::round(scaled_val));
        }
    }
}

// Helper function to generate random float within a specific range
float rand_float(float min_val, float max_val) {
    float scale = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return min_val + scale * (max_val - min_val);
}

// Initialization function with reasonable Neural Network ranges
void init_test_data(float X[SEQ_LEN][EMBED_DIM], 
                    float gamma[EMBED_DIM], 
                    float beta[EMBED_DIM]) 
{
    // Fixed seed for 100% reproducible testing across C Sim and RTL Co-Sim
    srand(42); 

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            // Simulated raw feature map distribution [-15.0, 15.0]
            X[i][j] = rand_float(-15.0f, 15.0f);
        }
    }

    for (int j = 0; j < EMBED_DIM; j++) {
        // gamma usually centers around 1.0 after training [0.5, 2.0]
        gamma[j] = rand_float(0.5f, 2.0f);
        // beta usually centers around 0.0 after training [-1.0, 1.0]
        beta[j] = rand_float(-1.0f, 1.0f);
    }
}

int main() {
    
    // Static allocation to prevent stack overflow on large dimensions
    static float X[SEQ_LEN][EMBED_DIM];
    static float gamma[EMBED_DIM];
    static float beta[EMBED_DIM];

    static int8_t out_ref[SEQ_LEN][EMBED_DIM];
    static int8_dt out_opt[SEQ_LEN][EMBED_DIM]; // HLS custom data type

    // 1. Generate pseudo-random data
    printf("Generating test data with fixed seed...\n");
    init_test_data(X, gamma, beta);

    // 2. Run Software Golden Model
    printf("Running Software Golden Model...\n");
    layernorm_golden(X, gamma, beta, out_ref, SEQ_LEN, EMBED_DIM, EPSILON);

    // 3. Run Hardware Optimized Model (HLS DUT)
    printf("Running Hardware HLS Model...\n");
    layernorm(X, gamma, beta, out_opt);

    // 4. Compare Results
    int max_err = 0;
    int error_count = 0;
    int tolerance = 1; // Allow +/- 1 difference due to float rounding and accumulator tree differences

    printf("\n=== Detailed Comparison ===\n");
    printf("Pos [i, j]   | Ref | Opt | Diff\n");
    printf("----------------------------------\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            
            // Cast to int before calculating difference
            int val_ref = static_cast<int>(out_ref[i][j]);
            int val_opt = static_cast<int>(out_opt[i][j]);
            int err = std::abs(val_ref - val_opt);

            max_err = std::max(max_err, err);

            // Only print if the error exceeds acceptable tolerance
            if (err > tolerance) {
                printf("OUT [%3d, %3d] | %4d | %4d | %3d\n", i, j, val_ref, val_opt, err);
                error_count++;
            }
        }
    }

    printf("----------------------------------\n");
    printf("Total Mismatches (Diff > %d): %d\n", tolerance, error_count);
    printf("Maximum Absolute Error: %d\n", max_err);

    if (error_count == 0) {
        printf("\n>>> TEST PASSED! <<<\n\n");
    } else {
        printf("\n>>> TEST FAILED! <<<\n\n");
    }

    return (error_count == 0) ? 0 : 1;
}