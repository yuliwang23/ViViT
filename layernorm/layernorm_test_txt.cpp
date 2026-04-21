#include "layernorm.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
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

// Load x[SEQ_LEN][EMBED_DIM] from file
bool load_x(const char* filename, float X[SEQ_LEN][EMBED_DIM]) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        printf("ERROR: Could not open %s\n", filename);
        return false;
    }
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            if (!(infile >> X[i][j])) {
                printf("ERROR: Data mismatch in %s at [%d][%d]\n", filename, i, j);
                return false;
            }
    return true;
}

// Load 1D array[EMBED_DIM] from file
bool load_1d(const char* filename, float arr[EMBED_DIM]) {
    ifstream infile(filename);
    if (!infile.is_open()) {
        printf("ERROR: Could not open %s\n", filename);
        return false;
    }
    for (int j = 0; j < EMBED_DIM; j++)
        if (!(infile >> arr[j])) {
            printf("ERROR: Data mismatch in %s at [%d]\n", filename, j);
            return false;
        }
    return true;
}

int main() {
    
    // Static allocation to prevent stack overflow on large dimensions
    static float X[SEQ_LEN][EMBED_DIM];
    static float gamma[EMBED_DIM];
    static float beta[EMBED_DIM];

    static int8_t out_ref[SEQ_LEN][EMBED_DIM];
    static int8_dt out_opt[SEQ_LEN][EMBED_DIM]; // HLS custom data type

    // 1. Load input data from .txt files
    printf("Loading test data from files...\n");

    if (!load_x("input_x.txt", X))         return 1;
    printf("  input_x.txt     loaded (%dx%d floats)\n", SEQ_LEN, EMBED_DIM);

    if (!load_1d("input_gamma.txt", gamma)) return 1;
    printf("  input_gamma.txt loaded (%d floats)\n", EMBED_DIM);

    if (!load_1d("input_beta.txt",  beta))  return 1;
    printf("  input_beta.txt  loaded (%d floats)\n", EMBED_DIM);

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