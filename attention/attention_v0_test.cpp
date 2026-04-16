//#include "attention_v0.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <algorithm>

constexpr int SEQ_LEN = 14 * 14 + 1;
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float scale = 0.125;
constexpr int VEC_MAX = SEQ_LEN;
constexpr int tile_size = 16; 

void softmax_software(float in[VEC_MAX],
                      float out[VEC_MAX],
                      int length)
{
    float max_val = -1e38f;
    float sum_exp = 0.0f;

    for (int i = 0; i < length; i++)
    {
        float x = in[i];
        if (x > max_val)
        {
            float diff = max_val - x;
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
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int Out[SEQ_LEN][HEAD_DIM])
{
	static float score_float[SEQ_LEN][SEQ_LEN];
	static float softmax_out[SEQ_LEN];
	static float scale = 1.0f / std::sqrt((float)HEAD_DIM);

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            int sum = 0;
            for (int d = 0; d < HEAD_DIM; d++)
                sum += Q[i][d] * K[j][d];
            score_float[i][j] = (float)sum * scale;
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
            float sum = 0;
            for (int j = 0; j < SEQ_LEN; j++)
                sum += score_float[i][j] * (float)V[j][d];
            Out[i][d] = (int)std::round(sum);
        }
    }
}

// Helper function to read matrix data from a text file
bool load_matrix_from_file(const char* filename, int mat[SEQ_LEN][HEAD_DIM]) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error: Could not open file %s\n", filename);
        return false;
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            if (fscanf(fp, "%d", &mat[i][d]) != 1) {
                printf("Error: Data insufficient or format error in %s at [%d][%d]\n", filename, i, d);
                fclose(fp);
                return false;
            }
        }
    }
    fclose(fp);
    return true;
}

int main() {
    // Static allocation to prevent Stack Overflow
    static int Q_input[SEQ_LEN][HEAD_DIM];
    static int K_input[SEQ_LEN][HEAD_DIM];
    static int V_input[SEQ_LEN][HEAD_DIM];
    static int Out_software[SEQ_LEN][HEAD_DIM]; // Reference calculated by software
    static int Out_hardware[SEQ_LEN][HEAD_DIM]; // Output loaded from your hardware file

    // 1. Load Input Files
    printf("--- Step 1: Loading Input Files ---\n");
    if (!load_matrix_from_file("input_q.txt", Q_input)) return -1;
    if (!load_matrix_from_file("input_k.txt", K_input)) return -1;
    if (!load_matrix_from_file("input_v.txt", V_input)) return -1;
    printf("Successfully loaded input_q.txt, input_k.txt, and input_v.txt\n");

    // 2. Run Software Reference
    printf("\n--- Step 2: Running Software Simulation (attention_software) ---\n");
    attention_software(Q_input, K_input, V_input, Out_software);
    printf("Software calculation completed.\n");

    // 3. Load External Hardware Output
    printf("\n--- Step 3: Loading Hardware Output File (output_opt.txt) ---\n");
    if (!load_matrix_from_file("Out_output.txt", Out_hardware)) {
        printf("Please ensure 'Out_output_opt.txt' exists in the directory.\n");
        return -1;
    }
    printf("Successfully loaded Out_output_opt.txt\n");

    // 4. Compare Results
    int max_err = 0;
    int total_elements = SEQ_LEN * HEAD_DIM;
    int zero_err = 0, err_1 = 0, err_2 = 0, err_3 = 0, err_gt3 = 0;

    printf("\n=== Detailed Comparison (First 10 mismatches) ===\n");
    printf("%-12s | %-12s | %-12s | %-6s\n", "Index [i,d]", "Software", "Hardware", "Diff");
    printf("----------------------------------------------------------\n");

    int diff_count = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            int sw_val = Out_software[i][d];
            int hw_val = Out_hardware[i][d];
            int diff = std::abs(sw_val - hw_val);

            if (diff > max_err) max_err = diff;

            if (diff == 0) zero_err++;
            else if (diff == 1) err_1++;
            else if (diff == 2) err_2++;
            else if (diff == 3) err_3++;
            else err_gt3++;

            // Print first 10 differences for debugging
            if (diff > 0 && diff_count < 10) {
                printf("Out [%3d,%2d] | %12d | %12d | %6d\n", i, d, sw_val, hw_val, diff);
                diff_count++;
            }
        }
    }

    // 5. Summary Report
    printf("----------------------------------------------------------\n");
    printf("Total Elements Checked : %d\n", total_elements);
    printf("Max Absolute Error     : %d\n", max_err);
    printf("\nError Distribution Statistics:\n");
    printf("  - Error = 0 (Exact)  : %d (%.2f%%)\n", zero_err, (float)zero_err/total_elements*100);
    printf("  - Error = 1          : %d (%.2f%%)\n", err_1, (float)err_1/total_elements*100);
    printf("  - Error = 2          : %d (%.2f%%)\n", err_2, (float)err_2/total_elements*100);
    printf("  - Error = 3          : %d (%.2f%%)\n", err_3, (float)err_3/total_elements*100);
    if (err_gt3 > 0) {
        printf("  - Error > 3          : %d (%.2f%%)\n", err_gt3, (float)err_gt3/total_elements*100);
    }

    // Pass/Fail Criteria (Threshold = 5)
    if (max_err <= 5) {
        printf("\n[RESULT] Test PASSED!\n");
    } else {
        printf("\n[RESULT] Test FAILED - Error exceeds tolerance.\n");
    }

    return 0;
}