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

int max_int(int a, int b) {
    return (a > b) ? a : b;
}

int main() {
    static int Q_test[SEQ_LEN][HEAD_DIM];
    static int K_test[SEQ_LEN][HEAD_DIM];
    static int V_test[SEQ_LEN][HEAD_DIM];

    static int Out_ref[SEQ_LEN][HEAD_DIM];
    static int Out_opt[SEQ_LEN][HEAD_DIM];

    srand(0);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            Q_test[i][d] = (rand() % 21) - 10;
            K_test[i][d] = (rand() % 21) - 10;
            V_test[i][d] = (rand() % 21) - 10;
        }
    }

    attention_software(Q_test, K_test, V_test, Out_ref);
    // attention_opt must be defined in another file or linked
    attention_opt(Q_test, K_test, V_test, Out_opt);

    int max_err = 0;
    int total_elements = SEQ_LEN * HEAD_DIM;
    int zero_count = 0;
    int one = 0;
    int two = 0;
    int three = 0;

    printf("\n=== Detailed Comparison ===\n");
    printf("Index [i, d]     |  Ref Value  |  Opt Value  |  Diff (Abs)\n");
    printf("----------------------------------------------------------\n");

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < HEAD_DIM; d++) {
            int err = std::abs(Out_ref[i][d] - Out_opt[i][d]);
            max_err = max_int(max_err, err);
            printf("Out  [%d, %d]     | %11d | %11d | %11d\n", 
                    i, d, Out_ref[i][d], Out_opt[i][d], err);
            if (err == 0) zero_count++;
            else if (err == 1) one++;
            else if (err == 2) two++;
            else if (err == 3) three++;
            
        }
    }

    printf("----------------------------------------------------------\n");
    printf("Total Max error = %d\n", max_err);
    printf("Zero error count: %d (%.2f%%)\n", zero_count, (float)zero_count / total_elements * 100);
    printf("One error count: %d (%.2f%%)\n", one, (float)one / total_elements * 100);
    printf("Two error count: %d (%.2f%%)\n", two, (float)two / total_elements * 100);
    printf("Three error count: %d (%.2f%%)\n", three, (float)three / total_elements * 100);

    if (max_err > 5) {
        printf("Test FAILED!\n");
        return 0;
    } else {
        printf("Test PASSED!\n");
        return 0;
    }
}
