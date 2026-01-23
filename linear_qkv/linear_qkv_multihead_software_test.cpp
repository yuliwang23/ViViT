#include <cstdlib>
#include <cmath>
#include <cstdio>
#include "linear_qkv_multihead_software.cpp"

float max(float a, float b){
    if (a > b)
        return a;
    else
        return b;
}

void linear_qkv_naive(
    float X[SEQ_LEN][EMBED_DIM],
    float Wq[EMBED_DIM][EMBED_DIM],
    float bq[EMBED_DIM],
    float Wk[EMBED_DIM][EMBED_DIM],
    float bk[EMBED_DIM],
    float Wv[EMBED_DIM][EMBED_DIM],
    float bv[EMBED_DIM],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int h = 0; h < EMBED_DIM; h++) {
            float q = bq[h];
            float k = bk[h];
            float v = bv[h];
            for (int j = 0; j < EMBED_DIM; j++) {
                q += X[i][j] * Wq[j][h];
                k += X[i][j] * Wk[j][h];
                v += X[i][j] * Wv[j][h];
            }
            int hc = h / HEAD_DIM;
            int hr = h % HEAD_DIM;
            Q[hc][i][hr] = q;
            K[hc][i][hr] = k;
            V[hc][i][hr] = v;
        }
    }
}

int main() {
    static float X[SEQ_LEN][EMBED_DIM];
    static float Wq[EMBED_DIM][EMBED_DIM];
    static float Wk[EMBED_DIM][EMBED_DIM];
    static float Wv[EMBED_DIM][EMBED_DIM];
    static float bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM];

    static float Q_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float K_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float V_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    static float Q_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float K_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static float V_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    // init random
    srand(0);
    for (int i=0;i<SEQ_LEN;i++)
        for (int j=0;j<EMBED_DIM;j++)
            X[i][j] = ((float)rand()/RAND_MAX - 0.5f);

    for (int i=0;i<EMBED_DIM;i++) {
        bq[i]=bk[i]=bv[i]=((float)rand()/RAND_MAX - 0.5f);
        for (int j=0;j<EMBED_DIM;j++) {
            Wq[j][i] = ((float)rand()/RAND_MAX - 0.5f);
            Wk[j][i] = ((float)rand()/RAND_MAX - 0.5f);
            Wv[j][i] = ((float)rand()/RAND_MAX - 0.5f);
        }
    }

    linear_qkv_naive(X,Wq,bq,Wk,bk,Wv,bv,Q_ref,K_ref,V_ref);
    linear_qkv_multihead_opt(X,Wq,bq,Wk,bk,Wv,bv,Q_opt,K_opt,V_opt);

float max_err = 0;


    printf("\n=== Detailed Comparison ===\n");
    printf("Type [h, i, d]   |  Ref Value  |  Opt Value  |  Diff (Abs)\n");
    printf("----------------------------------------------------------\n");

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                

                float err_q = fabs(Q_ref[h][i][d] - Q_opt[h][i][d]);
                float err_k = fabs(K_ref[h][i][d] - K_opt[h][i][d]);
                float err_v = fabs(V_ref[h][i][d] - V_opt[h][i][d]);


                max_err = max(max_err, err_q);
                max_err = max(max_err, err_k);
                max_err = max(max_err, err_v);


                if (err_q > 1e-4f) printf("Q    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, Q_ref[h][i][d], Q_opt[h][i][d], err_q);
                if (err_k > 1e-4f) printf("K    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, K_ref[h][i][d], K_opt[h][i][d], err_k);
                if (err_v > 1e-4f) printf("V    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, V_ref[h][i][d], V_opt[h][i][d], err_v);
            }
        }
    }

    printf("----------------------------------------------------------\n");
    printf("Total Max error = %e\n", max_err);
    return 0;
}
