#include "linear_qkv_multihead.h"
#include <cstdlib>
#include <cmath>
#include <cstdio>


int max(int a, int b){
    if (a > b)
        return a;
    else
        return b;
}

void linear_qkv_naive(
    int X[SEQ_LEN][EMBED_DIM],
    int Wq[EMBED_DIM][EMBED_DIM],
    int bq[EMBED_DIM],
    int Wk[EMBED_DIM][EMBED_DIM],
    int bk[EMBED_DIM],
    int Wv[EMBED_DIM][EMBED_DIM],
    int bv[EMBED_DIM],
    int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int h = 0; h < EMBED_DIM; h++) {
            int q = bq[h];
            int k = bk[h];
            int v = bv[h];
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
    static int X[SEQ_LEN][EMBED_DIM];
    static int Wq[EMBED_DIM][EMBED_DIM];
    static int WTq[EMBED_DIM][EMBED_DIM];
    static int Wk[EMBED_DIM][EMBED_DIM];
    static int WTk[EMBED_DIM][EMBED_DIM];
    static int Wv[EMBED_DIM][EMBED_DIM];
    static int WTv[EMBED_DIM][EMBED_DIM];
    static int bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM];

    static int Q_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int K_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int V_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    static int Q_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int K_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int V_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];

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
    for (int i=0;i<EMBED_DIM;i++) {
        for (int j=0;j<EMBED_DIM;j++) {
            WTq[i][j] = Wq[j][i];
        }
    }

    for (int i=0;i<EMBED_DIM;i++) {
        for (int j=0;j<EMBED_DIM;j++) {
            WTk[i][j] = Wk[j][i];
        }
    }

    for (int i=0;i<EMBED_DIM;i++) {
        for (int j=0;j<EMBED_DIM;j++) {
            WTv[i][j] = Wv[j][i];
        }
    }

    linear_qkv_naive(X,Wq,bq,Wk,bk,Wv,bv,Q_ref,K_ref,V_ref);
    linear_qkv_multihead(X,WTq,bq,WTk,bk,WTv,bv,Q_opt,K_opt,V_opt);

int max_err = 0;


    printf("\n=== Detailed Comparison ===\n");
    printf("Type [h, i, d]   |  Ref Value  |  Opt Value  |  Diff (Abs)\n");
    printf("----------------------------------------------------------\n");

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                

                int err_q = fabs(Q_ref[h][i][d] - Q_opt[h][i][d]);
                int err_k = fabs(K_ref[h][i][d] - K_opt[h][i][d]);
                int err_v = fabs(V_ref[h][i][d] - V_opt[h][i][d]);


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
