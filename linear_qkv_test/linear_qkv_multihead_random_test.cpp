#include "linear_qkv_multihead.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <ctime>

using namespace std;

// Random 初始化
void init_matrix(int mat[][EMBED_DIM], int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = rand() % 101;
        }
    }
}

void init_vector(int vec[], int size) {
    for (int i = 0; i < size; i++) {
        vec[i] = rand() % 101;
    }
}

// Transpose
void transpose_matrix(
    int in[EMBED_DIM][EMBED_DIM],
    int out[EMBED_DIM][EMBED_DIM])
{
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            out[j][i] = in[i][j];
        }
    }
}

void write_output(const char* filename, int data[NUM_HEADS][SEQ_LEN][HEAD_DIM]) {
    std::ofstream fout(filename);

    if (!fout) {
        std::cout << "Cannot open file: " << filename << std::endl;
        return;
    }

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                fout << data[h][i][d] << " ";
            }
            fout << std::endl;
        }
        fout << std::endl;
    }

    fout.close();
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

    srand(time(NULL));  // 每次不同

    static int X[SEQ_LEN][EMBED_DIM];

    static int Wq[EMBED_DIM][EMBED_DIM];
    static int Wk[EMBED_DIM][EMBED_DIM];
    static int Wv[EMBED_DIM][EMBED_DIM];

    static int WTq[EMBED_DIM][EMBED_DIM];
    static int WTk[EMBED_DIM][EMBED_DIM];
    static int WTv[EMBED_DIM][EMBED_DIM];

    static int bq[EMBED_DIM], bk[EMBED_DIM], bv[EMBED_DIM];

    static int Q_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int K_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int V_ref[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    static int Q_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int K_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    static int V_opt[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    // Generate Random Data
    init_matrix(X, SEQ_LEN, EMBED_DIM);

    init_matrix(Wq, EMBED_DIM, EMBED_DIM);
    init_matrix(Wk, EMBED_DIM, EMBED_DIM);
    init_matrix(Wv, EMBED_DIM, EMBED_DIM);

    init_vector(bq, EMBED_DIM);
    init_vector(bk, EMBED_DIM);
    init_vector(bv, EMBED_DIM);

    // Transpose
    transpose_matrix(Wq, WTq);
    transpose_matrix(Wk, WTk);
    transpose_matrix(Wv, WTv);

    linear_qkv_naive(X, Wq, bq, Wk, bk, Wv, bv, Q_ref, K_ref, V_ref);
    linear_qkv_multihead(X, WTq, bq, WTk, bk, WTv, bv, Q_opt, K_opt, V_opt);

    // Compare
    int max_err = 0;

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {

                int err_q = abs(Q_ref[h][i][d] - Q_opt[h][i][d]);
                int err_k = abs(K_ref[h][i][d] - K_opt[h][i][d]);
                int err_v = abs(V_ref[h][i][d] - V_opt[h][i][d]);

                max_err = max(max_err, err_q);
                max_err = max(max_err, err_k);
                max_err = max(max_err, err_v);

                if (err_q != 0)
                    printf("Q [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,Q_ref[h][i][d],Q_opt[h][i][d],err_q);

                if (err_k != 0)
                    printf("K [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,K_ref[h][i][d],K_opt[h][i][d],err_k);

                if (err_v != 0)
                    printf("V [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,V_ref[h][i][d],V_opt[h][i][d],err_v);
            }
        }
    }

    printf("\nMax error = %d\n", max_err);

    if (max_err == 0)
        printf("PASS\n");
    else
        printf("FAIL\n");

    write_output("Q_ref.txt", Q_ref);
    write_output("K_ref.txt", K_ref);
    write_output("V_ref.txt", V_ref);
    
    write_output("Q_opt.txt", Q_opt);
    write_output("K_opt.txt", K_opt);
    write_output("V_opt.txt", V_opt);

    return 0;
}