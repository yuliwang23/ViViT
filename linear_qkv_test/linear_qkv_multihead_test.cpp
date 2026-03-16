#include "linear_qkv_multihead.h"
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>

using namespace std;

bool read_matrix(const char* filename, int rows, int cols, int mat[][EMBED_DIM]) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cout << "Cannot open file: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fin >> mat[i][j];
        }
    }

    fin.close();
    return true;
}

bool read_vector(const char* filename, int size, int vec[]) {
    std::ifstream fin(filename);
    if (!fin) {
        std::cout << "Cannot open file: " << filename << std::endl;
        return false;
    }

    for (int i = 0; i < size; i++) {
        fin >> vec[i];
    }

    fin.close();
    return true;
}

int max_int(int a, int b){
    return (a > b) ? a : b;
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

// 完整比對函數
bool compare_matrices(int W[EMBED_DIM][EMBED_DIM], int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM], const char* name) {
    bool identical = true;
    for(int i=0;i<SEQ_LEN;i++){
        for(int h=0;h<EMBED_DIM;h++){
            int hc = h / HEAD_DIM;
            int hr = h % HEAD_DIM;
            if(Q[hc][i][hr] != W[i%EMBED_DIM][h]){
                cout << name << " mismatch at [" << i << "," << h << "] "
                     << "Q=" << Q[hc][i][hr] << " W=" << W[i%EMBED_DIM][h] << endl;
                identical = false;
            }
        }
    }
    return identical;
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

    // ==============================
    // Read input
    // ==============================

    if (!read_matrix("X.txt", SEQ_LEN, EMBED_DIM, X)) return -1;

    if (!read_matrix("Wq.txt", EMBED_DIM, EMBED_DIM, Wq)) return -1;
    if (!read_matrix("Wq_transpose.txt", EMBED_DIM, EMBED_DIM, WTq)) return -1;

    if (!read_matrix("Wk.txt", EMBED_DIM, EMBED_DIM, Wk)) return -1;
    if (!read_matrix("Wk_transpose.txt", EMBED_DIM, EMBED_DIM, WTk)) return -1;

    if (!read_matrix("Wv.txt", EMBED_DIM, EMBED_DIM, Wv)) return -1;
    if (!read_matrix("Wv_transpose.txt", EMBED_DIM, EMBED_DIM, WTv)) return -1;

    if (!read_vector("bq.txt", EMBED_DIM, bq)) return -1;
    if (!read_vector("bk.txt", EMBED_DIM, bk)) return -1;
    if (!read_vector("bv.txt", EMBED_DIM, bv)) return -1;

    // ==============================
    // Run
    // ==============================

    linear_qkv_naive(X,Wq,bq,Wk,bk,Wv,bv,Q_ref,K_ref,V_ref);
    linear_qkv_multihead(X,Wq,bq,Wk,bk,Wv,bv,Q_opt,K_opt,V_opt);

    int max_err = 0;

    printf("\n=== Detailed Comparison ===\n");
    printf("Type [h,i,d] | Ref | Opt | Diff\n");
    printf("----------------------------------\n");

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {

                int err_q = abs(Q_ref[h][i][d] - Q_opt[h][i][d]);
                int err_k = abs(K_ref[h][i][d] - K_opt[h][i][d]);
                int err_v = abs(V_ref[h][i][d] - V_opt[h][i][d]);

                max_err = max(max_err, err_q);
                max_err = max(max_err, err_k);
                max_err = max(max_err, err_v);


                /*if (err_q > 1e-4f) printf("Q    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, Q_ref[h][i][d], Q_opt[h][i][d], err_q);
                if (err_k > 1e-4f) printf("K    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, K_ref[h][i][d], K_opt[h][i][d], err_k);
                if (err_v > 1e-4f) printf("V    [%d, %d, %d]   | %11.5f | %11.5f | %11.5f\n", h, i, d, V_ref[h][i][d], V_opt[h][i][d], err_v);*/

                if (err_q != 0)
                    printf("Q [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,Q_ref[h][i][d],Q_opt[h][i][d],err_q);

                if (err_k != 0)
                    printf("K [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,K_ref[h][i][d],K_opt[h][i][d],err_k);

                if (err_v != 0)
                    printf("V [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,V_ref[h][i][d],V_opt[h][i][d],err_v);
                
                /*if (err_q > 0)
                    printf("Q [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,Q_ref[h][i][d],Q_opt[h][i][d],err_q);

                if (err_k > 0)
                    printf("K [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,K_ref[h][i][d],K_opt[h][i][d],err_k);

                if (err_v > 0)
                    printf("V [%d,%d,%d] | %d | %d | %d\n",
                        h,i,d,V_ref[h][i][d],V_opt[h][i][d],err_v);*/
            }
        }
    }

    /*printf("----------------------------------\n");
    printf("Total Max error = %d\n", max_err);

    cout << "==== Debug Input ====" << endl;

    cout << "X[0][0] = " << X[0][0] << endl;
    cout << "X[0][1] = " << X[0][1] << endl;

    cout << "Wq[0][0] = " << Wq[0][0] << endl;
    cout << "Wq[0][1] = " << Wq[0][1] << endl;

    cout << "bq[0] = " << bq[0] << endl;*/

    // Debug check
    /*std::cout << "X[0][0] = " << X[0][0] << std::endl;
    std::cout << "Wq[0][0] = " << Wq[0][0] << std::endl;
    std::cout << "Q_ref[0][0][0] = " << Q_ref[0][0][0] << std::endl;
    std::cout << "Q_opt[0][0][0] = " << Q_opt[0][0][0] << std::endl;*/

    write_output("Q_opt.txt", Q_opt);
    write_output("K_opt.txt", K_opt);
    write_output("V_opt.txt", V_opt);

    return 0;
}