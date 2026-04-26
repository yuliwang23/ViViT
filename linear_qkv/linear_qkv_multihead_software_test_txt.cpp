#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include "linear_qkv_multihead_software.cpp"

// MAX macro
float max(float a, float b){
    return (a > b) ? a : b;
}

// -------------------------------------------------
// Function to read a 2D matrix from txt
// -------------------------------------------------
bool read_matrix(const char* filename, int rows, int cols, float mat[][EMBED_DIM]) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            fin >> mat[i][j];
    fin.close();
    return true;
}

// -------------------------------------------------
// Function to read a vector from txt
// -------------------------------------------------
bool read_vector(const char* filename, int size, float vec[]) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }
    for (int i = 0; i < size; i++)
        fin >> vec[i];
    fin.close();
    return true;
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

// -------------------------------------------------
// Function to write a 3D array to txt
// -------------------------------------------------
bool write_3d_matrix(const char* filename, int dim0, int dim1, int dim2, float mat[][SEQ_LEN][HEAD_DIM]) {
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error opening file for write: " << filename << std::endl;
        return false;
    }
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            for (int k = 0; k < dim2; k++) {
                fout << mat[i][j][k];
                if (k != dim2 - 1) fout << " ";
            }
            fout << "\n";
        }
    }
    fout.close();
    return true;
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

    // ----------------------------
    // Read all inputs from txt
    // ----------------------------
    if (!read_matrix("X.txt", SEQ_LEN, EMBED_DIM, X)) return EXIT_FAILURE;
    if (!read_matrix("Wq_transpose.txt", EMBED_DIM, EMBED_DIM, Wq)) return EXIT_FAILURE;
    if (!read_matrix("Wk_transpose.txt", EMBED_DIM, EMBED_DIM, Wk)) return EXIT_FAILURE;
    if (!read_matrix("Wv_transpose.txt", EMBED_DIM, EMBED_DIM, Wv)) return EXIT_FAILURE;

    if (!read_vector("bq.txt", EMBED_DIM, bq)) return EXIT_FAILURE;
    if (!read_vector("bk.txt", EMBED_DIM, bk)) return EXIT_FAILURE;
    if (!read_vector("bv.txt", EMBED_DIM, bv)) return EXIT_FAILURE;

    // ----------------------------
    // Run reference and optimized QKV
    // ----------------------------
    linear_qkv_naive(X,Wq,bq,Wk,bk,Wv,bv,Q_ref,K_ref,V_ref);
    //linear_qkv_multihead_opt(X,Wq,bq,Wk,bk,Wv,bv,Q_opt,K_opt,V_opt);

    // ----------------------------
    // Compare results
    // ----------------------------
    /*float max_err = 0;

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
    printf("Total Max error = %e\n", max_err);*/
    // ----------------------------
    // Write outputs to txt
    // ----------------------------
    write_3d_matrix("Q_output_sw.txt", NUM_HEADS, SEQ_LEN, HEAD_DIM, Q_opt);
    write_3d_matrix("K_output_sw.txt", NUM_HEADS, SEQ_LEN, HEAD_DIM, K_opt);
    write_3d_matrix("V_output_sw.txt", NUM_HEADS, SEQ_LEN, HEAD_DIM, V_opt);

    printf("Output written to Q_output.txt, K_output.txt, V_output.txt\n");
    return 0;
}