#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================================================================
// Transformer Dimensions
// ============================================================================
const int SEQ_LEN = 197;
const int EMBED_DIM = 768;
const int NUM_HEADS = 12;
const int HEAD_DIM = 64; // EMBED_DIM / NUM_HEADS
const float EPSILON = 1e-5f;

// ============================================================================
// Core Functions (From User)
// ============================================================================
void layernorm(const float x[SEQ_LEN][EMBED_DIM],
               const float gamma[EMBED_DIM],
               const float beta[EMBED_DIM],
               float out[SEQ_LEN][EMBED_DIM],
               int M, int N, float eps)
{
    for (int i = 0; i < M; i++) {
        float mean = 0.0f;
        float var = 0.0f;

        // Mean
        for (int j = 0; j < N; j++) mean += x[i][j];
        mean /= N;

        // Variance
        for (int j = 0; j < N; j++) {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= N;
        float inv_std = 1.0f / sqrtf(var + eps);

        // Normalize
        for (int j = 0; j < N; j++) {
            out[i][j] = gamma[j] * ((x[i][j] - mean) * inv_std) + beta[j];
        }
    }
}

void golden_qkv(
    const int8_t  X[SEQ_LEN][EMBED_DIM],
    const int8_t  Wq[EMBED_DIM][EMBED_DIM],
    const int8_t  Wk[EMBED_DIM][EMBED_DIM],
    const int8_t  Wv[EMBED_DIM][EMBED_DIM],
    const float   bq[EMBED_DIM],
    const float   bk[EMBED_DIM],
    const float   bv[EMBED_DIM],
    const float   scale_input[SEQ_LEN],
    const float   scale_wq[EMBED_DIM],
    const float   scale_wk[EMBED_DIM],
    const float   scale_wv[EMBED_DIM],
    float         Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float         K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float         V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {
        const float si = scale_input[i];
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int col = h * HEAD_DIM + d;

                int32_t acc_q = 0, acc_k = 0, acc_v = 0;
                for (int e = 0; e < EMBED_DIM; e++) {
                    acc_q += (int32_t)X[i][e] * (int32_t)Wq[e][col];
                    acc_k += (int32_t)X[i][e] * (int32_t)Wk[e][col];
                    acc_v += (int32_t)X[i][e] * (int32_t)Wv[e][col];
                }

                Q[h][i][d] = (float)acc_q * si * scale_wq[col] + bq[col];
                K[h][i][d] = (float)acc_k * si * scale_wk[col] + bk[col];
                V[h][i][d] = (float)acc_v * si * scale_wv[col] + bv[col];
            }
        }
    }
}

// ============================================================================
// Helper Functions for File I/O
// ============================================================================
template <typename T>
void load_file_float(const char* filename, T* ptr, int total_elements) {
    ifstream infile(filename);
    if (!infile.is_open()) { cerr << "Error: Could not open " << filename << endl; exit(1); }
    for (int i = 0; i < total_elements; i++) {
        if (!(infile >> ptr[i])) {
            cerr << "Error: Data mismatch or missing elements in " << filename << " at index " << i << endl; exit(1);
        }
    }
    cout << "Loaded: " << filename << endl;
}

void load_file_int8(const char* filename, int8_t* ptr, int total_elements) {
    ifstream infile(filename);
    if (!infile.is_open()) { cerr << "Error: Could not open " << filename << endl; exit(1); }
    int temp;
    for (int i = 0; i < total_elements; i++) {
        if (!(infile >> temp)) {
            cerr << "Error: Data mismatch or missing elements in " << filename << " at index " << i << endl; exit(1);
        }
        ptr[i] = static_cast<int8_t>(temp);
    }
    cout << "Loaded: " << filename << endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    // 1. Data Allocation
    float (*x_in)[EMBED_DIM]     = new float[SEQ_LEN][EMBED_DIM];
    float (*ln_out)[EMBED_DIM]   = new float[SEQ_LEN][EMBED_DIM];
    float *gamma                 = new float[EMBED_DIM];
    float *beta                  = new float[EMBED_DIM];

    int8_t (*Wq)[EMBED_DIM]      = new int8_t[EMBED_DIM][EMBED_DIM];
    int8_t (*Wk)[EMBED_DIM]      = new int8_t[EMBED_DIM][EMBED_DIM];
    int8_t (*Wv)[EMBED_DIM]      = new int8_t[EMBED_DIM][EMBED_DIM];

    float *bq = new float[EMBED_DIM]; float *bk = new float[EMBED_DIM]; float *bv = new float[EMBED_DIM];
    float *scale_wq = new float[EMBED_DIM]; float *scale_wk = new float[EMBED_DIM]; float *scale_wv = new float[EMBED_DIM];

    // Bridge buffers between LayerNorm and QKV
    int8_t (*q_X)[EMBED_DIM]     = new int8_t[SEQ_LEN][EMBED_DIM];
    float *scale_X               = new float[SEQ_LEN];

    // Outputs
    float (*Q)[SEQ_LEN][HEAD_DIM] = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    float (*K)[SEQ_LEN][HEAD_DIM] = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    float (*V)[SEQ_LEN][HEAD_DIM] = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    // 2. Load the 12 input files (matching the image specification)
    cout << "--- Loading Inputs ---" << endl;
    load_file_float("input_x.txt",  &x_in[0][0],   SEQ_LEN * EMBED_DIM);
    load_file_float("gamma.txt",    gamma,         EMBED_DIM);
    load_file_float("beta.txt",     beta,          EMBED_DIM);

    // Load Weights as-is (No transposition needed as requested)
    load_file_int8("Wk.txt",        &Wk[0][0],     EMBED_DIM * EMBED_DIM);
    load_file_int8("Wv.txt",        &Wv[0][0],     EMBED_DIM * EMBED_DIM);
    load_file_int8("Wq.txt",        &Wq[0][0],     EMBED_DIM * EMBED_DIM);

    // load_file_int8("Wk_transposed.txt",        &Wk[0][0],     EMBED_DIM * EMBED_DIM);
    // load_file_int8("Wv_transposed.txt",        &Wv[0][0],     EMBED_DIM * EMBED_DIM);
    // load_file_int8("Wq_transposed.txt",        &Wq[0][0],     EMBED_DIM * EMBED_DIM);

    load_file_float("bq.txt",       bq,            EMBED_DIM);
    load_file_float("bk.txt",       bk,            EMBED_DIM);
    load_file_float("bv.txt",       bv,            EMBED_DIM);

    load_file_float("scale_wq.txt", scale_wq,      EMBED_DIM);
    load_file_float("scale_wk.txt", scale_wk,      EMBED_DIM);
    load_file_float("scale_wv.txt", scale_wv,      EMBED_DIM);

    // 3. Compute LayerNorm
    cout << "\n--- Computing LayerNorm ---" << endl;
    layernorm(x_in, gamma, beta, ln_out, SEQ_LEN, EMBED_DIM, EPSILON);

    // 4. Dynamic Quantization (Bridging LN output to INT8 QKV input)
    cout << "--- Performing Dynamic Quantization on LayerNorm output ---" << endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        // Find Max Absolute Value in the current row
        float max_val = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) {
            max_val = max(max_val, abs(ln_out[i][j]));
        }
        
        max_val = max(max_val, EPSILON);
        float scale = 127.5f / max_val;
        
        // Quantize and store
        for (int j = 0; j < EMBED_DIM; j++) {
            float scaled_val = ln_out[i][j] * scale;
            scaled_val = max(-128.0f, min(127.0f, round(scaled_val)));
            q_X[i][j] = static_cast<int8_t>(round(scaled_val));
        }
        
        // Compute inverse scale for the hardware multiplier
        scale_X[i] = 1.0f / scale;
    }

    // 5. Compute QKV Matrix Multiplication
    cout << "--- Computing Golden QKV ---" << endl;
    golden_qkv(q_X, Wq, Wk, Wv, bq, bk, bv, scale_X, scale_wq, scale_wk, scale_wv, Q, K, V);

    // 6. Output to files (Newline every 64 elements = HEAD_DIM)
    cout << "\n--- Writing Outputs ---" << endl;
    const char* out_files[3] = {"Q_output_golden.txt", "K_output_golden.txt", "V_output_golden.txt"};
    float (*out_ptrs[3])[SEQ_LEN][HEAD_DIM] = {Q, K, V};

    for (int t = 0; t < 3; t++) {
        ofstream out_stream(out_files[t]);
        if (!out_stream.is_open()) { cerr << "Error: Could not open " << out_files[t] << " for writing." << endl; return 1; }
        
        out_stream << fixed << setprecision(6);
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int i = 0; i < SEQ_LEN; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    out_stream << out_ptrs[t][h][i][d] << " ";
                }
                out_stream << "\n"; // Newline exactly every 64 elements
            }
        }
        out_stream.close();
        cout << "Wrote to: " << out_files[t] << endl;
    }

    // 7. Cleanup
    delete[] x_in; delete[] ln_out; delete[] gamma; delete[] beta;
    delete[] Wq; delete[] Wk; delete[] Wv;
    delete[] bq; delete[] bk; delete[] bv;
    delete[] scale_wq; delete[] scale_wk; delete[] scale_wv;
    delete[] q_X; delete[] scale_X;
    delete[] Q; delete[] K; delete[] V;

    cout << "\nSuccess: Golden references generated completely!" << endl;
    return 0;
}