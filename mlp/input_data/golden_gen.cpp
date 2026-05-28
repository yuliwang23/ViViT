#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iomanip>

using namespace std;

// ============================================================================
// MLP Dimensions  (對應 mlp.h)
// ============================================================================
const int   SEQ_LEN    = 197;
const int   EMBED_DIM  = 768;    // = IN_DIM = OUT_DIM
const int   HIDDEN_DIM = 3072;   // 4 * EMBED_DIM
const float EPSILON    = 1e-5f;

// ============================================================================
// 精確 GELU (erf 版本)
//   GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
// ============================================================================
static inline float gelu_exact(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

// ============================================================================
// LayerNorm  (對應 do_layernorm_dataflow / compute_mean_variance)
//   out = gamma * (x - mean) * inv_std + beta
// ============================================================================
void layernorm(const float x[SEQ_LEN][EMBED_DIM],
               const float gamma[EMBED_DIM],
               const float beta[EMBED_DIM],
               float       out[SEQ_LEN][EMBED_DIM],
               int M, int N, float eps)
{
    for (int i = 0; i < M; i++) {
        float mean = 0.0f;
        for (int j = 0; j < N; j++) mean += x[i][j];
        mean /= N;

        float var = 0.0f;
        for (int j = 0; j < N; j++) {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= N;
        float inv_std = 1.0f / sqrtf(var + eps);

        for (int j = 0; j < N; j++) {
            out[i][j] = gamma[j] * ((x[i][j] - mean) * inv_std) + beta[j];
        }
    }
}

// ============================================================================
// Dynamic Per-Row Quantization (對應 compute_mean_variance 的量化段
//   及 Stage 2b 的 quant_hidden_loop)
//   scale = 127.5 / max_abs ;  q = round(v * scale) , clamp[-128,127]
//   回傳 inverse scale (= 1/scale = max_abs/127.5) 供反量化用
// ----------------------------------------------------------------------------
//   注意:LayerNorm 段硬體用 std::round;Stage 2b 用 (f>=0)?(f+0.5):(f-0.5)。
//   兩者對 .5 邊界的行為僅在極少數情況不同,golden 統一採 round-half-away,
//   與硬體 Stage 2b 一致,且對 LayerNorm 段在數值上幾乎不可區分。
// ============================================================================
static inline int8_t quantize_one(float v, float scale) {
    float f  = v * scale;
    float qf = (f >= 0.0f) ? floorf(f + 0.5f) : ceilf(f - 0.5f); // round half away from zero
    if (qf >  127.0f) qf =  127.0f;
    if (qf < -128.0f) qf = -128.0f;
    return static_cast<int8_t>(qf);
}

template <int ROWS, int COLS>
void quantize_rows(const float in[ROWS][COLS],
                   int8_t      q_out[ROWS][COLS],
                   float       inv_scale[ROWS],
                   int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        float max_val = 0.0f;
        for (int j = 0; j < cols; j++) {
            max_val = max(max_val, fabsf(in[i][j]));
        }
        if (max_val < EPSILON) max_val = EPSILON;
        float scale = 127.5f / max_val;

        for (int j = 0; j < cols; j++) {
            q_out[i][j] = quantize_one(in[i][j], scale);
        }
        inv_scale[i] = 1.0f / scale;  // = max_val / 127.5
    }
}

// ============================================================================
// Dense layer:  out[i][col] = (sum_e X[i][e] * W[col][e]) * sX[i]*sW[col] + b[col]
//   權重佈局:W 以 [OUT][IN] 排列 (row = output feature),與 HLS W_tile 載入一致:
//     W1: [HIDDEN_DIM][EMBED_DIM]
//     W2: [EMBED_DIM ][HIDDEN_DIM]
//   ( mlp.cpp 中 W_tile[t][k] 為第 t 個 output feature 的第 k 個 input package,
//     即權重以 output-major 儲存 )
// ============================================================================
template <int OUT, int IN, int ROWS>
void dense_int8(const int8_t X[ROWS][IN],
                const int8_t W[OUT][IN],
                const float  bias[OUT],
                const float  scale_x[ROWS],
                const float  scale_w[OUT],
                float        out[ROWS][OUT],
                int rows)
{
    for (int i = 0; i < rows; i++) {
        const float si = scale_x[i];
        for (int col = 0; col < OUT; col++) {
            int32_t acc = 0;
            for (int e = 0; e < IN; e++) {
                acc += (int32_t)X[i][e] * (int32_t)W[col][e];
            }
            out[i][col] = (float)acc * si * scale_w[col] + bias[col];
        }
    }
}

// ============================================================================
// File I/O helpers (沿用 golden_qkv_gen.cpp 風格)
// ============================================================================
template <typename T>
void load_file_float(const char* filename, T* ptr, int total_elements) {
    ifstream infile(filename);
    if (!infile.is_open()) { cerr << "Error: Could not open " << filename << endl; exit(1); }
    for (int i = 0; i < total_elements; i++) {
        if (!(infile >> ptr[i])) {
            cerr << "Error: Data mismatch in " << filename << " at index " << i << endl; exit(1);
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
            cerr << "Error: Data mismatch in " << filename << " at index " << i << endl; exit(1);
        }
        ptr[i] = static_cast<int8_t>(temp);
    }
    cout << "Loaded: " << filename << endl;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    // ---- 1. Allocation ----
    float (*x_in)[EMBED_DIM]    = new float[SEQ_LEN][EMBED_DIM];
    float (*ln_out)[EMBED_DIM]  = new float[SEQ_LEN][EMBED_DIM];
    float *gamma                = new float[EMBED_DIM];
    float *beta                 = new float[EMBED_DIM];

    // W1: [HIDDEN_DIM][EMBED_DIM]  (output-major)
    int8_t (*W1)[EMBED_DIM]     = new int8_t[HIDDEN_DIM][EMBED_DIM];
    // W2: [EMBED_DIM][HIDDEN_DIM]  (output-major)
    int8_t (*W2)[HIDDEN_DIM]    = new int8_t[EMBED_DIM][HIDDEN_DIM];

    float *b1       = new float[HIDDEN_DIM];
    float *b2       = new float[EMBED_DIM];
    float *scale_w1 = new float[HIDDEN_DIM];
    float *scale_w2 = new float[EMBED_DIM];

    // 中間 buffer
    int8_t (*q_X)[EMBED_DIM]    = new int8_t[SEQ_LEN][EMBED_DIM];
    float  *scale_X             = new float[SEQ_LEN];

    float  (*H_fp)[HIDDEN_DIM]  = new float[SEQ_LEN][HIDDEN_DIM];  // GELU 後
    int8_t (*q_H)[HIDDEN_DIM]   = new int8_t[SEQ_LEN][HIDDEN_DIM];
    float  *scale_H             = new float[SEQ_LEN];

    float  (*Y)[EMBED_DIM]      = new float[SEQ_LEN][EMBED_DIM];   // 最終輸出

    // ---- 2. Load inputs ----
    cout << "--- Loading Inputs ---" << endl;
    load_file_float("input.txt",  &x_in[0][0], SEQ_LEN * EMBED_DIM);
    load_file_float("gamma.txt",    gamma,       EMBED_DIM);
    load_file_float("beta.txt",     beta,        EMBED_DIM);

    load_file_int8 ("W1.txt",       &W1[0][0],   HIDDEN_DIM * EMBED_DIM);
    load_file_int8 ("W2.txt",       &W2[0][0],   EMBED_DIM  * HIDDEN_DIM);

    load_file_float("b1.txt",       b1,          HIDDEN_DIM);
    load_file_float("b2.txt",       b2,          EMBED_DIM);
    load_file_float("scale_w1.txt", scale_w1,    HIDDEN_DIM);
    load_file_float("scale_w2.txt", scale_w2,    EMBED_DIM);

    // ---- 3. LayerNorm ----
    cout << "\n--- Computing LayerNorm ---" << endl;
    layernorm(x_in, gamma, beta, ln_out, SEQ_LEN, EMBED_DIM, EPSILON);

    // ---- 4. Dynamic Quantization (LN output -> int8) ----
    cout << "--- Quantizing LayerNorm output ---" << endl;
    quantize_rows<SEQ_LEN, EMBED_DIM>(ln_out, q_X, scale_X, SEQ_LEN, EMBED_DIM);

    // ---- 5. Dense0 (W1) ----
    cout << "--- Computing Dense0 (W1) ---" << endl;
    dense_int8<HIDDEN_DIM, EMBED_DIM, SEQ_LEN>(
        q_X, W1, b1, scale_X, scale_w1, H_fp, SEQ_LEN);

    // ---- 6. GELU (精確 erf) ----
    cout << "--- Applying exact GELU ---" << endl;
    for (int i = 0; i < SEQ_LEN; i++)
        for (int j = 0; j < HIDDEN_DIM; j++)
            H_fp[i][j] = gelu_exact(H_fp[i][j]);

    // ---- 7. Dynamic Quantization (hidden -> int8) ----
    cout << "--- Quantizing hidden activation ---" << endl;
    quantize_rows<SEQ_LEN, HIDDEN_DIM>(H_fp, q_H, scale_H, SEQ_LEN, HIDDEN_DIM);

    // ---- 8. Dense1 (W2) -> FP32 output ----
    cout << "--- Computing Dense1 (W2) ---" << endl;
    dense_int8<EMBED_DIM, HIDDEN_DIM, SEQ_LEN>(
        q_H, W2, b2, scale_H, scale_w2, Y, SEQ_LEN);

    // ---- 9. Write output (每 EMBED_DIM 個值換行,對齊一個 token 一行) ----
    cout << "\n--- Writing Output ---" << endl;
    {
        ofstream out("golden_output.txt");
        if (!out.is_open()) { cerr << "Error: cannot open output file" << endl; return 1; }
        out << fixed << setprecision(6);
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < EMBED_DIM; j++) {
                out << Y[i][j] << " ";
            }
            out << "\n";
        }
        out.close();
        cout << "Wrote to: golden_output.txt" << endl;
    }

    // ---- 10. Cleanup ----
    delete[] x_in; delete[] ln_out; delete[] gamma; delete[] beta;
    delete[] W1; delete[] W2;
    delete[] b1; delete[] b2; delete[] scale_w1; delete[] scale_w2;
    delete[] q_X; delete[] scale_X;
    delete[] H_fp; delete[] q_H; delete[] scale_H;
    delete[] Y;

    cout << "\nSuccess: MLP golden reference generated!" << endl;
    return 0;
}