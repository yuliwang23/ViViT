#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <ap_int.h>
#include "mlp.h" 

using namespace std;

#define NEG_INF -1e9f

union FloatInt {
    float f;
    uint32_t i;
};

// =======================================================================
//   Helper: Generalized Quantization for Arbitrary Matrix Sizes
// =======================================================================
template <int ROWS, int COLS>
void quantize_weight_per_channel(
    float W_fp32[ROWS][COLS],
    int8_t W_int8[ROWS][COLS],
    float hw_scale[COLS]
) {
    for (int j = 0; j < COLS; j++) {
        float max_abs = 1e-5f;
        for (int i = 0; i < ROWS; i++) {
            max_abs = std::max(max_abs, std::abs(W_fp32[i][j]));
        }
        float quant_scaler = 127.5f / max_abs;
        for (int i = 0; i < ROWS; i++) {
            float scaled_val = W_fp32[i][j] * quant_scaler;
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            W_int8[i][j] = static_cast<int8_t>(scaled_val);
        }
        hw_scale[j] = 1.0f / quant_scaler; 
    }
}

// =======================================================================
//   Helper: Software Simulation of Hardware GELU (MUX + fast_exp)
// =======================================================================
inline float gelu_hw_sim(float x) {
    float z = 1.702f * x;
    float ax = std::abs(z);
    float e = std::exp(-ax); // Match hls::exp behavior
    float denom = 1.0f + e;
    float sig = 1.0f / denom;
    float final_sig = (x < 0.0f) ? (1.0f - sig) : sig;
    return x * final_sig;
}

// =======================================================================
//   Golden Reference Model for MLP (LayerNorm -> Lin1 -> GELU -> Lin2 -> Res)
// =======================================================================
void mlp_golden(
    float X_fp32[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM], float beta[EMBED_DIM],
    int8_t W1[EMBED_DIM][MLP_DIM], float b1[MLP_DIM], float scale_w1[MLP_DIM],
    int8_t W2[MLP_DIM][EMBED_DIM], float b2[EMBED_DIM], float scale_w2[EMBED_DIM],
    float Out_ref[SEQ_LEN][EMBED_DIM])
{
    // --------------------------------------------------------------------
    // Step 1: LayerNorm & Input Quantization (2-Pass / 2N Algorithm)
    // --------------------------------------------------------------------
    vector<vector<int8_t>> X_int8(SEQ_LEN, vector<int8_t>(EMBED_DIM));
    vector<float> token_scale_X(SEQ_LEN);

    for (int i = 0; i < SEQ_LEN; i++) {
        float sum = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) sum += X_fp32[i][j];
        float global_mean = sum * EMBED_DIM_INV;

        float variance_sum = 0.0f;
        for (int j = 0; j < EMBED_DIM; j++) {
            float diff = X_fp32[i][j] - global_mean;
            variance_sum += diff * diff;
        }
        float var = variance_sum * EMBED_DIM_INV; 
        float inv_std = 1.0f / std::sqrt(var + EPSILON); 

        vector<float> norm_buffer(EMBED_DIM);
        float max_val = 0.0f;
        
        for (int j = 0; j < EMBED_DIM; j++) {
            norm_buffer[j] = (X_fp32[i][j] - global_mean) * inv_std * gamma[j] + beta[j];
            max_val = std::max(max_val, std::abs(norm_buffer[j]));
        }
        max_val = std::max(max_val, EPSILON);

        float scale = 127.5f / max_val;
        token_scale_X[i] = 1.0f / scale;
        
        for (int j = 0; j < EMBED_DIM; j++) {
            float scaled_val = norm_buffer[j] * scale;
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            X_int8[i][j] = static_cast<int8_t>(scaled_val);
        }
    }

    // --------------------------------------------------------------------
    // Step 2 & 3: Linear 1 & Hardware GELU
    // --------------------------------------------------------------------
    vector<vector<float>> Post_GELU_fp32(SEQ_LEN, vector<float>(MLP_DIM));
    vector<float> global_max_gelu(SEQ_LEN, NEG_INF);

    for (int i = 0; i < SEQ_LEN; i++) {
        float si = token_scale_X[i];
        for (int j = 0; j < MLP_DIM; j++) {
            int32_t acc = 0;
            // INT8 Dot Product
            for (int k = 0; k < EMBED_DIM; k++) {
                acc += (int32_t)X_int8[i][k] * (int32_t)W1[k][j];
            }
            // Dequantize to FP32
            float pre_gelu = (float)acc * si * scale_w1[j] + b1[j];
            
            // Apply HW-Aligned GELU
            float post_gelu = gelu_hw_sim(pre_gelu);
            Post_GELU_fp32[i][j] = post_gelu;
            
            // Track global max for Next Layer Quantization
            global_max_gelu[i] = std::max(global_max_gelu[i], std::abs(post_gelu));
        }
    }

    // --------------------------------------------------------------------
    // Step 4: Quantize GELU Output (Dynamic per Token)
    // --------------------------------------------------------------------
    vector<vector<int8_t>> Post_GELU_int8(SEQ_LEN, vector<int8_t>(MLP_DIM));
    vector<float> token_scale_MLP1(SEQ_LEN);

    for (int i = 0; i < SEQ_LEN; i++) {
        float max_val = std::max(global_max_gelu[i], EPSILON);
        float scale = 127.0f / max_val;
        token_scale_MLP1[i] = max_val / 127.0f; 

        for (int j = 0; j < MLP_DIM; j++) {
            float scaled_val = Post_GELU_fp32[i][j] * scale;
            Post_GELU_int8[i][j] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, scaled_val))));
        }
    }

    // --------------------------------------------------------------------
    // Step 5 & 6: Linear 2 & FP32 Residual Connection
    // --------------------------------------------------------------------
    for (int i = 0; i < SEQ_LEN; i++) {
        float si = token_scale_MLP1[i];
        for (int j = 0; j < EMBED_DIM; j++) {
            int32_t acc = 0;
            // INT8 Dot Product
            for (int k = 0; k < MLP_DIM; k++) {
                acc += (int32_t)Post_GELU_int8[i][k] * (int32_t)W2[k][j];
            }
            
            // Dequantize and add Original Input as Residual
            float proj_val = (float)acc * si * scale_w2[j] + b2[j];
            Out_ref[i][j] = proj_val + X_fp32[i][j];
        }
    }
}

// =======================================================================
//   Main Testbench
// =======================================================================
int main() {
    srand(42); 

    cout << "Allocating memory via Heap for MLP Tensors..." << endl;
    
    // FP32 Matrices
    auto h_X_fp32 = new float[SEQ_LEN][EMBED_DIM];
    auto h_W1_fp32 = new float[EMBED_DIM][MLP_DIM];
    auto h_W2_fp32 = new float[MLP_DIM][EMBED_DIM];
    
    // INT8 Matrices
    auto h_W1 = new int8_t[EMBED_DIM][MLP_DIM];
    auto h_W2 = new int8_t[MLP_DIM][EMBED_DIM];
    
    // Transposed INT8 Matrices (For HLS Data Packing)
    // W1_T is [3072][768], W2_T is [768][3072]
    auto h_W1_T = new int8_t[MLP_DIM][EMBED_DIM];
    auto h_W2_T = new int8_t[EMBED_DIM][MLP_DIM];
    
    // Biases & Scales
    auto h_b1 = new float[MLP_DIM]; auto h_b2 = new float[EMBED_DIM];
    auto h_scale_w1 = new float[MLP_DIM]; auto h_scale_w2 = new float[EMBED_DIM];
    auto h_gamma = new float[EMBED_DIM]; auto h_beta = new float[EMBED_DIM];

    // Output Tensors
    auto h_Out_opt = new float[SEQ_LEN][EMBED_DIM];
    auto h_Out_ref = new float[SEQ_LEN][EMBED_DIM];

    const int PACK_INT8 = 256 / 8;   // 32 int8 per 256-bit word
    const int PACK_FLOAT = 256 / 32; // 8 floats per 256-bit word

    // Packed Hardware Structures
    uint256_dt* packed_X     = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT]; 
    uint256_dt* packed_W1    = new uint256_dt[(EMBED_DIM * MLP_DIM) / PACK_INT8];
    uint256_dt* packed_W2    = new uint256_dt[(MLP_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Out   = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT];
    uint256_dt* packed_gamma = new uint256_dt[EMBED_DIM / PACK_FLOAT];
    uint256_dt* packed_beta  = new uint256_dt[EMBED_DIM / PACK_FLOAT];

    // NEW: Allocate HBM buffer for Intermediate FP32 data
    const int INTERMEDIATE_CHUNKS = (SEQ_LEN * MLP_DIM) / PACK_FLOAT;
    uint256_dt* packed_Intermediate = new uint256_dt[INTERMEDIATE_CHUNKS];

    // --------------------------------------------------------------------
    // Initialize Random Data
    // --------------------------------------------------------------------
    cout << "Initializing random data..." << endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_X_fp32[i][j] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        }
    }
    
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < MLP_DIM; j++) {
            h_W1_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
        }
    }
    for (int i = 0; i < MLP_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_W2_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
        }
    }
    
    for(int i = 0; i < MLP_DIM; i++) {
        h_b1[i] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
    }
    for(int i = 0; i < EMBED_DIM; i++) {
        h_b2[i] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
        h_gamma[i] = 1.0f;
        h_beta[i]  = 0.0f;
    }
    
    // Quantize Matrices
    quantize_weight_per_channel<EMBED_DIM, MLP_DIM>(h_W1_fp32, h_W1, h_scale_w1);
    quantize_weight_per_channel<MLP_DIM, EMBED_DIM>(h_W2_fp32, h_W2, h_scale_w2);

    // Transpose Matrices for Hardware Memory Layout
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < MLP_DIM; j++) {
            h_W1_T[j][i] = h_W1[i][j];
        }
    }
    for (int i = 0; i < MLP_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_W2_T[j][i] = h_W2[i][j];
        }
    }

    // --------------------------------------------------------------------
    // Execute Golden Model
    // --------------------------------------------------------------------
    cout << "Calculating Golden Reference..." << endl;
    mlp_golden(
        h_X_fp32, h_gamma, h_beta, 
        h_W1, h_b1, h_scale_w1, 
        h_W2, h_b2, h_scale_w2, 
        h_Out_ref
    );

    // --------------------------------------------------------------------
    // Pack Data for HLS
    // --------------------------------------------------------------------
    cout << "Packing data for Hardware Kernel..." << endl;
    
    int pack_idx = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j += PACK_FLOAT) {
            uint256_dt wide_word = 0;
            for (int k = 0; k < PACK_FLOAT; k++) {
                FloatInt f_val; f_val.f = h_X_fp32[i][j + k];
                wide_word((k + 1) * 32 - 1, k * 32) = f_val.i;
            }
            packed_X[pack_idx++] = wide_word;
        }
    }

    // Pack W1_T [3072][768]
    pack_idx = 0;
    for (int i = 0; i < MLP_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j += PACK_INT8) {
            uint256_dt word_w1 = 0;
            for (int k = 0; k < PACK_INT8; k++) {
                word_w1((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_W1_T[i][j + k];
            }
            packed_W1[pack_idx++] = word_w1;
        }
    }

    // Pack W2_T [768][3072]
    pack_idx = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < MLP_DIM; j += PACK_INT8) {
            uint256_dt word_w2 = 0;
            for (int k = 0; k < PACK_INT8; k++) {
                word_w2((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_W2_T[i][j + k];
            }
            packed_W2[pack_idx++] = word_w2;
        }
    }

    FloatInt f_one, f_zero;
    f_one.f = 1.0f; f_zero.f = 0.0f;
    for(int i = 0; i < EMBED_DIM / PACK_FLOAT; i++) {
        uint256_dt word_g = 0, word_b = 0;
        for (int k = 0; k < PACK_FLOAT; k++) {
            word_g((k + 1) * 32 - 1, k * 32) = f_one.i;
            word_b((k + 1) * 32 - 1, k * 32) = f_zero.i;
        }
        packed_gamma[i] = word_g; packed_beta[i] = word_b;
    }

    // --------------------------------------------------------------------
    // Execute HW Simulation
    // --------------------------------------------------------------------
    cout << "Running HLS Kernel Simulation..." << endl;
    mlp(
        packed_X,       // Serves as both Input and Residual
        packed_gamma, packed_beta,
        packed_W1, packed_W2,
        h_b1, h_b2,
        h_scale_w1, h_scale_w2,
        packed_Out,     // Final Output
        packed_Intermediate // NEW: HBM buffer for FP32 intermediate results
    );

    // --------------------------------------------------------------------
    // Unpack Results & Verify
    // --------------------------------------------------------------------
    cout << "Unpacking Results..." << endl;
    pack_idx = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j += PACK_FLOAT) {
            uint256_dt word_out = packed_Out[pack_idx++];
            for (int k = 0; k < PACK_FLOAT; k++) {
                FloatInt f_out;
                ap_uint<32> out_bits = word_out((k + 1) * 32 - 1, k * 32);
                f_out.i = out_bits.to_uint();
                h_Out_opt[i][j + k] = f_out.f;
            }
        }
    }

    cout << "Comparing results..." << endl;
    float max_err = 0.0f;
    int err_cnt = 0;
    const float TOLERANCE = 0.01f; 

    double sum_abs_err = 0.0, sum_sq_err  = 0.0;
    double dot_product = 0.0, norm_ref_sq = 0.0, norm_opt_sq = 0.0;
    int total_elements = SEQ_LEN * EMBED_DIM;

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            float ref_val = h_Out_ref[i][d];
            float opt_val = h_Out_opt[i][d];
            float diff = fabs(ref_val - opt_val);

            if (diff > max_err) max_err = diff;
            if (diff > TOLERANCE) err_cnt++;

            sum_abs_err += diff;
            sum_sq_err  += (diff * diff);
            dot_product += (ref_val * opt_val);
            norm_ref_sq += (ref_val * ref_val);
            norm_opt_sq += (opt_val * opt_val);
        }
    }

    double mae = sum_abs_err / total_elements;
    double mse = sum_sq_err / total_elements;
    double cos_sim = dot_product / (std::sqrt(norm_ref_sq) * std::sqrt(norm_opt_sq));

    printf("\n========================================\n");
    printf(" Verification Result\n");
    printf("========================================\n");
    printf("Max Error         : %f\n", max_err);
    printf("Error Count (>%g) : %d / %d (%.2f%%)\n", TOLERANCE, err_cnt, total_elements, (float)err_cnt / total_elements * 100.0f);
    printf("MAE (Mean Abs Err): %f\n", mae);
    printf("MSE (Mean Sq Err) : %f\n", mse);
    printf("Cosine Similarity : %f\n", cos_sim);
    printf("========================================\n");

    if (err_cnt == 0 || cos_sim > 0.99) {
        printf(">> C-Simulation PASSED! (or highly accurate)\n");
    } else {
        printf(">> C-Simulation FAILED with significant divergence.\n");
    }

    // Cleanup Dynamic Memory
    delete[] h_X_fp32; delete[] h_W1_fp32; delete[] h_W2_fp32;
    delete[] h_W1; delete[] h_W2; delete[] h_W1_T; delete[] h_W2_T;
    delete[] h_b1; delete[] h_b2; delete[] h_scale_w1; delete[] h_scale_w2;
    delete[] h_gamma; delete[] h_beta; delete[] h_Out_opt; delete[] h_Out_ref;
    
    delete[] packed_X; delete[] packed_W1; delete[] packed_W2;
    delete[] packed_Out; delete[] packed_gamma; delete[] packed_beta;
    delete[] packed_Intermediate; // NEW: Free the intermediate buffer

    return (err_cnt == 0) ? 0 : 1;
}