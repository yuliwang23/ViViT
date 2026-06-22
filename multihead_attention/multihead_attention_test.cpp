#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <ap_int.h>
#include "multihead_attention.h" 

using namespace std;

union FloatInt {
    float f;
    uint32_t i;
};

// Quantize FP32 weights to INT8 per-channel
void quantize_weight_per_channel(
    float W_fp32[EMBED_DIM][EMBED_DIM],
    int8_t W_int8[EMBED_DIM][EMBED_DIM],
    float hw_scale[EMBED_DIM]
) {
    for (int j = 0; j < EMBED_DIM; j++) {
        float max_abs = 1e-5f;
        for (int i = 0; i < EMBED_DIM; i++) {
            max_abs = std::max(max_abs, std::abs(W_fp32[i][j]));
        }
        float quant_scaler = 127.5f / max_abs;
        for (int i = 0; i < EMBED_DIM; i++) {
            float scaled_val = W_fp32[i][j] * quant_scaler;
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            W_int8[i][j] = static_cast<int8_t>(scaled_val);
        }
        hw_scale[j] = 1.0f / quant_scaler; 
    }
}

// Golden Reference Model for Full Multi-Head Attention
void multihead_attention_golden(
    float X_fp32[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM], float beta[EMBED_DIM],
    int8_t Wq[EMBED_DIM][EMBED_DIM], float bq[EMBED_DIM], float sq[EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM], float bk[EMBED_DIM], float sk[EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM], float bv[EMBED_DIM], float sv[EMBED_DIM],
    int8_t Wo[EMBED_DIM][EMBED_DIM], float bo[EMBED_DIM], float so[EMBED_DIM],
    float Out_ref[SEQ_LEN][EMBED_DIM])
{
    // Step 1: LayerNorm & Quantization
    vector<vector<int8_t>> X_int8(SEQ_LEN, vector<int8_t>(EMBED_DIM));
    vector<float> token_scale(SEQ_LEN);

    for (int i = 0; i < SEQ_LEN; i++) {
        float global_mean = 0.0f; 
        float M2AB = 0.0f;

        for(int j = 0, iter = 0; j < EMBED_DIM; j+=16, iter++){     
            float local_sum = 0.0f;
            float local_sum_sq = 0.0f;
            for(int k = 0; k < 16; k++){
                float val = X_fp32[i][j+k];
                local_sum += val;
                local_sum_sq += val * val;
            }
            float X_b_mean = local_sum * 0.0625f; 
            float mean_diff = X_b_mean - global_mean;
            float M_2_B = local_sum_sq - (local_sum * X_b_mean);
            float mean_diff_sq = mean_diff * mean_diff;
            
            global_mean = MEAN_WT[iter] * mean_diff + global_mean;
            M2AB = M2_WT[iter] * mean_diff_sq + M_2_B + M2AB;
        }
        
        float var = M2AB * EMBED_DIM_INV; 
        float inv_std = 1.0f / std::sqrt(var + EPSILON); 

        vector<float> norm_buffer(EMBED_DIM);
        float max_val = 0.0f;
        for(int j = 0; j < EMBED_DIM; j+=16){
            float local_max = 0.0f;
            for(int k = 0; k < 16; k++){
                int idx = j + k;
                norm_buffer[idx] = (X_fp32[i][idx] - global_mean) * inv_std * gamma[idx] + beta[idx];
                local_max = std::max(local_max, std::abs(norm_buffer[idx]));
            }
            max_val = std::max(max_val, local_max);
        }
        max_val = std::max(max_val, EPSILON);

        float scale = 127.5f / max_val;
        token_scale[i] = 1.0f / scale;
        
        for (int j = 0; j < EMBED_DIM; j++) {
            float scaled_val = norm_buffer[j] * scale;
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            X_int8[i][j] = static_cast<int8_t>(scaled_val);
        }
    }

    // Step 2: Linear QKV
    vector<vector<vector<float>>> Q(NUM_HEADS, vector<vector<float>>(SEQ_LEN, vector<float>(HEAD_DIM)));
    vector<vector<vector<float>>> K(NUM_HEADS, vector<vector<float>>(SEQ_LEN, vector<float>(HEAD_DIM)));
    vector<vector<vector<float>>> V(NUM_HEADS, vector<vector<float>>(SEQ_LEN, vector<float>(HEAD_DIM)));

    for (int i = 0; i < SEQ_LEN; i++) {
        float si = token_scale[i];
        for (int j = 0; j < EMBED_DIM; j++) {
            int32_t acc_q = 0, acc_k = 0, acc_v = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                acc_q += (int32_t)X_int8[i][k] * (int32_t)Wq[k][j];
                acc_k += (int32_t)X_int8[i][k] * (int32_t)Wk[k][j];
                acc_v += (int32_t)X_int8[i][k] * (int32_t)Wv[k][j];
            }
            int head_idx = j / HEAD_DIM;
            int head_dim = j % HEAD_DIM;
            Q[head_idx][i][head_dim] = (float)acc_q * si * sq[j] + bq[j];
            K[head_idx][i][head_dim] = (float)acc_k * si * sk[j] + bk[j];
            V[head_idx][i][head_dim] = (float)acc_v * si * sv[j] + bv[j];
        }
    }

    // Step 3: Multi-Head Attention Score & Context
    vector<vector<float>> Context(SEQ_LEN, vector<float>(EMBED_DIM));
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            vector<float> scores(SEQ_LEN);
            float max_score = NEG_INF;
            
            // Q * K^T
            for (int j = 0; j < SEQ_LEN; j++) {
                float acc = 0.0f;
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc += Q[h][i][d] * K[h][j][d];
                }
                scores[j] = acc;
                if (acc > max_score) max_score = acc;
            }
            
            // Softmax
            float exp_sum = 0.0f;
            for (int j = 0; j < SEQ_LEN; j++) {
                scores[j] = std::exp(scores[j] - max_score);
                exp_sum += scores[j];
            }
            for (int j = 0; j < SEQ_LEN; j++) {
                scores[j] /= exp_sum;
            }
            
            // Attn * V
            for (int d = 0; d < HEAD_DIM; d++) {
                float acc = 0.0f;
                for (int j = 0; j < SEQ_LEN; j++) {
                    acc += scores[j] * V[h][j][d];
                }
                Context[i][h * HEAD_DIM + d] = acc;
            }
        }
    }

    // Step 4 & 5: Output Projection & Residual
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            float acc_o = 0.0f;
            // Note: Emulating FP32 MAC for Output Projection to avoid complex INT8 re-quantize in Golden
            for (int k = 0; k < EMBED_DIM; k++) {
                acc_o += Context[i][k] * (float)Wo[k][j] * so[j]; 
            }
            Out_ref[i][j] = acc_o + bo[j] + X_fp32[i][j]; // Output Proj + Residual
        }
    }
}

int main() {
    srand(42); 

    cout << "Allocating memory via Heap to prevent Windows WrapC destructor crash..." << endl;
    
    // FP32 Matrices
    auto h_X_fp32 = new float[SEQ_LEN][EMBED_DIM];
    auto h_Wq_fp32 = new float[EMBED_DIM][EMBED_DIM];
    auto h_Wk_fp32 = new float[EMBED_DIM][EMBED_DIM];
    auto h_Wv_fp32 = new float[EMBED_DIM][EMBED_DIM];
    auto h_Wo_fp32 = new float[EMBED_DIM][EMBED_DIM]; // New: Output Projection Weight
    
    // INT8 Matrices
    auto h_Wq = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wk = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wv = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wo = new int8_t[EMBED_DIM][EMBED_DIM];
    
    // Transposed INT8 Matrices (For HLS Packing)
    auto h_Wq_T = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wk_T = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wv_T = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wo_T = new int8_t[EMBED_DIM][EMBED_DIM];
    
    // Biases & Scales
    auto h_bq = new float[EMBED_DIM]; auto h_bk = new float[EMBED_DIM]; 
    auto h_bv = new float[EMBED_DIM]; auto h_bo = new float[EMBED_DIM];
    auto h_sq = new float[EMBED_DIM]; auto h_sk = new float[EMBED_DIM]; 
    auto h_sv = new float[EMBED_DIM]; auto h_so = new float[EMBED_DIM];
    auto h_gamma = new float[EMBED_DIM]; auto h_beta = new float[EMBED_DIM];

    // Output Tensors
    auto h_Out_opt = new float[SEQ_LEN][EMBED_DIM];
    auto h_Out_ref = new float[SEQ_LEN][EMBED_DIM];

    const int PACK_INT8 = 256 / 8;   
    const int PACK_FLOAT = 256 / 32; 

    // Packed Hardware Structures
    uint256_dt* packed_X  = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT]; 
    uint256_dt* packed_Wq = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Wk = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Wv = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Wo = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Out = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT];

    cout << "Initializing random data..." << endl;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_X_fp32[i][j] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        }
    }
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_Wq_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
            h_Wk_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
            h_Wv_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
            h_Wo_fp32[i][j] = -0.5f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 1.0f));
        }
    }
    for(int i = 0; i < EMBED_DIM; i++){
        h_bq[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_bk[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_bv[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_bo[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_gamma[i] = 1.0f;
        h_beta[i]  = 0.0f;
    }
    
    quantize_weight_per_channel(h_Wq_fp32, h_Wq, h_sq);
    quantize_weight_per_channel(h_Wk_fp32, h_Wk, h_sk);
    quantize_weight_per_channel(h_Wv_fp32, h_Wv, h_sv);
    quantize_weight_per_channel(h_Wo_fp32, h_Wo, h_so);

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_Wq_T[j][i] = h_Wq[i][j];
            h_Wk_T[j][i] = h_Wk[i][j];
            h_Wv_T[j][i] = h_Wv[i][j];
            h_Wo_T[j][i] = h_Wo[i][j];
        }
    }

    cout << "Calculating Golden Reference..." << endl;
    multihead_attention_golden(
        h_X_fp32, h_gamma, h_beta, 
        h_Wq, h_bq, h_sq, 
        h_Wk, h_bk, h_sk, 
        h_Wv, h_bv, h_sv, 
        h_Wo, h_bo, h_so, 
        h_Out_ref
    );

    cout << "Packing data for Hardware Kernel..." << endl;
    int pack_idx = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j += PACK_FLOAT) {
            uint256_dt wide_word = 0;
            for (int k = 0; k < PACK_FLOAT; k++) {
                FloatInt f_val;
                f_val.f = h_X_fp32[i][j + k];
                wide_word((k + 1) * 32 - 1, k * 32) = f_val.i;
            }
            packed_X[pack_idx++] = wide_word;
        }
    }

    pack_idx = 0;
    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j += PACK_INT8) {
            uint256_dt word_q = 0, word_k = 0, word_v = 0, word_o = 0;
            for (int k = 0; k < PACK_INT8; k++) {
                word_q((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wq_T[i][j + k];
                word_k((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wk_T[i][j + k];
                word_v((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wv_T[i][j + k];
                word_o((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wo_T[i][j + k];
            }
            packed_Wq[pack_idx] = word_q;
            packed_Wk[pack_idx] = word_k;
            packed_Wv[pack_idx] = word_v;
            packed_Wo[pack_idx] = word_o;
            pack_idx++;
        }
    }

    uint256_dt* packed_gamma = new uint256_dt[EMBED_DIM / PACK_FLOAT];
    uint256_dt* packed_beta = new uint256_dt[EMBED_DIM / PACK_FLOAT];
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

    cout << "Running HLS Kernel Simulation..." << endl;
    multihead_attention(
        packed_X,       // Serves as both Input and Residual
        packed_gamma, packed_beta,
        packed_Wq, packed_Wk, packed_Wv, packed_Wo,
        h_bq, h_bk, h_bv, h_bo,
        h_sq, h_sk, h_sv, h_so,
        packed_Out      // Final Output
    );

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
    
    // Note: Tolerance is set slightly higher because the Golden Model uses FP32 
    // for Attention & Output Projection, while the HLS hardware implements 
    // full INT8 quantization internally for these modules.
    const float TOLERANCE = 5.0f; 

    for (int i = 0; i < SEQ_LEN; i++) {
        for (int d = 0; d < EMBED_DIM; d++) {
            float diff = fabs(h_Out_ref[i][d] - h_Out_opt[i][d]);
            if (diff > max_err) max_err = diff;
            if (diff > TOLERANCE) {
                // To avoid spamming the console, only print first 10 errors
                if (err_cnt < 10) { 
                    printf("Mismatch at Out[%d][%d]: Ref=%f, Opt=%f (diff=%f)\n", i, d, h_Out_ref[i][d], h_Out_opt[i][d], diff);
                }
                err_cnt++;
            }
        }
    }

    printf("\nVerification Result:\n");
    printf("Max Error: %f\n", max_err);
    if (err_cnt == 0) {
        printf(">> C-Simulation PASSED!\n");
    } else {
        printf(">> C-Simulation FAILED with %d errors.\n", err_cnt);
    }

    // Cleanup Dynamic Memory
    delete[] h_X_fp32; delete[] h_Wq_fp32; delete[] h_Wk_fp32; delete[] h_Wv_fp32; delete[] h_Wo_fp32;
    delete[] h_Wq; delete[] h_Wk; delete[] h_Wv; delete[] h_Wo;
    delete[] h_Wq_T; delete[] h_Wk_T; delete[] h_Wv_T; delete[] h_Wo_T;
    delete[] h_bq; delete[] h_bk; delete[] h_bv; delete[] h_bo;
    delete[] h_sq; delete[] h_sk; delete[] h_sv; delete[] h_so;
    delete[] h_gamma; delete[] h_beta;
    delete[] h_Out_opt; delete[] h_Out_ref;
    
    delete[] packed_X; delete[] packed_Wq; delete[] packed_Wk; delete[] packed_Wv; delete[] packed_Wo;
    delete[] packed_Out; delete[] packed_gamma; delete[] packed_beta;

    return (err_cnt == 0) ? 0 : 1;
}