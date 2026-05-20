#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <ap_int.h>
#include "linear_qkv_multihead.h" 

using namespace std;

union FloatInt {
    float f;
    uint32_t i;
};

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

void linear_qkv_golden(
    float X_fp32[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM], float beta[EMBED_DIM],
    int8_t Wq[EMBED_DIM][EMBED_DIM], float bq[EMBED_DIM], float sq[EMBED_DIM],
    int8_t Wk[EMBED_DIM][EMBED_DIM], float bk[EMBED_DIM], float sk[EMBED_DIM],
    int8_t Wv[EMBED_DIM][EMBED_DIM], float bv[EMBED_DIM], float sv[EMBED_DIM],
    float Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
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

        // Temporary buffers for layer normalization
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
        float si = 1.0f / scale;
        vector<int8_t> X_int8(EMBED_DIM);
        for (int j = 0; j < EMBED_DIM; j++) {
            float scaled_val = norm_buffer[j] * scale;
            scaled_val = std::max(-128.0f, std::min(127.0f, std::round(scaled_val)));
            X_int8[j] = static_cast<int8_t>(scaled_val);
        }

        for (int j = 0; j < EMBED_DIM; j++) {
            int32_t acc_q = 0, acc_k = 0, acc_v = 0;
            for (int k = 0; k < EMBED_DIM; k++) {
                acc_q += (int32_t)X_int8[k] * (int32_t)Wq[k][j];
                acc_k += (int32_t)X_int8[k] * (int32_t)Wk[k][j];
                acc_v += (int32_t)X_int8[k] * (int32_t)Wv[k][j];
            }
            float res_q = (float)acc_q * si * sq[j] + bq[j];
            float res_k = (float)acc_k * si * sk[j] + bk[j];
            float res_v = (float)acc_v * si * sv[j] + bv[j];

            int head_idx = j / HEAD_DIM;
            int head_dim = j % HEAD_DIM;
            Q[head_idx][i][head_dim] = res_q;
            K[head_idx][i][head_dim] = res_k;
            V[head_idx][i][head_dim] = res_v;
        }
    }
}

int main() {
    srand(42); 

    cout << "Allocating memory via Heap to prevent Windows WrapC destructor crash..." << endl;
    
    // Dynamically allocate multi-dimensional arrays to avoid stack overflow and static cleanup bugs
    auto h_X_fp32 = new float[SEQ_LEN][EMBED_DIM];
    auto h_Wq_fp32 = new float[EMBED_DIM][EMBED_DIM];
    auto h_Wk_fp32 = new float[EMBED_DIM][EMBED_DIM];
    auto h_Wv_fp32 = new float[EMBED_DIM][EMBED_DIM];
    
    auto h_Wq = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wk = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wv = new int8_t[EMBED_DIM][EMBED_DIM];
    
    auto h_Wq_T = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wk_T = new int8_t[EMBED_DIM][EMBED_DIM];
    auto h_Wv_T = new int8_t[EMBED_DIM][EMBED_DIM];
    
    auto h_bq = new float[EMBED_DIM];
    auto h_bk = new float[EMBED_DIM];
    auto h_bv = new float[EMBED_DIM];
    auto h_sq = new float[EMBED_DIM];
    auto h_sk = new float[EMBED_DIM];
    auto h_sv = new float[EMBED_DIM];
    auto h_gamma = new float[EMBED_DIM];
    auto h_beta = new float[EMBED_DIM];

    auto h_Q_opt = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    auto h_K_opt = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    auto h_V_opt = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    auto h_Q_ref = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    auto h_K_ref = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];
    auto h_V_ref = new float[NUM_HEADS][SEQ_LEN][HEAD_DIM];

    const int PACK_INT8 = 256 / 8;   
    const int PACK_FLOAT = 256 / 32; 

    // Allocate packed hardware structures on heap
    uint256_dt* packed_X = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT]; 
    uint256_dt* packed_Wq = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Wk = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Wv = new uint256_dt[(EMBED_DIM * EMBED_DIM) / PACK_INT8];
    uint256_dt* packed_Q = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT];
    uint256_dt* packed_K = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT];
    uint256_dt* packed_V = new uint256_dt[(SEQ_LEN * EMBED_DIM) / PACK_FLOAT];

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
        }
    }
    for(int i = 0; i < EMBED_DIM; i++){
        h_bq[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_bk[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_bv[i] = -1.0f + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / 2.0f));
        h_gamma[i] = 1.0f;
        h_beta[i]   = 0.0f;
    }
    
    quantize_weight_per_channel(h_Wq_fp32, h_Wq, h_sq);
    quantize_weight_per_channel(h_Wk_fp32, h_Wk, h_sk);
    quantize_weight_per_channel(h_Wv_fp32, h_Wv, h_sv);

    for (int i = 0; i < EMBED_DIM; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            h_Wq_T[j][i] = h_Wq[i][j];
            h_Wk_T[j][i] = h_Wk[i][j];
            h_Wv_T[j][i] = h_Wv[i][j];
        }
    }

    cout << "Calculating Golden Reference..." << endl;
    linear_qkv_golden(h_X_fp32, h_gamma, h_beta, h_Wq, h_bq, h_sq, h_Wk, h_bk, h_sk, h_Wv, h_bv, h_sv, h_Q_ref, h_K_ref, h_V_ref);

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
            uint256_dt word_q = 0, word_k = 0, word_v = 0;
            for (int k = 0; k < PACK_INT8; k++) {
                word_q((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wq_T[i][j + k];
                word_k((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wk_T[i][j + k];
                word_v((k + 1) * 8 - 1, k * 8) = (ap_uint<8>)h_Wv_T[i][j + k];
            }
            packed_Wq[pack_idx] = word_q;
            packed_Wk[pack_idx] = word_k;
            packed_Wv[pack_idx] = word_v;
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
    linear_qkv_multihead(
        packed_X, packed_gamma, packed_beta,
        packed_Wk, packed_Wv, packed_Wq,
        h_bq, h_bk, h_bv, h_sq, h_sk, h_sv,
        packed_Q, packed_K, packed_V
    );

    cout << "Unpacking Results..." << endl;
    pack_idx = 0;
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j += PACK_FLOAT) {
                uint256_dt word_q = packed_Q[pack_idx];
                uint256_dt word_k = packed_K[pack_idx];
                uint256_dt word_v = packed_V[pack_idx];
                pack_idx++;

                for (int k = 0; k < PACK_FLOAT; k++) {
                    FloatInt fq, fk, fv;
                    ap_uint<32> q_bits = word_q((k + 1) * 32 - 1, k * 32);
                    ap_uint<32> k_bits = word_k((k + 1) * 32 - 1, k * 32);
                    ap_uint<32> v_bits = word_v((k + 1) * 32 - 1, k * 32);

                    fq.i = q_bits.to_uint();
                    fk.i = k_bits.to_uint();
                    fv.i = v_bits.to_uint();

                    h_Q_opt[h][i][j + k] = fq.f;
                    h_K_opt[h][i][j + k] = fk.f;
                    h_V_opt[h][i][j + k] = fv.f;
                }
            }
        }
    }

    cout << "Comparing results..." << endl;
    float max_err = 0.0f;
    int err_cnt = 0;
    const float TOLERANCE = 1.0f; 

    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                float diff_q = fabs(h_Q_ref[h][i][d] - h_Q_opt[h][i][d]);
                if (diff_q > max_err) max_err = diff_q;
                if (diff_q > TOLERANCE) {
                    printf("Mismatch at Q[%d][%d][%d]: Ref=%f, Opt=%f (diff=%f)\n", h, i, d, h_Q_ref[h][i][d], h_Q_opt[h][i][d], diff_q);
                    err_cnt++;
                }
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

    // Explicitly clean up all dynamically allocated memory to ensure zero crash during exit tracking
    delete[] h_X_fp32; delete[] h_Wq_fp32; delete[] h_Wk_fp32; delete[] h_Wv_fp32;
    delete[] h_Wq; delete[] h_Wk; delete[] h_Wv;
    delete[] h_Wq_T; delete[] h_Wk_T; delete[] h_Wv_T;
    delete[] h_bq; delete[] h_bk; delete[] h_bv;
    delete[] h_sq; delete[] h_sk; delete[] h_sv;
    delete[] h_gamma; delete[] h_beta;
    delete[] h_Q_opt; delete[] h_K_opt; delete[] h_V_opt;
    delete[] h_Q_ref; delete[] h_K_ref; delete[] h_V_ref;
    
    delete[] packed_X; delete[] packed_Wq; delete[] packed_Wk; delete[] packed_Wv;
    delete[] packed_Q; delete[] packed_K; delete[] packed_V;
    delete[] packed_gamma; delete[] packed_beta;

    return (err_cnt == 0) ? 0 : 1;
}
