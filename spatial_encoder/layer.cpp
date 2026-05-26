#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <limits> // for float min
// openmp
#include <omp.h>
#include "layer.h"

using namespace std;


int gelu_int(int x)
{
    float xf = static_cast<float>(x);
    float y = 0.5f * xf * (1.0f + tanhf(0.79788456f * (xf + 0.044715f * xf * xf * xf)));
    return static_cast<int>(roundf(y));
}


void layernorm_host(int in[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    const float eps = 1.0e-5f;
    const float output_scale = 16.0f;

    for (int i = 0; i < SEQ_LEN; i++)
    {
        float mean = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
            mean += static_cast<float>(in[i][d]);
        mean /= static_cast<float>(EMBED_DIM);

        float var = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float diff = static_cast<float>(in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(EMBED_DIM);

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float norm = (static_cast<float>(in[i][d]) - mean) * inv_std;
            out[i][d] = static_cast<int>(roundf(norm * output_scale));
        }
    }
}

void mlp_quant_host(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t W1[EMBED_DIM][MLP_DIM],
    float b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    int8_t W2[MLP_DIM][EMBED_DIM],
    float b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    vector<float> hidden(MLP_DIM);

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            float acc = b1[h];
            const float scale = scale_input[i] * scale_w1[h];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(W1[d][h]) * scale;
            }
            float gelu = 0.5f * acc * (1.0f + tanhf(0.79788456f * (acc + 0.044715f * acc * acc * acc)));
            hidden[h] = gelu;
        }

        for (int d = 0; d < EMBED_DIM; d++)
        {
            float acc = b2[d];
            const float scale = scale_w2[d];
            for (int h = 0; h < MLP_DIM; h++)
            {
                acc += hidden[h] * static_cast<float>(W2[h][d]) * scale;
            }
            out[i][d] = static_cast<int>(roundf(acc));
        }
    }
    
}

void residual_add(int a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = a[i][d] + b[i][d];
}

void residual_add(float a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<int>(roundf(a[i][d] + static_cast<float>(b[i][d])));
}

void residual_add(float a[SEQ_LEN][EMBED_DIM], float b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<int>(roundf(a[i][d] + b[i][d]));
}


void concat_attention_heads(int8_t attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
                                   int8_t attention_concat[SEQ_LEN][EMBED_DIM])
{
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < SEQ_LEN; i++)
            for (int d = 0; d < HEAD_DIM; d++)
                attention_concat[i][h * HEAD_DIM + d] = attention_heads[h][i][d];
}




void quantize_attention_heads(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN])
{

    for (int i = 0; i < SEQ_LEN; i++)
    {
        float max_abs = 0.0f;
        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float value = fabsf(attention_heads[h][i][d]);
                if (value > max_abs)
                    max_abs = value;
            }
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        attention_scales[i] = scale;

        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float scaled = attention_heads[h][i][d] / scale;
                int q = static_cast<int>(roundf(scaled));
                if (q > 127)
                    q = 127;
                else if (q < -128)
                    q = -128;
                attention_heads_q[h][i][d] = static_cast<int8_t>(q);
            }
        }
    }
}

void linear_quant_host_fp32_input(
    int in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float weight_scale = scale_weight[o] * scale_input[i];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * weight_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }
}








// opt. with openMP

void quantize_attention_heads_mp(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        float max_abs = 0.0f;

        // =========================
        // Find max abs (reduction)
        // =========================
        for (int h = 0; h < NUM_HEADS; h++)
        {
            #pragma omp simd reduction(max:max_abs)
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float value = fabsf(attention_heads[h][i][d]);
                if (value > max_abs)
                    max_abs = value;
            }
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        attention_scales[i] = scale;

        // =========================
        // Quantization
        // =========================
        for (int h = 0; h < NUM_HEADS; h++)
        {
            #pragma omp simd
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float scaled = attention_heads[h][i][d] / scale;
                int q = static_cast<int>(roundf(scaled));

                if (q > 127)
                    q = 127;
                else if (q < -128)
                    q = -128;

                attention_heads_q[h][i][d] = static_cast<int8_t>(q);
            }
        }
    }
}


void concat_attention_heads_mp(int8_t attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
                                   int8_t attention_concat[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < SEQ_LEN; i++)
            memcpy(&attention_concat[i][h * HEAD_DIM], &attention_heads[h][i][0], sizeof(int) * HEAD_DIM);
            // HLS SLM error may caused by here
            // original error code: memcpy(attention_concat + i * EMBED_DIM + h * HEAD_DIM, attention_heads + h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM, sizeof(int) * HEAD_DIM);    
            //for (int d = 0; d < HEAD_DIM; d++)          
            //attention_concat[i][h * HEAD_DIM + d] = attention_heads[h][i][d];
}


void residual_add_mp(
    const int a[SEQ_LEN][EMBED_DIM],
    const int b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            out[i][d] = a[i][d] + b[i][d];
        }
    }
}

void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const int   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float sum = a[i][d] +  static_cast<float>(b[i][d]);
            out[i][d] = static_cast<int>(roundf(sum));
        }
    }
}

void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const float   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float sum = a[i][d] +  static_cast<float>(b[i][d]);
            out[i][d] = static_cast<int>(roundf(sum));
        }
    }
}

void linear_quant_host_fp32_input_mp(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float weight_scale = scale_weight[o] * scale_input[i];

            #pragma omp simd reduction(+:acc)
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += (float)in[i][d] *
                       (float)weight[d][o] *
                       weight_scale;
            }

            out[i][o] = (int)roundf(acc);
        }
    }
}

void mlp_quant_host_mp(
    const int8_t in[SEQ_LEN][EMBED_DIM],
    const float scale_input[SEQ_LEN],
    const int8_t W1[EMBED_DIM][MLP_DIM],
    const float b1[MLP_DIM],
    const float scale_w1[MLP_DIM],
    const int8_t W2[MLP_DIM][EMBED_DIM],
    const float b2[EMBED_DIM],
    const float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel
    {
        std::vector<float> hidden(MLP_DIM);

        #pragma omp for
        for (int i = 0; i < SEQ_LEN; i++)
        {
            // Layer 1
            for (int h = 0; h < MLP_DIM; h++)
            {
                float acc = b1[h];
                const float scale = scale_input[i] * scale_w1[h];

                #pragma omp simd reduction(+:acc)
                for (int d = 0; d < EMBED_DIM; d++)
                {
                    acc += static_cast<float>(in[i][d]) *
                        static_cast<float>(W1[d][h]) *
                        scale;
                }

                float x = acc;
                hidden[h] = 0.5f * x *
                    (1.0f + tanhf(0.79788456f *
                    (x + 0.044715f * x * x * x)));
            }

            // Layer 2
            for (int d = 0; d < EMBED_DIM; d++)
            {
                float acc = b2[d];
                const float scale = scale_w2[d];

                #pragma omp simd reduction(+:acc)
                for (int h = 0; h < MLP_DIM; h++)
                {
                    acc += hidden[h] *
                        static_cast<float>(W2[h][d]) *
                        scale;
                }

                out[i][d] = static_cast<int>(roundf(acc));
            }
        }
    }   
}


/* ===================== temporal encoder ==========================
spatial encoder's layernorm
posembedding
>temporal encoder * 12
    layernorm
    attention_layer: quant->QKV->dequant->SoftMax->Quant->Out Projection & Concat->dequant->Residual
    layernorm
    MLP: quant->matmul->dequant->act. func->quant->matmul->dequant->act. func->residual
temportal encoder's layernorm
 =================================================================== */

/*
 void temporal_layernorm(
    float in[TEMP_LEN][EMBED_DIM],
    float kernel[EMBED_DIM],
    float bias[EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM],
    float out_scale[TEMP_LEN],
){
    float eps = 0.000001;
    for (int i = 0; i < TEMP_LEN; i++){
        // compute layer mean & variance
        float l_mean = 0;
        float l_max = 0;
        double l_var = 0;
        for (int j = 0; j < EMBED_DIM; j++){
            l_mean += in[i][j];            
        }
        l_mean /= EMBED_DIM;
        for (int j = 0; j < EMBED_DIM; j++){
            l_var += (in[i][j] - l_mean) * (in[i][j] - l_mean);            
        }
        l_var /= EMBED_DIM;


        for (int j = 0; j < EMBED_DIM; j++){
            double temp = (in[i][j] - l_mean) / std::sqrt(l_var + eps);
            out[i][j] = (float)(temp * kernel[j] + bias[j]);
            if (std::fabsf(out[i][j]) > l_max) l_max = std::fabsf(out[i][j]);
        }
        out_scale[i] = (l_max > 0.0f) ? (l_max / 127.0f) : 1.0f; 
    }
}

void matmul(const int A[], const int B[], int C[], const int M, const int N, const int K, const int tile_size){
    for (int i = 0; i < M; i+=tile_size){
        for (int j = 0; j < N; j+=tile_size){
            for (int k = 0; k < K; k+=tile_size){
                for (int tile_i = i; tile_i < i + tile_size; tile_i++){
                    for (int tile_j = j; tile_j < j + tile_size; tile_j++){
                        for (int tile_k = k; tile_k < k + tile_size; tile_k++){
                            // C[i][k] = A[i][j] * B[j][k]
                            *(C + tile_i * K + tile_k) = *(A + tile_i * N + tile_j) * *(B + tile_j * K + tile_k);
                        }
                    }
                }
            }
        }
    }
}

void transpose(const int in[], int out[], int M, int N){
    // out[N][M] = in[M][N]
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            *(out + j * M + i) = *(in + i * N + j);
        }
    }
}

void softmax(){

}

void temporal_attention(
    float in[TEMP_LEN][EMBED_DIM],
    float in_scale[TEMP_LEN],
    int   Wq[EMBED_DIM][EMBED_DIM],
    int   Wk[EMBED_DIM][EMBED_DIM],
    int   Wv[EMBED_DIM][EMBED_DIM],
    int   Wo[EMBED_DIM][EMBED_DIM],
    float w_scale[EMBED_DIM],
    float q_scale[EMBED_DIM],
    float v_scale[EMBED_DIM],
    float o_scale[EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM]
){
    // quant active
    int in_quant[TEMP_LEN][EMBED_DIM];
    for (int i = 0; i < TEMP_LEN; i++){
        for (int j = 0; j < EMBED_DIM; j++){
            in_quant[i][j] = (int)roundf(in[i][j] / in_scale[i]);
        }
    }

    // get Q, K, V
    int Q[TEMP_LEN][EMBED_DIM];
    int K[TEMP_LEN][EMBED_DIM];
    int V[TEMP_LEN][EMBED_DIM];
    int KT[TEMP_LEN][EMBED_DIM];
    matmul(in_quant, Wq, Q);
    matmul(in_quant, Wk, K);
    matmul(in_quant, Wv, V);
    transpose(K, KT, TEMP_LEN, EMBED_DIM);
    
    
    //...

}


void temporal_residual(
    float in1[TEMP_LEN][EMBED_DIM],
    float in2[TEMP_LEN][EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM]
){
    temporal_posEmbd(in1, in2, out);
    return ;
}

void temporal_posEmbd(
    float in[TEMP_LEN][EMBED_DIM],
    float pos_embd[TEMP_LEN][EMBED_DIM],
    float out[TEMP_LEN][EMBED_DIM],
){

    const int tile_size = 64;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < TEMP_LEN; i++){
        for (int j = 0; j < EMBED_DIM; j += tile_size){
            int k_bound = j + tile_size;
            
            //#pragma omp for
            for (int k = j; k < k_bound; k++){
                out[i][k] = in[i][k] + pos_embd[i][k];
            }
        }
    }
    return ;
}
*/