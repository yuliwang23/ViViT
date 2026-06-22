//==============================================================
// File: tensor_ops.cpp
// Description: Core tensor operations
//==============================================================

#include "tensor_ops.h"
// #include <math.h>

//==============================================================
// Matrix Multiplication
//==============================================================
// C[M][N] = A[M][K] * B[K][N]
void matmul(float A[M_MAX][K_MAX],
            float B[K_MAX][N_MAX],
            float C[M_MAX][N_MAX],
            int M, int K, int N)
{
    // 3-level loop for matmul
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
}

//==============================================================
// Elementwise Add
//==============================================================
void tensor_add(float A[M_MAX][N_MAX],
                float B[M_MAX][N_MAX],
                float C[M_MAX][N_MAX],
                int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

float fast_exp(float x)
{
    float y = x * 1.44269504f;
    int I = (int)y;
    float F = y - (float)I;

    // 2^F ªº½u©Êªñ¦ü (1 + F)
    float approx_2_F = 1.0f + F;

    return ldexpf(approx_2_F, I);
}

//==============================================================
// Softmax
//==============================================================

void softmax(float in[VEC_MAX],
             float out[VEC_MAX],
             int length)
{
    using std::exp; // TODO: change into hls::exp

    float max_val = -1e38f;
    float sum_exp = 0.0f;

    // Online calculate Max and Sum
    for (int i = 0; i < length; i++)
    {
        float x = in[i];

        if (x > max_val)
        {
            // sum_new = sum_old *
            // exp(max_old - max_new) + exp(x - max_new)
            float diff = max_val - x;
            sum_exp = sum_exp * exp(diff) + 1.0f;
            max_val = x;
        }
        else
        {
            // sum_new = sum_old + exp(x - max_old)
            float diff = x - max_val;
            sum_exp += exp(diff);
        }
    }

    // Normalization
    for (int i = 0; i < length; i++)
    {
        float diff = in[i] - max_val;
        out[i] = exp(diff) / sum_exp;
    }
}

// void softmax(float in[VEC_MAX],
//              float out[VEC_MAX],
//              int length)
// {
//     float max_val = in[0];
//     for (int i = 1; i < length; i++)
//     {
//         if (in[i] > max_val)
//             max_val = in[i];
//     }

//     float sum = 0;
//     float exp_buffer[VEC_MAX];

//     for (int i = 0; i < length; i++)
//     {
//         exp_buffer[i] = expf(in[i] - max_val); // LUT
//         sum += exp_buffer[i];
//     }
//     float inv_sum = 1.0f / sum;
//     for (int i = 0; i < length; i++)
//     {
//         // out[i] = exp_buffer[i] / sum;
//         out[i] = exp_buffer[i] * inv_sum;
//     }
// }

//==============================================================
// GELU Activation
//==============================================================
// TODO: sigmoid use LUT
float sigmoid_pwl(float y)
{
    if (y < -2.0f)
        return 0.0f;
    if (y > 2.0f)
        return 1.0f;
    return 0.25f * y + 0.5f;
}

float gelu(float x)
{
    // return 0.5f * x
    //* (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x ^ 3)));
    float y = 1.702f * x;
    float sig = sigmoid_pwl(y);
    return x * sig;
}

void tensor_gelu(float in[M_MAX][N_MAX],
                 float out[M_MAX][N_MAX],
                 int M, int N)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            out[i][j] = gelu(in[i][j]);
        }
    }
}

//==============================================================
// Layer Normalization
//==============================================================
// out[i][j] = gamma[j] * ((x[i][j] - mean) / sqrt(var + eps)) + beta[j]
void layernorm(float x[M_MAX][N_MAX],
               float gamma[N_MAX],
               float beta[N_MAX],
               float out[M_MAX][N_MAX],
               int M, int N,
               float eps)
{

    for (int i = 0; i < M; i++)
    {
        float mean = 0.0f;
        float var = 0.0f;

        // Mean
        for (int j = 0; j < N; j++)
            mean += x[i][j];
        mean /= N;

        // Variance
        for (int j = 0; j < N; j++)
        {
            float diff = x[i][j] - mean;
            var += diff * diff;
        }
        var /= N;
        float inv_std = 1.0f / sqrtf(var + eps);

        // Normalize
        for (int j = 0; j < N; j++)
        {
            out[i][j] = gamma[j] * ((x[i][j] - mean) * inv_std) + beta[j];
        }
    }
}