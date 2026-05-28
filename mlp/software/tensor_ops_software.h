#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#define M_MAX 256
#define N_MAX 256
#define K_MAX 256
#define VEC_MAX 256

#include <math.h>

// Matrix multiplication
void matmul(float A[M_MAX][K_MAX],
            float B[K_MAX][N_MAX],
            float C[M_MAX][N_MAX],
            int M, int K, int N);

// Elementwise add
void tensor_add(float A[M_MAX][N_MAX],
                float B[M_MAX][N_MAX],
                float C[M_MAX][N_MAX],
                int M, int N);

// Softmax
void softmax(float in[VEC_MAX],
             float out[VEC_MAX],
             int length);

// GELU
float gelu(float x);
void tensor_gelu(float in[M_MAX][N_MAX],
                 float out[M_MAX][N_MAX],
                 int M, int N);

// LayerNorm
void layernorm(float x[M_MAX][N_MAX],
               float gamma[N_MAX],
               float beta[N_MAX],
               float out[M_MAX][N_MAX],
               int M, int N,
               float eps);

#endif