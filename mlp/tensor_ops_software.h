#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H

#include "../vivit_config.h"

#define M_MAX VIVIT_MLP_DIM
#define N_MAX VIVIT_MLP_DIM
#define K_MAX VIVIT_MLP_DIM
#define VEC_MAX VIVIT_MLP_DIM

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
