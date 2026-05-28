#include "mlp.h"
#include "tensor_ops.h"

//==============================================================
// MLP Forward
// Structure:
//   hidden = GELU( input * W1 + B1 )
//   output =        hidden * W2 + B2
//==============================================================
void mlp_forward(
    float input[M_MAX][K_MAX],  // [M, IN_DIM]
    float w1[K_MAX][N_MAX],     // [IN_DIM, HIDDEN_DIM]
    float b1[M_MAX][N_MAX],     // [M, HIDDEN_DIM] (broadcasted)
    float w2[N_MAX][K_MAX],     // [HIDDEN_DIM, OUT_DIM]
    float b2[M_MAX][K_MAX],     // [M, OUT_DIM] (broadcasted)
    float output[M_MAX][K_MAX], // [M, OUT_DIM]
    int M,                      // seq_len
    int IN_DIM,
    int HIDDEN_DIM,
    int OUT_DIM)
{
    // Intermediate buffers
    static float hidden[M_MAX][N_MAX];
    static float tmp[M_MAX][N_MAX];

    // Linear 1: hidden = input * w1
    matmul(input, w1, hidden, M, IN_DIM, HIDDEN_DIM);

    // Add bias: hidden = hidden + b1
    tensor_add(hidden, b1, hidden, M, HIDDEN_DIM);

    // GELU activation (PWL version from tensor_ops)
    tensor_gelu(hidden, hidden, M, HIDDEN_DIM);

    // Linear 2: output = hidden * w2
    matmul(hidden, w2, tmp, M, HIDDEN_DIM, OUT_DIM);

    // Add bias: output = tmp + b2
    tensor_add(tmp, b2, output, M, OUT_DIM);
}
