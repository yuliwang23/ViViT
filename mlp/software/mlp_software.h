#ifndef MLP_H
#define MLP_H

// Max dimensions (compile-time constants for HLS)
#define SEQ_MAX 64
#define IN_DIM_MAX 768
#define HIDDEN_DIM_MAX 3072
#define OUT_DIM_MAX 768

void mlp_forward(
    float input[SEQ_MAX][IN_DIM_MAX],
    float w1[IN_DIM_MAX][HIDDEN_DIM_MAX],
    float b1[HIDDEN_DIM_MAX],
    float w2[HIDDEN_DIM_MAX][OUT_DIM_MAX],
    float b2[OUT_DIM_MAX],
    float output[SEQ_MAX][OUT_DIM_MAX],
    int seq_len,
    int in_dim,
    int hidden_dim,
    int out_dim);

#endif
