#ifndef MLP_H
#define MLP_H

#include "../vivit_config.h"

// Max dimensions (compile-time constants for HLS)
#define SEQ_MAX VIVIT_SEQ_LEN
#define IN_DIM_MAX VIVIT_EMBED_DIM
#define HIDDEN_DIM_MAX VIVIT_MLP_DIM
#define OUT_DIM_MAX VIVIT_EMBED_DIM

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
