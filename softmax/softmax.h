#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
// #include <ap_int.h>
// #include <hls_stream.h>
#include <cmath>

constexpr int SEQ_LEN = 197; // num_tokens
constexpr int EMBED_DIM = 768;    // hidden_size
constexpr int MLP_DIM = 3072;     // mlp_dim
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;

// For attention softmax / vector ops
constexpr int VEC_MAX = SEQ_LEN;

