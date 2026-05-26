#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include "../vivit_config.h"

// Constants for Vision Transformer (ViT)
constexpr int SEQ_LEN = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int MLP_DIM = VIVIT_MLP_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int HEAD_DIM = VIVIT_HEAD_DIM;
constexpr int VEC_MAX = SEQ_LEN;

// Data type for 256-bit memory bus
typedef ap_uint<256> uint256_dt;

// Union for safe bit-casting between float and uint32_t
typedef union {
    float f;
    uint32_t i;
} float_cast_t;

// Important: Use extern "C" in header to match the .cpp implementation
#ifdef __cplusplus
extern "C" {
#endif

void softmax(uint256_dt* in, uint256_dt* out);

#ifdef __cplusplus
}
#endif

// Sub-function declarations (C++ linkage is fine here)
void load_HBM2BRAM(uint256_dt* in, float BRAM_input[VEC_MAX]);
void calculate_max_and_sum(float BRAM_input[VEC_MAX], float &max_val, float &sum_exp);
void normalize_BRAM2HBM(float BRAM_input[VEC_MAX], float max_val, float sum_exp, uint256_dt* out);

#endif // SOFTMAX_H
