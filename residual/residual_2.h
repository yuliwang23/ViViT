#ifndef RESIDUAL_H
#define RESIDUAL_H
#include <stdint.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include "../vivit_config.h"

// ============================================================
// Design parameters
// ============================================================
constexpr int SEQ_LEN   = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int HEAD_DIM  = VIVIT_HEAD_DIM;
constexpr int TILE_SIZE = VIVIT_LINEAR_TILE_SIZE;

typedef ap_uint<256> bus_t; // 256 / 32 = 8 floats


extern "C" {
void residual(
    bus_t* A,
    bus_t* B,
    bus_t* C);
}
#endif

