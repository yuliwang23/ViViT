#ifndef RESIDUAL_H
#define RESIDUAL_H
#include <stdint.h>
#include "hls_stream.h"
#include "ap_int.h"
#include "ap_fixed.h"

// ============================================================
// Design parameters
// ============================================================
constexpr int SEQ_LEN   = 14 * 14 + 1;          // 197
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM  = EMBED_DIM / NUM_HEADS; // 64
constexpr int TILE_SIZE = 64;                    // == HEAD_DIM

typedef ap_uint<256> bus_t; // 256 / 32 = 8 floats


extern "C" {
void residual(
    bus_t* A,
    bus_t* B,
    bus_t* C);
}
#endif


