// ============================================================
// residual.h
// ============================================================

#ifndef RESIDUAL_H
#define RESIDUAL_H

#include <ap_int.h>
#include <stdint.h>
#include "../vivit_config.h"

constexpr int SEQ_LEN   = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;

constexpr int VEC_SIZE = 8;              // 256 / 32
constexpr int PER_ROW  = EMBED_DIM / VEC_SIZE;

typedef ap_uint<256> bus_dt;

typedef union {
    uint32_t u;
    float    f;
} fp_conv;

// packed interface version

extern "C" {

void residual(
    bus_dt A[SEQ_LEN][PER_ROW],
    bus_dt B[SEQ_LEN][PER_ROW],
    bus_dt C[SEQ_LEN][PER_ROW]);

}

// helper functions

void compute_residual(
    bus_dt bram_a,
    bus_dt bram_b,
    bus_dt &bram_c);

#endif
