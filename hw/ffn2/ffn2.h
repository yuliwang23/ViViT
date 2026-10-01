#ifndef FFN2_H
#define FFN2_H

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

typedef ap_uint<512> uint512_dt;
typedef ap_uint<256> uint256_dt;
typedef ap_uint<64> int64_dt;
typedef ap_int<32> int32_dt;
typedef ap_int<8> int8_dt;

static constexpr int SEQ_LEN =
    197;

static constexpr int EMBED_DIM =
    768;

static constexpr int MLP_DIM =
    3072;

static constexpr int INT8_PACK =
    8;

static constexpr int FP32_PACK =
    8;

static constexpr int MLP_DIM_package =
    MLP_DIM /
    INT8_PACK;

static constexpr int TILE_OUT_DIM =
    64;

static constexpr int TILE_SIZE =
    64;

static constexpr int MLP1_NUM_TILES =
    MLP_DIM /
    TILE_OUT_DIM;

static constexpr int MLP2_NUM_TILES =
    EMBED_DIM /
    TILE_SIZE;

static constexpr float EPSILON =
    1.0e-5f;

static_assert(
    INT8_PACK == 8,
    "INT8_PACK must be 8"
);

static_assert(
    FP32_PACK == 8,
    "FP32_PACK must be 8"
);

static_assert(
    MLP_DIM % INT8_PACK == 0,
    "MLP_DIM must be divisible by INT8_PACK"
);

static_assert(
    MLP_DIM % 16 == 0,
    "MLP_DIM must be divisible by 16"
);

static_assert(
    MLP_DIM % 64 == 0,
    "MLP_DIM must be divisible by 64"
);

static_assert(
    EMBED_DIM % 16 == 0,
    "EMBED_DIM must be divisible by 16"
);

static_assert(
    EMBED_DIM % 64 == 0,
    "EMBED_DIM must be divisible by 64"
);

static_assert(
    MLP_DIM_package % 16 == 0,
    "MLP_DIM_package must be divisible by 16"
);

static_assert(
    MLP_DIM % TILE_OUT_DIM == 0,
    "MLP_DIM must be divisible by TILE_OUT_DIM"
);

static_assert(
    EMBED_DIM % TILE_SIZE == 0,
    "EMBED_DIM must be divisible by TILE_SIZE"
);

static_assert(
    TILE_OUT_DIM % INT8_PACK == 0,
    "TILE_OUT_DIM must be divisible by INT8_PACK"
);

static_assert(
    TILE_SIZE % FP32_PACK == 0,
    "TILE_SIZE must be divisible by FP32_PACK"
);

static_assert(
    TILE_SIZE % 16 == 0,
    "TILE_SIZE must be divisible by 16"
);

static_assert(
    MLP1_NUM_TILES ==
        MLP_DIM / TILE_OUT_DIM,
    "Invalid MLP1_NUM_TILES"
);

static_assert(
    MLP2_NUM_TILES ==
        EMBED_DIM / TILE_SIZE,
    "Invalid MLP2_NUM_TILES"
);

extern "C" {

void FFN2(
    uint512_dt *HBM_FP32_Intermediate,
    float *HBM_GELU_Max,
    uint512_dt *W2,
    uint512_dt *x,
    float *b2,
    float *scale_w2,
    uint512_dt *out
);

}

#endif
