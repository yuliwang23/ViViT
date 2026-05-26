/*
// LUT for 1/k where k = 1 to N 
//(nB/nAB)
template <int N>
struct MeanWeightLUT {
    float data[N];
    
    // constexpr constructor guarantees execution during C++ compilation
    constexpr MeanWeightLUT() : data() {
        for (int i = 0; i < N; ++i) {
            // k represents the current iteration step (k = 1 to N)
            float k = static_cast<float>(i + 1);
            data[i] = 1.0f / k;
        }
    }
};

// LUT for 16*(k-1)/k where k = 1 to N 
//(nA*nB/nAB)
template <int N>
struct M2WeightLUT {
    float data[N];
    
    // constexpr constructor guarantees execution during C++ compilation
    constexpr M2WeightLUT() : data() {
        for (int i = 0; i < N; ++i) {
            // k represents the current iteration step (k = 1 to N)
            float k = static_cast<float>(i + 1);
            // Numerator is 8 * (k - 1), which equals 8 * i
            float numerator = 16.0f * static_cast<float>(i);
            data[i] = numerator / k;
        }
    }
};

//constexpr MeanWeightLUT<NUM_ITERATIONS> MEAN_WT;
//constexpr M2WeightLUT<NUM_ITERATIONS>   M2_WT;
*/

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <array>
#include <cfloat>
#include "../vivit_config.h"

constexpr int   SEQ_LEN       = VIVIT_SEQ_LEN;
constexpr int   EMBED_DIM     = VIVIT_EMBED_DIM;
constexpr int   NUM_ITERATIONS = VIVIT_LAYER_NORM_ITERATIONS;
constexpr float EPSILON       = 1e-5f;
constexpr float EMBED_DIM_INV = 1.0f / static_cast<float>(EMBED_DIM);

typedef ap_uint<256> uint256_dt;
typedef ap_int<8>    int8_dt;

// 改成 static const float array，Vitis parser 完全相容
// MEAN_WT[i] = 1/(i+1)
static const float MEAN_WT[NUM_ITERATIONS] = {
#if VIVIT_SMALL_CONFIG
    1.0f/1,  1.0f/2,  1.0f/3,  1.0f/4
#else
    1.0f/1,  1.0f/2,  1.0f/3,  1.0f/4,  1.0f/5,  1.0f/6,
    1.0f/7,  1.0f/8,  1.0f/9,  1.0f/10, 1.0f/11, 1.0f/12,
    1.0f/13, 1.0f/14, 1.0f/15, 1.0f/16, 1.0f/17, 1.0f/18,
    1.0f/19, 1.0f/20, 1.0f/21, 1.0f/22, 1.0f/23, 1.0f/24,
    1.0f/25, 1.0f/26, 1.0f/27, 1.0f/28, 1.0f/29, 1.0f/30,
    1.0f/31, 1.0f/32, 1.0f/33, 1.0f/34, 1.0f/35, 1.0f/36,
    1.0f/37, 1.0f/38, 1.0f/39, 1.0f/40, 1.0f/41, 1.0f/42,
    1.0f/43, 1.0f/44, 1.0f/45, 1.0f/46, 1.0f/47, 1.0f/48
#endif
};

// M2_WT[i] = 16*i/(i+1)
static const float M2_WT[NUM_ITERATIONS] = {
#if VIVIT_SMALL_CONFIG
    0.0f,
    16.0f*1/2,  16.0f*2/3,  16.0f*3/4
#else
    0.0f,
    16.0f*1/2,  16.0f*2/3,  16.0f*3/4,  16.0f*4/5,  16.0f*5/6,
    16.0f*6/7,  16.0f*7/8,  16.0f*8/9,  16.0f*9/10, 16.0f*10/11,
    16.0f*11/12,16.0f*12/13,16.0f*13/14,16.0f*14/15,16.0f*15/16,
    16.0f*16/17,16.0f*17/18,16.0f*18/19,16.0f*19/20,16.0f*20/21,
    16.0f*21/22,16.0f*22/23,16.0f*23/24,16.0f*24/25,16.0f*25/26,
    16.0f*26/27,16.0f*27/28,16.0f*28/29,16.0f*29/30,16.0f*30/31,
    16.0f*31/32,16.0f*32/33,16.0f*33/34,16.0f*34/35,16.0f*35/36,
    16.0f*36/37,16.0f*37/38,16.0f*38/39,16.0f*39/40,16.0f*40/41,
    16.0f*41/42,16.0f*42/43,16.0f*43/44,16.0f*44/45,16.0f*45/46,
    16.0f*46/47,16.0f*47/48
#endif
};


extern "C" {
void layernorm(
    float x[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    int8_dt out[SEQ_LEN][EMBED_DIM],
    float scaler[SEQ_LEN]
);
}

void load_weights_HBM2SRAM(
    float HBM_gamma[EMBED_DIM],
    float HBM_beta[EMBED_DIM],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
);

void load_X_HBM2FIFO(
    float X[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X
);

inline void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM]
);

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    float *scaler,
    int8_dt out[EMBED_DIM]
);

void store_out_BRAM2FIFO(
    int8_dt out_BRAM[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_out
);

void store_out_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_out,
    int8_dt Output_Global[EMBED_DIM]
);

void load_X_HBM2FIFO_all(
    float x[SEQ_LEN][EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X
);

void compute_all(
    hls::stream<uint256_dt> &FIFO_X,
    float scaler[SEQ_LEN],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_out
);

void store_out_FIFO2HBM_all(
    hls::stream<uint256_dt> &FIFO_out,
    int8_dt out[SEQ_LEN][EMBED_DIM]
);
