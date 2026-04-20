#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <array>
#include <cfloat>

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

constexpr int NUM_ITERATIONS = 48;

constexpr MeanWeightLUT<NUM_ITERATIONS> MEAN_WT;
constexpr M2WeightLUT<NUM_ITERATIONS>   M2_WT;


constexpr int SEQ_LEN = 197; // 14*14*16 --> per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr float EPSILON = 1e-5;
constexpr float EMBED_DIM_INV = 1.0f / EMBED_DIM;
typedef ap_uint<256> uint256_dt;
typedef ap_int<8> int8_dt;

void layernorm(
    float x[SEQ_LEN][EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    int8_dt out[SEQ_LEN][EMBED_DIM]
);

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

void load_X_FIFO2BRAM(
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