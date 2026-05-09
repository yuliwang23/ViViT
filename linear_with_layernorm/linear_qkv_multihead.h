#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <cmath>
#include <array>
#include <cfloat>


constexpr int SEQ_LEN = 197; // 14*14*16 --> per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr int EMBED_DIM_package = 96;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int tile_size_dim = 64;
constexpr int TILE_SIZE = 64;
constexpr int tile_size_sql = 64;
constexpr int tile_sql_time = SEQ_LEN / tile_size_sql;
constexpr int tile_sql_remain = SEQ_LEN % tile_size_sql;
constexpr int DSP_PER_CYCLE = 16;

constexpr int   NUM_ITERATIONS = 48;
constexpr float EPSILON       = 1e-5f;
constexpr float EMBED_DIM_INV = 1.0f / static_cast<float>(EMBED_DIM);

typedef ap_uint<256> uint256_dt;
typedef ap_int<8>    int8_dt;
typedef ap_int<16>   int16_dt;
typedef ap_int<26>   int26_dt;
typedef ap_int<64>   int64_dt;
 

// MEAN_WT[i] = 1/(i+1)
static const float MEAN_WT[NUM_ITERATIONS] = {
    1.0f/1,  1.0f/2,  1.0f/3,  1.0f/4,  1.0f/5,  1.0f/6,
    1.0f/7,  1.0f/8,  1.0f/9,  1.0f/10, 1.0f/11, 1.0f/12,
    1.0f/13, 1.0f/14, 1.0f/15, 1.0f/16, 1.0f/17, 1.0f/18,
    1.0f/19, 1.0f/20, 1.0f/21, 1.0f/22, 1.0f/23, 1.0f/24,
    1.0f/25, 1.0f/26, 1.0f/27, 1.0f/28, 1.0f/29, 1.0f/30,
    1.0f/31, 1.0f/32, 1.0f/33, 1.0f/34, 1.0f/35, 1.0f/36,
    1.0f/37, 1.0f/38, 1.0f/39, 1.0f/40, 1.0f/41, 1.0f/42,
    1.0f/43, 1.0f/44, 1.0f/45, 1.0f/46, 1.0f/47, 1.0f/48
};

// M2_WT[i] = 16*i/(i+1)
static const float M2_WT[NUM_ITERATIONS] = {
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
};