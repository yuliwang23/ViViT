#include "residual.h"

constexpr int VEC_SIZE = 8;                  // 256 bits / 32 bits(float)
constexpr int WORD_PER_ROW = EMBED_DIM / 8; // 768 / 8 = 96


typedef union {
    uint32_t u;
    float    f;
} fp_conv;

extern "C" {

void residual(
    bus_t* A,
    bus_t* B,
    bus_t* C)
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 max_widen_bitwidth=256 depth=VIVIT_EMBED_FP_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 max_widen_bitwidth=256 depth=VIVIT_EMBED_FP_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 max_widen_bitwidth=256 depth=VIVIT_EMBED_FP_PACKED_DEPTH

#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // total words:
    // 197 * 96 = 18912

    const int TOTAL_WORDS = SEQ_LEN * WORD_PER_ROW;

MAIN_LOOP:
    for (int idx = 0; idx < TOTAL_WORDS; idx++)
    {
#pragma HLS PIPELINE II=1

        //--------------------------------------------------
        // AXI read
        //--------------------------------------------------

        bus_t wordA = A[idx]; // 8 floats
        bus_t wordB = B[idx];

        //--------------------------------------------------
        // local SIMD registers
        //--------------------------------------------------

        float localA[VEC_SIZE];
        float localB[VEC_SIZE];
        float localC[VEC_SIZE];

#pragma HLS ARRAY_PARTITION variable=localA complete
#pragma HLS ARRAY_PARTITION variable=localB complete
#pragma HLS ARRAY_PARTITION variable=localC complete

        //--------------------------------------------------
        // unpack 256-bit -> 8 floats
        //--------------------------------------------------

    UNPACK_LOOP:
        for (int k = 0; k < VEC_SIZE; k++)
        {
#pragma HLS UNROLL

            fp_conv tmpA;
            fp_conv tmpB;

            tmpA.u = wordA.range(32 * k + 31, 32 * k).to_uint();
            tmpB.u = wordB.range(32 * k + 31, 32 * k).to_uint();

            localA[k] = tmpA.f;
            localB[k] = tmpB.f;
        }

        //--------------------------------------------------
        // SIMD FP add
        //--------------------------------------------------

    COMPUTE_LOOP:
        for (int k = 0; k < VEC_SIZE; k++) // 8
        {
#pragma HLS UNROLL

            localC[k] = localA[k] + localB[k];
        }

        //--------------------------------------------------
        // pack 8 floats -> 256-bit
        //--------------------------------------------------

        bus_t outWord;

    PACK_LOOP:
        for (int k = 0; k < VEC_SIZE; k++)
        {
#pragma HLS UNROLL

            fp_conv tmpC;
            tmpC.f = localC[k];

            outWord.range(32 * k + 31, 32 * k) = tmpC.u;
        }

        //--------------------------------------------------
        // AXI write
        //--------------------------------------------------

        C[idx] = outWord;
    }
}

}
