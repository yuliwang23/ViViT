// ============================================================
// residual.cpp
// ============================================================

#include "residual.h"

extern "C" {

void residual(
    bus_dt A[SEQ_LEN][PER_ROW],
    bus_dt B[SEQ_LEN][PER_ROW],
    bus_dt C[SEQ_LEN][PER_ROW])
{
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 depth=VIVIT_EMBED_FP_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1 depth=VIVIT_EMBED_FP_PACKED_DEPTH
#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2 depth=VIVIT_EMBED_FP_PACKED_DEPTH

#pragma HLS INTERFACE s_axilite port=A bundle=control
#pragma HLS INTERFACE s_axilite port=B bundle=control
#pragma HLS INTERFACE s_axilite port=C bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // =====================================================
    // local BRAM
    // =====================================================

    bus_dt BRAM_A[PER_ROW];
    bus_dt BRAM_B[PER_ROW];
    bus_dt BRAM_C[PER_ROW];

#pragma HLS BIND_STORAGE variable=BRAM_A type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_B type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_C type=ram_2p impl=bram

    // =====================================================
    // token loop
    // =====================================================

    for (int token = 0; token < SEQ_LEN; token++)
    {
#pragma HLS LOOP_TRIPCOUNT min=VIVIT_SEQ_LEN max=VIVIT_SEQ_LEN

        // =================================================
        // load from HBM
        // =================================================

        for (int j = 0; j < PER_ROW; j++)
        {
#pragma HLS PIPELINE II=1

            BRAM_A[j] = A[token][j];
            BRAM_B[j] = B[token][j];
        }

        // =================================================
        // vector residual compute
        // =================================================

        for (int j = 0; j < PER_ROW; j++)
        {
#pragma HLS PIPELINE II=1

            compute_residual(
                BRAM_A[j],
                BRAM_B[j],
                BRAM_C[j]);
        }

        // =================================================
        // write back
        // =================================================

        for (int j = 0; j < PER_ROW; j++)
        {
#pragma HLS PIPELINE II=1

            C[token][j] = BRAM_C[j];
        }
    }
}

} // extern "C"

// ============================================================
// 256-bit vector residual compute
// ============================================================

void compute_residual(
    bus_dt bram_a,
    bus_dt bram_b,
    bus_dt &bram_c)
{
#pragma HLS INLINE

    bram_c = 0;

    for (int i = 0; i < VEC_SIZE; i++)
    {
#pragma HLS UNROLL

        fp_conv a;
        fp_conv b;
        fp_conv c;

        // unpack fp32

        a.u = bram_a.range(i * 32 + 31, i * 32);
        b.u = bram_b.range(i * 32 + 31, i * 32);

        // fp add

        c.f = a.f + b.f;

        // repack fp32

        bram_c.range(i * 32 + 31, i * 32) = c.u;
    }
}
