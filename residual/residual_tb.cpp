// ============================================================
// residual_tb.cpp
// ============================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "residual.h"

#define MAX_ERR 1e-5

// ============================================================
// golden reference
// ============================================================

void residual_golden(
    float A[SEQ_LEN][EMBED_DIM],
    float B[SEQ_LEN][EMBED_DIM],
    float C[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++) {

        for (int j = 0; j < EMBED_DIM; j++) {

            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// ============================================================
// pack float tensor -> 256-bit tensor
// ============================================================

void pack_tensor(
    float in[SEQ_LEN][EMBED_DIM],
    bus_dt out[SEQ_LEN][PER_ROW])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < PER_ROW; j++)
        {
            bus_dt packed = 0;

            for (int k = 0; k < VEC_SIZE; k++)
            {
                fp_conv tmp;

                tmp.f = in[i][j * VEC_SIZE + k];

                packed.range(
                    k * 32 + 31,
                    k * 32) = tmp.u;
            }

            out[i][j] = packed;
        }
    }
}

// ============================================================
// unpack 256-bit tensor -> float tensor
// ============================================================

void unpack_tensor(
    bus_dt in[SEQ_LEN][PER_ROW],
    float out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < PER_ROW; j++)
        {
            bus_dt packed = in[i][j];

            for (int k = 0; k < VEC_SIZE; k++)
            {
                fp_conv tmp;

                tmp.u = packed.range(
                    k * 32 + 31,
                    k * 32);

                out[i][j * VEC_SIZE + k] = tmp.f;
            }
        }
    }
}

// ============================================================
// main
// ============================================================

int main()
{
    static float A_fp[SEQ_LEN][EMBED_DIM];
    static float B_fp[SEQ_LEN][EMBED_DIM];

    static float C_sw[SEQ_LEN][EMBED_DIM];
    static float C_hw[SEQ_LEN][EMBED_DIM];

    // packed tensors

    static bus_dt A_bus[SEQ_LEN][PER_ROW];
    static bus_dt B_bus[SEQ_LEN][PER_ROW];
    static bus_dt C_bus[SEQ_LEN][PER_ROW];

    srand(0);

    printf("=====================================\n");
    printf("Residual Testbench Start\n");
    printf("=====================================\n");

    // =====================================================
    // init input
    // =====================================================

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < EMBED_DIM; j++)
        {
            A_fp[i][j] =
                (rand() % 1000) / 100.0f - 5.0f;

            B_fp[i][j] =
                (rand() % 1000) / 100.0f - 5.0f;
        }
    }

    // =====================================================
    // golden
    // =====================================================

    residual_golden(
        A_fp,
        B_fp,
        C_sw);

    // =====================================================
    // pack
    // =====================================================

    pack_tensor(A_fp, A_bus);
    pack_tensor(B_fp, B_bus);

    // =====================================================
    // run kernel
    // =====================================================

    residual(
        A_bus,
        B_bus,
        C_bus);

    // =====================================================
    // unpack
    // =====================================================

    unpack_tensor(
        C_bus,
        C_hw);

    // =====================================================
    // compare
    // =====================================================

    int error_count = 0;

    float max_diff = 0.0f;

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < EMBED_DIM; j++)
        {
            float diff =
                fabs(C_hw[i][j] - C_sw[i][j]);

            if (diff > max_diff)
                max_diff = diff;

            if (diff > MAX_ERR)
            {
                error_count++;

                if (error_count < 10)
                {
                    printf(
                        "Error [%d][%d]\n"
                        "A  = %f\n"
                        "B  = %f\n"
                        "HW = %f\n"
                        "SW = %f\n"
                        "diff = %f\n\n",
                        i,
                        j,
                        A_fp[i][j],
                        B_fp[i][j],
                        C_hw[i][j],
                        C_sw[i][j],
                        diff);
                }
            }
        }
    }

    // =====================================================
    // final
    // =====================================================

    if (error_count == 0)
    {
        printf("TEST PASSED\n");
        printf("Max diff = %.8f\n", max_diff);
    }
    else
    {
        printf("TEST FAILED\n");
        printf("Total errors = %d\n", error_count);
        printf("Max diff = %.8f\n", max_diff);
    }

    printf("=====================================\n");

    return (error_count == 0) ? 0 : 1;
}
