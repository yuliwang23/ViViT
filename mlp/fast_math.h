#ifndef FAST_MATH_H
#define FAST_MATH_H

#include <ap_int.h>
#include <stdint.h>

float fast_exp(float x)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    float y = x * 1.44269504f;
    int I = (int)y;
    float F = y - (float)I;

    // floor 修正：若 y 為負且有小數部分，truncate 會比 floor 大 1，
    // 此時 F 會是負值，需要 I -= 1 且 F += 1.0 來對齊 floor
    if (F < 0.0f) {
        I -= 1;
        F += 1.0f;
    }
    // 修正後 F 保證落在 [0, 1)

    float approx_2_F = 1.0f + F;    // 2^F 的線性近似

    union { float f; uint32_t u; } val, result;
    val.f = approx_2_F;

    int orig_exp = (int)((val.u >> 23) & 0xFF);
    int new_exp  = orig_exp + I;

    result.u = (val.u & 0x807FFFFF) | ((uint32_t)(new_exp & 0xFF) << 23);

    return result.f;
}

float fast_reciprocal(float x)
{
#pragma HLS PIPELINE II=1
#pragma HLS INLINE off

    union { float f; uint32_t u; } val, result;
    val.f = x;

    result.u = 0x7EF311C3u - val.u;
    float y = result.f;

    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);

    return y;
}

// sigmoid_top: 依正負號分兩種情況，避免 fast_exp 吃到正數大引數而 overflow
//   x >= 0:  sigmoid(x) = 1 / (1 + e^-x)
//   x <  0:  sigmoid(x) = 1 - 1/(1 + e^-|x|)
float sigmoid_top(float x)
{
#pragma HLS INLINE

    bool  neg = (x < 0.0f); // 判斷 x 正負
    float ax  = neg ? -x : x; // ax >= 0

    float e         = fast_exp(-ax); // 引數 <= 0
    float denom     = 1.0f + e; // denom >= 1
    float inv_denom = fast_reciprocal(denom);

    return neg ? (1.0f - inv_denom) : inv_denom;
}

// GELU(x) ≈ x * Sigmoid(1.702x)
float gelu_approx_top(float x)
{
#pragma HLS INLINE
    return x * sigmoid_top(1.702f * x);
}

#endif // FAST_MATH_H