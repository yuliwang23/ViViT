#include "exp.h"
// using std::exp;

extern "C" {
void exp_top(float x_in, float *y_out)
{
    #pragma HLS INTERFACE s_axilite port=x_in   bundle=control
    #pragma HLS INTERFACE s_axilite port=y_out  bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    *y_out = fast_exp(x_in);
}
}

float fast_exp(float x)
{
    #pragma HLS PIPELINE II=1
    #pragma HLS INLINE off

    float y = x * 1.44269504f;
    int I = (int)y;
    float F = y - (float)I;

    float approx_2_F = 1.0f + F;

    union { float f; uint32_t u; } val, result;

    val.f = approx_2_F;

    ap_uint<32> bits = val.u;
    ap_uint<8> orig_exp = bits.range(30,23);

    int new_exp = orig_exp + I;

    bits.range(30,23) = new_exp;

    result.u = bits;
    return result.f;
}