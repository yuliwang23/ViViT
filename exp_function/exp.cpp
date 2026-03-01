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

    // bit operation
    union { float f; uint32_t u; } val, result; // 以 float / uint32 方式解讀這 32 bits

    val.f = approx_2_F;
    
    // 提取原本的指數（bias = 127），加上 I，再放回去
    int orig_exp = (int)((val.u >> 23) & 0xFF);  // 取出 8-bit 指數。右移 23 位後只保留最低 8 位
    int new_exp  = orig_exp + I;                  // 假設 I = 3 (代表我們要乘以 2³): new_exp = 127 + 3 = 130。 這樣新的 float 指數欄位 = 130，代表真實指數 = 130 - 127 = 3 

    result.u = (val.u & 0x807FFFFF) | ((uint32_t)(new_exp & 0xFF) << 23);
    // 左半: 清除舊指數，保留符號和尾數；右半: 把新指數移到正確位置。

    return result.f;

}

/*
float fast_exp(float x) // 把 e^x = 2^(x * log(2)e) 換成 2^y 來計算
{
    float y = x * 1.44269504f; // y = x * log(2)e
    int I = (int)y;
    float F = y - (float)I; // 把 y 拆成整數部分 I 和小數部分 F (0 <= F < 1)

    // 2^F 的線性近似 (1 + F)
    float approx_2_F = 1.0f + F; // 算2^F。直接用 1 + F 是一個非常粗略但極快的線性近似

    return ldexpf(approx_2_F, I); // ldexpf(val, exp): val * 2^exp = 2^F * 2^I
}
*/