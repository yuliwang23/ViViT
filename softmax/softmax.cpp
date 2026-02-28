#include "softmax.h"
void softmax(float in[VEC_MAX],
             float out[VEC_MAX])
{
    using std::exp; // TODO: change into hls::exp

    float max_val = -1e38f;
    float sum_exp = 0.0f;

    // Online calculate Max and Sum
    for (int i = 0; i < VEC_MAX; i++)
    {
        float x = in[i];

        if (x > max_val)
        {
            // sum_new = sum_old *
            // exp(max_old - max_new) + exp(x - max_new)
            float diff = max_val - x;
            sum_exp = sum_exp * exp(diff) + 1.0f;
            max_val = x;
        }
        else
        {
            // sum_new = sum_old + exp(x - max_old)
            float diff = x - max_val;
            sum_exp += exp(diff);
        }
    }

    // Normalization
    for (int i = 0; i < VEC_MAX; i++)
    {
        float diff = in[i] - max_val;
        out[i] = exp(diff) / sum_exp;
    }
}
