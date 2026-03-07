#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <algorithm>
#include "softmax.h"

// ---------------------------------------------------------
// 1. Software Golden Model (Reference)
// ---------------------------------------------------------
// Changed data types from 'int' to 'float' to preserve mathematical 
// precision of probabilities and exponential functions.

void softmax_sw_golden(float in[VEC_MAX],float out[VEC_MAX])     
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


// ---------------------------------------------------------
// 2. Hardware Design Under Test (DUT)
// ---------------------------------------------------------
// This is where you will place your HLS-optimized code 
// (e.g., the LUT + Interpolation ap_fixed version).
void softmax_hw_dut(float input[VEC_MAX], float output[VEC_MAX])
{
    uint256_dt input_bus[VEC_MAX / 8];
    uint256_dt output_bus[VEC_MAX / 8];


    for (int i = 0; i < VEC_MAX / 8; i++) {
        uint256_dt chunk;
        for (int k = 0; k < 8; k++) {
            float_cast_t converter;
            converter.f = input[i * 8 + k];
            chunk.range(k * 32 + 31, k * 32) = converter.i;
        }
        input_bus[i] = chunk;
    }

    softmax(input_bus, output_bus);
    for (int i = 0; i < VEC_MAX / 8; i++) {
        uint256_dt chunk = output_bus[i];
        for (int k = 0; k < 8; k++) {
            float_cast_t converter;
            converter.i = chunk.range(k * 32 + 31, k * 32);
            output[i * 8 + k] = converter.f;
        }
    }
}

// ---------------------------------------------------------
// 3. Testbench Main Function
// ---------------------------------------------------------
int main()
{
    // Arrays for input and outputs
    float input_data[VEC_MAX];
    float sw_output[VEC_MAX];
    float hw_output[VEC_MAX];

    // Setup Random Number Generator
    // Neural network logits typically fall in a moderate range like [-15.0, 15.0]
    std::mt19937 gen(12345); // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-15.0f, 15.0f);

    std::cout << "[INFO] Generating " << VEC_MAX << " random input logits..." << std::endl;
    for (int i = 0; i < VEC_MAX; i++)
    {
        input_data[i] = dist(gen);
    }

    // Execute Software Golden Model
    std::cout << "[INFO] Running Software Golden Model..." << std::endl;
    softmax_sw_golden(input_data, sw_output);

    // Execute Hardware DUT Model
    std::cout << "[INFO] Running Hardware DUT Model..." << std::endl;
    softmax_hw_dut(input_data, hw_output);

    // ---------------------------------------------------------
    // 4. Verification and Error Calculation
    // ---------------------------------------------------------
    float max_abs_error = 0.0f;
    const float ERROR_TOLERANCE = 1e-3f; // Acceptable error margin for hardware

    std::cout << "\n[RESULT] Comparing outputs (showing first 5 elements):" << std::endl;
    std::cout << std::left << std::setw(10) << "Index" 
              << std::setw(15) << "Input" 
              << std::setw(15) << "SW_Golden" 
              << std::setw(15) << "HW_DUT" 
              << "Error" << std::endl;
    std::cout << "-----------------------------------------------------------------" << std::endl;

    for (int i = 0; i < VEC_MAX; i++)
    {
        float err = std::abs(sw_output[i] - hw_output[i]);
        max_abs_error = std::max(max_abs_error, err);

        // Print first 5 results for visual inspection
        if (i < 5)
        {
            std::cout << std::left << std::setw(10) << i 
                      << std::fixed << std::setprecision(6)
                      << std::setw(15) << input_data[i] 
                      << std::setw(15) << sw_output[i] 
                      << std::setw(15) << hw_output[i] 
                      << err << std::endl;
        }
    }

    std::cout << "-----------------------------------------------------------------" << std::endl;
    std::cout << "\nMax Absolute Error: " << max_abs_error << std::endl;

    // Check if the hardware implementation passes the test
    if (max_abs_error <= ERROR_TOLERANCE)
    {
        std::cout << "[SUCCESS] Hardware Validation Passed!" << std::endl;
        return 1; // Return 0 indicates success in C++ environments
    }
    else
    {
        std::cout << "[FAILED] Error exceeds tolerance threshold of " << ERROR_TOLERANCE << std::endl;
        return 1; // Return non-zero indicates failure
    }
}
