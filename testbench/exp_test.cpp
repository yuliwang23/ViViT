#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include "exp.h"

// ---------------------------------------------------------
// Wrapper for your HLS exponential function
// ---------------------------------------------------------
float custom_hls_exp(float x) 
{
    float y_out = 0.0f;
    exp_top(x, &y_out);
    return y_out;
}

int main() 
{
    // ---------------------------------------------------------
    // 1. Testbench Configuration
    // ---------------------------------------------------------
    const int NUM_TESTS = 100000;

    // Sweep testing range
    const float MIN_VAL = -8.0f;
    const float MAX_VAL = 0.0f;

    const float ERROR_THRESHOLD = 1.0f;

    // ---------------------------------------------------------
    // 2. Error Tracking
    // ---------------------------------------------------------
    float max_abs_err = 0.0f;
    double sum_err = 0.0;
    double sum_sq_err = 0.0;

    float worst_input = 0.0f;

    std::cout << "[INFO] Starting Exponential Function Hardware Validation..." << std::endl;
    std::cout << "[INFO] Sweep range: [" << MIN_VAL << ", " << MAX_VAL << "]" << std::endl;
    std::cout << "[INFO] Total samples: " << NUM_TESTS << "\n" << std::endl;

    // ---------------------------------------------------------
    // 3. Sweep Testing Loop
    // ---------------------------------------------------------
    for (int i = 0; i < NUM_TESTS; i++) 
    {
        // Uniform sweep across range
        float x = MIN_VAL + (MAX_VAL - MIN_VAL) * i / (NUM_TESTS - 1);

        // Reference (Golden Model)
        float ref_val = std::exp(x);

        // Hardware result
        float act_val = custom_hls_exp(x);

        // Absolute error
        float err = std::abs(ref_val - act_val);

        // Update statistics
        if (err > max_abs_err)
        {
            max_abs_err = err;
            worst_input = x;
        }

        sum_err += err;
        sum_sq_err += (double)err * (double)err;

        // Print first few samples
        if (i < 5) 
        {
            std::cout << "Sample " << i
                      << " | x: " << std::setw(10) << x
                      << " | Ref: " << std::setw(12) << ref_val
                      << " | Act: " << std::setw(12) << act_val
                      << " | Err: " << err
                      << std::endl;
        }
    }

    // ---------------------------------------------------------
    // 4. Final Statistics
    // ---------------------------------------------------------
    double mean_err = sum_err / NUM_TESTS;
    double rms_err = std::sqrt(sum_sq_err / NUM_TESTS);

    // ---------------------------------------------------------
    // 5. Report
    // ---------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << "          VALIDATION RESULTS            " << std::endl;
    std::cout << "========================================" << std::endl;

    std::cout << std::fixed << std::setprecision(8);

    std::cout << "Max Absolute Error : " << max_abs_err << std::endl;
    std::cout << "Worst Case Input   : " << worst_input << std::endl;
    std::cout << "Mean Error         : " << mean_err << std::endl;
    std::cout << "RMS Error          : " << rms_err << std::endl;

    std::cout << "========================================" << std::endl;

    if (max_abs_err <= ERROR_THRESHOLD) 
    {
        std::cout << "\n[SUCCESS] TEST PASSED!" << std::endl;
        return 0;
    } 
    else 
    {
        std::cout << "\n[FAILED] TEST FAILED! Max error exceeds threshold: "
                  << ERROR_THRESHOLD << std::endl;
        return 1;
    }
}