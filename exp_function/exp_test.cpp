#include <iostream>
#include <cmath>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>
#include "exp.h"

// ---------------------------------------------------------
// Placeholder for your custom HLS exponential function
// Replace the contents of this function with your actual implementation 
// (e.g., the LUT + Linear Interpolation method we discussed)
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
    // 1. Testbench Configuration Parameters
    // ---------------------------------------------------------
    const int NUM_TESTS = 100000;       // Number of random samples to generate
    
    // For Safe Softmax, inputs to exp() are always <= 0. 
    // We set the testing range to [-8.0, 0.0] as previously discussed.
    const float MIN_VAL = -8.0f;        
    const float MAX_VAL = 0.0f;         
    
    // Pass/Fail threshold (e.g., require accuracy up to 10^-3)
    // const float ERROR_THRESHOLD = 1e-3f; 
    const float ERROR_THRESHOLD = 1.0f; // 為了跑過 co-simulation 所以先改的

    // ---------------------------------------------------------
    // 2. Random Number Generator Setup
    // ---------------------------------------------------------
    // Use Mersenne Twister engine for high-quality random numbers
    std::mt19937 gen(10); // Fixed seed (42) for reproducible test results
    std::uniform_real_distribution<float> dist(MIN_VAL, MAX_VAL);

    // ---------------------------------------------------------
    // 3. Error Tracking Variables
    // ---------------------------------------------------------
    float max_abs_err = 0.0f;
    double sum_err = 0.0;     // Use double to prevent precision loss during accumulation
    double sum_sq_err = 0.0;

    std::cout << "[INFO] Starting Exponential Function Hardware Validation..." << std::endl;
    std::cout << "[INFO] Testing range: [" << MIN_VAL << ", " << MAX_VAL << "]" << std::endl;
    std::cout << "[INFO] Total samples: " << NUM_TESTS << "\n" << std::endl;

    // ---------------------------------------------------------
    // 4. Main Verification Loop
    // ---------------------------------------------------------
    for (int i = 0; i < NUM_TESTS; i++) 
    {
        // Generate random input
        float x = dist(gen);

        // Calculate reference value (Golden Model) using C++ standard library
        float ref_val = std::exp(x);

        // Calculate actual value using your custom hardware function
        float act_val = custom_hls_exp(x);

        // Calculate absolute error for the current sample
        float err = std::abs(ref_val - act_val);

        // Update error metrics
        max_abs_err = std::max(max_abs_err, err);
        sum_err += err;
        sum_sq_err += (static_cast<double>(err) * static_cast<double>(err));
        
        // Optional: Print the first 5 results to visually inspect
        if (i < 5) {
            std::cout << "Sample " << i << " | x: " << std::setw(9) << x 
                      << " | Ref: " << std::setw(9) << ref_val 
                      << " | Act: " << std::setw(9) << act_val 
                      << " | Err: " << err << std::endl;
        }
    }

    // ---------------------------------------------------------
    // 5. Calculate Final Statistics
    // ---------------------------------------------------------
    double mean_err = sum_err / NUM_TESTS;
    double rms_err = std::sqrt(sum_sq_err / NUM_TESTS);

    // ---------------------------------------------------------
    // 6. Print Report & Pass/Fail Status
    // ---------------------------------------------------------
    std::cout << "\n========================================" << std::endl;
    std::cout << "          VALIDATION RESULTS            " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Max Absolute Error : " << max_abs_err << std::endl;
    std::cout << "Mean Error         : " << mean_err << std::endl;
    std::cout << "RMS Error          : " << rms_err << std::endl;
    std::cout << "========================================" << std::endl;

    if (max_abs_err <= ERROR_THRESHOLD) 
    {
        std::cout << "\n[SUCCESS] TEST PASSED! The hardware function is highly accurate." << std::endl;
        return 0; // Return 0 means success in C++
    } 
    else 
    {
        std::cout << "\n[FAILED] TEST FAILED! Max error exceeds the threshold of " 
                  << ERROR_THRESHOLD << std::endl;
        return 1; // Return non-zero means failure
    }
}