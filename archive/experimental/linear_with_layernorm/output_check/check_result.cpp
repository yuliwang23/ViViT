#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// ============================================================================
// Transformer Dimensions & Verification Parameters
// ============================================================================
const int SEQ_LEN = 197;
const int EMBED_DIM = 768;
const int TOTAL_ELEMENTS = SEQ_LEN * EMBED_DIM;

// English Comment: 
// Tolerance for floating-point comparison. 
// Since quantization is done in INT8 and accumulated in INT26/32, the final float 
// conversion might have minor ULPs (Unit in the Last Place) differences.
const float TOLERANCE = 1e-2f; 

// English Comment:
// Helper function to verify a specific tensor (Q, K, or V).
// It reads the respective golden and hardware files, compares them, 
// and writes the detailed log to the provided output file stream.
bool verify_tensor(const string& tensor_name, ofstream& log_out) {
    string golden_file = tensor_name + "_output_golden.txt";
    string hw_file     = tensor_name + "_output.txt";

    log_out << "\n>>> Verifying Tensor: " << tensor_name << " <<<" << endl;
    log_out << "Golden Reference File : " << golden_file << endl;
    log_out << "Hardware Output File  : " << hw_file << endl;

    // 1. Open both files
    ifstream f_golden(golden_file);
    ifstream f_hw(hw_file);

    if (!f_golden.is_open()) {
        log_out << "[ERROR] Could not open " << golden_file << endl;
        cerr << "[ERROR] Could not open " << golden_file << endl;
        return false;
    }
    if (!f_hw.is_open()) {
        log_out << "[ERROR] Could not open " << hw_file << endl;
        cerr << "[ERROR] Could not open " << hw_file << endl;
        return false;
    }

    // 2. Variables for statistical analysis
    int mismatch_count = 0;
    float max_error = 0.0f;
    double sum_squared_error = 0.0;
    
    float val_g, val_h;
    int index = 0;

    // 3. Read and compare element by element
    // English Comment: 
    // The '>>' operator automatically handles spaces and newlines, 
    // so the physical line breaks (e.g., every 64 elements) won't affect the parsing.
    log_out << fixed << setprecision(6);
    while (index < TOTAL_ELEMENTS) {
        if (!(f_golden >> val_g)) {
            log_out << "[ERROR] Golden file ended prematurely at index " << index << endl;
            break;
        }
        if (!(f_hw >> val_h)) {
            log_out << "[ERROR] Hardware file ended prematurely at index " << index << endl;
            break;
        }

        // Calculate absolute error
        float diff = std::abs(val_g - val_h);
        
        // Update max error and MSE
        if (diff > max_error) {
            max_error = diff;
        }
        sum_squared_error += (diff * diff);

        // Check against tolerance
        if (diff > TOLERANCE) {
            // English Comment:
            // Print details for the first 10 errors only to avoid huge log files
            if (mismatch_count < 10) { 
                log_out << "    Mismatch at index " << index 
                        << " | Golden: " << val_g 
                        << " | HW: " << val_h 
                        << " | Diff: " << diff << endl;
            }
            mismatch_count++;
        }
        index++;
    }

    f_golden.close();
    f_hw.close();

    // 4. Print Verification Report for this specific tensor
    log_out << "-------------------------------------------------" << endl;
    log_out << "Report for " << tensor_name << ":" << endl;
    log_out << "Total Elements Checked : " << index << " / " << TOTAL_ELEMENTS << endl;
    log_out << "Maximum Absolute Error : " << max_error << endl;
    
    if (index > 0) {
        double mse = sum_squared_error / index;
        log_out << "Mean Squared Error(MSE): " << mse << endl;
    }

    log_out << "Mismatch Count         : " << mismatch_count << " elements exceeded tolerance (" << TOLERANCE << ")" << endl;

    bool passed = (mismatch_count == 0 && index == TOTAL_ELEMENTS);
    if (passed) {
        log_out << "[SUCCESS] " << tensor_name << " Hardware output matches Golden reference perfectly!" << endl;
    } else {
        log_out << "[FAILED] " << tensor_name << " Verification failed. Hardware output has discrepancies." << endl;
    }
    log_out << "-------------------------------------------------\n" << endl;

    return passed;
}

int main() {
    string log_filename = "verification_log.txt";
    ofstream log_file(log_filename);

    if (!log_file.is_open()) {
        cerr << "Error: Could not create log file " << log_filename << endl;
        return EXIT_FAILURE;
    }

    // Console output to indicate start
    cout << "=== Starting Hardware vs Software Verification ===" << endl;
    cout << "Verifying Q, K, V outputs..." << endl;
    cout << "Writing detailed log to : " << log_filename << endl;
    
    log_file << "=== Hardware vs Software Verification Log ===" << endl;
    log_file << "Error Tolerance        : " << TOLERANCE << endl;

    // English Comment:
    // List of tensors to verify. This allows easy scaling if you add more (e.g., 'Out', 'FFN').
    vector<string> tensors = {"Q", "K", "V"};
    bool all_passed = true;

    for (const string& tensor : tensors) {
        cout << "Checking " << tensor << " ... ";
        
        bool result = verify_tensor(tensor, log_file);
        
        if (result) {
            cout << "PASSED" << endl;
        } else {
            cout << "FAILED (See log for details)" << endl;
            all_passed = false;
        }
    }

    log_file << "=================================================" << endl;
    if (all_passed) {
        log_file << "FINAL RESULT: ALL TENSORS PASSED!" << endl;
        cout << "\n[SUCCESS] All hardware outputs (Q, K, V) match perfectly!" << endl;
    } else {
        log_file << "FINAL RESULT: ONE OR MORE TENSORS FAILED!" << endl;
        cout << "\n[FAILED] One or more hardware outputs have discrepancies. Check '" << log_filename << "'." << endl;
    }
    log_file << "=================================================" << endl;

    log_file.close();

    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}