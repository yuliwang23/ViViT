#include <fstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

constexpr int SEQ_LEN  = 197;
constexpr int EMBED_DIM = 768;

// Helper: random float in [min_val, max_val]
float rand_float(float min_val, float max_val) {
    float scale = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return min_val + scale * (max_val - min_val);
}

int main() {

    // Fixed seed for reproducibility (matches original testbench)
    srand(42);

    // ----------------------------------------------------------------
    // Generate input_x.txt  —  x[SEQ_LEN][EMBED_DIM]
    // Layout: SEQ_LEN rows, each row has EMBED_DIM space-separated floats
    // ----------------------------------------------------------------
    {
        std::ofstream f("input_x.txt");
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open input_x.txt for write\n";
            return 1;
        }
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < EMBED_DIM; j++) {
                // Simulated raw feature map distribution [-15.0, 15.0]
                f << rand_float(-15.0f, 15.0f);
                if (j < EMBED_DIM - 1) f << " ";
            }
            f << "\n";
        }
        std::cout << "Generated input_x.txt     (" << SEQ_LEN << " x " << EMBED_DIM << " floats)\n";
    }

    // ----------------------------------------------------------------
    // Generate input_gamma.txt  —  gamma[EMBED_DIM]
    // Layout: one row of EMBED_DIM space-separated floats
    // ----------------------------------------------------------------
    {
        std::ofstream f("input_gamma.txt");
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open input_gamma.txt for write\n";
            return 1;
        }
        for (int j = 0; j < EMBED_DIM; j++) {
            // gamma usually centers around 1.0 after training [0.5, 2.0]
            f << rand_float(0.5f, 2.0f);
            if (j < EMBED_DIM - 1) f << " ";
        }
        f << "\n";
        std::cout << "Generated input_gamma.txt (" << EMBED_DIM << " floats)\n";
    }

    // ----------------------------------------------------------------
    // Generate input_beta.txt  —  beta[EMBED_DIM]
    // Layout: one row of EMBED_DIM space-separated floats
    // ----------------------------------------------------------------
    {
        std::ofstream f("input_beta.txt");
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open input_beta.txt for write\n";
            return 1;
        }
        for (int j = 0; j < EMBED_DIM; j++) {
            // beta usually centers around 0.0 after training [-1.0, 1.0]
            f << rand_float(-1.0f, 1.0f);
            if (j < EMBED_DIM - 1) f << " ";
        }
        f << "\n";
        std::cout << "Generated input_beta.txt  (" << EMBED_DIM << " floats)\n";
    }

    std::cout << "Done.\n";
    return 0;
}