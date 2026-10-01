/*
#include "temporal_layer.h"
#include "host.h"

namespace vivit_temp {

float gelu_int_sw(float x)
{
    const float kAlpha = 0.79788456f;
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

static int8_t quantize_to_int8(float value, float scale)
{
    if (scale == 0.0f || !std::isfinite(scale)) {
        return 0;
    }
    int q = static_cast<int>(std::lrint(value / scale)); // lrint: float/double rounding to long type
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

void quantize_activation_sw(const float in[TEMP_LEN][EMBED_DIM],
                            int8_t out[TEMP_LEN][EMBED_DIM],
                            float scale[TEMP_LEN])
{
    for (int i = 0; i < TEMP_LEN; ++i) {
        float max_abs = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            max_abs = std::max(max_abs, std::fabs(in[i][d]));
        }
        scale[i] = (max_abs > 0.0f) ? (max_abs / 127.5f) : 1.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            out[i][d] = quantize_to_int8(in[i][d], scale[i]);
        }
    }
}

void layernorm_sw(const float in[TEMP_LEN][EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM],
                  const float scale[EMBED_DIM],
                  const float bias[EMBED_DIM])
{
    constexpr double kEps = 1.0e-5;
    for (int i = 0; i < TEMP_LEN; ++i) {
        double mean = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            mean += static_cast<double>(in[i][d]);
        }
        mean /= static_cast<double>(EMBED_DIM);

        double var = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double diff = static_cast<double>(in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(EMBED_DIM);

        const double inv_std = 1.0 / std::sqrt(var + kEps);
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double norm = (static_cast<double>(in[i][d]) - mean) * inv_std;
            out[i][d] = static_cast<float>(static_cast<double>(scale[d]) * norm +
                                           static_cast<double>(bias[d]));
        }
    }
}

void linear_quant_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
                     const float in_scale[TEMP_LEN],
                     const int8_t weight[EMBED_DIM][EMBED_DIM],
                     const float weight_scale[EMBED_DIM],
                     const float bias[EMBED_DIM],
                     float out[TEMP_LEN][EMBED_DIM])
{
    for (int i = 0; i < TEMP_LEN; ++i) {
        for (int o = 0; o < EMBED_DIM; ++o) {
            double acc = static_cast<double>(bias[o]);
            const double scale = static_cast<double>(in_scale[i]) *
                                 static_cast<double>(weight_scale[o]);
            for (int d = 0; d < EMBED_DIM; ++d) {
                acc += static_cast<double>(in[i][d]) * static_cast<double>(weight[d][o]) * scale;
            }
            out[i][o] = static_cast<float>(acc);
        }
    }
}


void attention_sw(const float in[TEMP_LEN][EMBED_DIM],
                  const int8_t Wq[EMBED_DIM][EMBED_DIM],
                  const int8_t Wk[EMBED_DIM][EMBED_DIM],
                  const int8_t Wv[EMBED_DIM][EMBED_DIM],
                  const int8_t Wo[EMBED_DIM][EMBED_DIM],
                  const float q_weight_scale[EMBED_DIM],
                  const float k_weight_scale[EMBED_DIM],
                  const float v_weight_scale[EMBED_DIM],
                  const float o_weight_scale[EMBED_DIM],
                  const float q_bias[EMBED_DIM],
                  const float k_bias[EMBED_DIM],
                  const float v_bias[EMBED_DIM],
                  const float o_bias[EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM])
{
    std::vector<int8_t> in_q_storage(TEMP_LEN * EMBED_DIM);
    int8_t (*in_q)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(in_q_storage.data());
    float in_scale[TEMP_LEN];
    quantize_activation_sw(in, in_q, in_scale);

    std::vector<float> q_storage(TEMP_LEN * EMBED_DIM), k_storage(TEMP_LEN * EMBED_DIM), v_storage(TEMP_LEN * EMBED_DIM);
    float (*Q)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(q_storage.data());
    float (*K)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(k_storage.data());
    float (*V)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(v_storage.data());
    linear_quant_sw(in_q, in_scale, Wq, q_weight_scale, q_bias, Q);
    linear_quant_sw(in_q, in_scale, Wk, k_weight_scale, k_bias, K);
    linear_quant_sw(in_q, in_scale, Wv, v_weight_scale, v_bias, V);

    std::vector<float> context(TEMP_LEN * EMBED_DIM, 0.0f);
    constexpr float kInvSqrtHeadDim = 1.0f / 8.0f;
    for (int h = 0; h < NUM_HEADS; ++h) {
        const int base = h * HEAD_DIM;
        for (int i = 0; i < TEMP_LEN; ++i) {
            float max_score = -std::numeric_limits<float>::infinity();
            float scores[TEMP_LEN];
            for (int j = 0; j < TEMP_LEN; ++j) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += static_cast<double>(Q[i][base + d]) * static_cast<double>(K[j][base + d]);
                }
                scores[j] = static_cast<float>(dot) * kInvSqrtHeadDim;
                max_score = std::max(max_score, scores[j]);
            }

            double denom = 0.0;
            for (int j = 0; j < TEMP_LEN; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }

            for (int d = 0; d < HEAD_DIM; ++d) { // 乘V
                double acc = 0.0;
                for (int j = 0; j < TEMP_LEN; ++j) {
                    acc += (static_cast<double>(scores[j]) / denom) * static_cast<double>(V[j][base + d]);
                }
                context[i * EMBED_DIM + base + d] = static_cast<float>(acc);
            }
        }
    }

    float (*context_2d)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(context.data());
    std::vector<int8_t> context_q_storage(TEMP_LEN * EMBED_DIM);
    int8_t (*context_q)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(context_q_storage.data());
    float context_scale[TEMP_LEN];
    quantize_activation_sw(context_2d, context_q, context_scale);
    linear_quant_sw(context_q, context_scale, Wo, o_weight_scale, o_bias, out);
}


void mlp_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
            const float scale_input[TEMP_LEN],
            const int8_t W1[EMBED_DIM][MLP_DIM],
            const float b1[MLP_DIM],
            const float scale_w1[MLP_DIM],
            const int8_t W2[MLP_DIM][EMBED_DIM],
            const float b2[EMBED_DIM],
            const float scale_w2[EMBED_DIM],
            float out[TEMP_LEN][EMBED_DIM])
{
    std::vector<float> hidden(MLP_DIM);
    std::vector<int8_t> hidden_q(MLP_DIM);
    float hidden_scale;
    for (int i = 0; i < TEMP_LEN; ++i) {
        for (int h = 0; h < MLP_DIM; ++h) {
            double acc = static_cast<double>(b1[h]);
            const double scale = static_cast<double>(scale_input[i]) * static_cast<double>(scale_w1[h]);
            for (int d = 0; d < EMBED_DIM; ++d) {
                acc += static_cast<double>(in[i][d]) * static_cast<double>(W1[d][h]) * scale;
            }
            hidden[h] = gelu_int_sw(static_cast<float>(acc));
        }

    
        float max_abs = 0.0f;
        for (int d = 0; d < MLP_DIM; ++d) {
            max_abs = std::max(max_abs, std::fabs(hidden[d]));
        }
        hidden_scale = (max_abs > 0.0f) ? (max_abs / 127.5f) : 1.0f;
        for (int d = 0; d < MLP_DIM; ++d) {
            hidden_q[d] = quantize_to_int8(hidden[d], hidden_scale);
        }


        for (int d = 0; d < EMBED_DIM; ++d) {
            double acc = static_cast<double>(b2[d]);
            for (int h = 0; h < MLP_DIM; ++h) {
                acc += static_cast<double>(hidden_q[h]) * static_cast<double>(W2[h][d]) *
                       static_cast<double>(scale_w2[d]) * static_cast<double>(hidden_scale);
            }
            out[i][d] = static_cast<float>(acc);
        }
    }
}

void residual_add(const float a[TEMP_LEN][EMBED_DIM],
                const float b[TEMP_LEN][EMBED_DIM],
                float out[TEMP_LEN][EMBED_DIM])
{
    for (int i = 0; i < TEMP_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
            out[i][d] = a[i][d] + b[i][d];
        }
    }
}

template <typename T>
bool load_txt_values(const std::string &path, T *dst, std::size_t count)
{
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "ERROR: cannot open " << path << '\n';
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!(fin >> value)) {
            std::cerr << "ERROR: expected " << count << " values in " << path
                      << ", stopped at " << i << '\n';
            return false;
        }
        if constexpr (std::is_same<T, int8_t>::value) {
            int q = static_cast<int>(std::lrint(value));
            q = std::max(-128, std::min(127, q));
            dst[i] = static_cast<int8_t>(q);
        } else {
            dst[i] = static_cast<T>(value);
        }
    }
    return true;
}

bool load_Temporal_encoder_block(const std::string &root, int block, EncoderBlockParams &p)
{
    const std::string dir = root + "/encoderblock_" + std::to_string(block) + "/";
    return load_txt_values(dir + "LayerNorm_0_scale.txt", p.ln0_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_0_bias.txt", p.ln0_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_1_scale.txt", p.ln1_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_1_bias.txt", p.ln1_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_kernel_qvalue.txt", p.wq.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_kernel_qvalue.txt", p.wk.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_kernel_qvalue.txt", p.wv.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_kernel_qvalue.txt", p.wo.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_kernel_scale.txt", p.q_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_kernel_scale.txt", p.k_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_kernel_scale.txt", p.v_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_kernel_scale.txt", p.o_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_bias.txt", p.q_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_bias.txt", p.k_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_bias.txt", p.v_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_bias.txt", p.o_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_kernel_qvalue.txt", p.mlp_w1.data(), EMBED_DIM * MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_kernel_qvalue.txt", p.mlp_w2.data(), MLP_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_bias.txt", p.mlp_b1.data(), MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_bias.txt", p.mlp_b2.data(), EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_kernel_scale.txt", p.mlp_s1.data(), MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_kernel_scale.txt", p.mlp_s2.data(), EMBED_DIM);
}

} // namespace vivit_temp


namespace {



bool load_frame_input(const std::string &path, int frame_index, float out[TEMP_LEN][EMBED_DIM])
{
    if (frame_index < 0 || frame_index >= FRAME) {
        std::cerr << "ERROR: frame index must be in [0, " << (FRAME - 1) << "]\n";
        return false;
    }

    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "ERROR: cannot open " << path << '\n';
        return false;
    }

    const std::size_t frame_values = static_cast<std::size_t>(TEMP_LEN) * EMBED_DIM;
    const std::size_t skip_values = static_cast<std::size_t>(frame_index) * frame_values;
    double value = 0.0;
    for (std::size_t i = 0; i < skip_values; ++i) {
        if (!(fin >> value)) {
            std::cerr << "ERROR: input file ended while skipping to frame " << frame_index << '\n';
            return false;
        }
    }
    for (std::size_t i = 0; i < frame_values; ++i) {
        if (!(fin >> value)) {
            std::cerr << "ERROR: expected " << frame_values << " values for frame " << frame_index
                      << ", stopped at " << i << '\n';
            return false;
        }
        out[i / EMBED_DIM][i % EMBED_DIM] = static_cast<float>(value);
    }
    return true;
}


void print_usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " [params_root] [input_txt] [output_txt] [frame_index] [num_blocks]\n"
              << "  params_root : default params/SpatialTransformer\n"
              << "  input_txt   : default params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt\n"
              << "  output_txt  : default /tmp/vivit_sw_frame0_cls.txt\n"
              << "  frame_index : default 0\n"
              << "  num_blocks  : default 12\n";
}

} // namespace


#if 0
int main(int argc, char **argv)
{
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    const std::string params_root = (argc > 1) ? argv[1] : "/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer";
    const std::string input_path = (argc > 2) ? argv[2] : "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0619_result/sample10_intermediate_TemporalTransformer__encoderblock_0__layer_norm_input_1.txt";
    const std::string output_path = (argc > 3) ? argv[3] : "/home/undergraduate/rjchen25/test/vivit_sw_temporal_frame0_cls.txt";
    // "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0619_result/sample10_intermediate_TemporalTransformer__encoderblock_0__encoder_output.txt"
    const int frame_index = (argc > 4) ? std::atoi(argv[4]) : 0;
    int num_blocks = (argc > 5) ? std::atoi(argv[5]) : NUM_ENCODER_BLOCKS;
    if (num_blocks < 0 || num_blocks > NUM_ENCODER_BLOCKS) {
        std::cerr << "ERROR: num_blocks must be in [0, " << NUM_ENCODER_BLOCKS << "]\n";
        return EXIT_FAILURE;
    }

    std::vector<float> state_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> norm_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> attn_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> mlp_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> tmp_storage(TEMP_LEN * EMBED_DIM);
    std::vector<int8_t> mlp_in_storage(TEMP_LEN * EMBED_DIM);
    float (*state)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(state_storage.data());
    float (*norm)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(norm_storage.data());
    float (*attn)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(attn_storage.data());
    float (*tmp_out)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(attn_storage.data());
    float (*mlp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(mlp_storage.data());
    float (*tmp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(tmp_storage.data());
    int8_t (*mlp_in)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(mlp_in_storage.data());
    float mlp_in_scale[TEMP_LEN];

    if (!load_frame_input(input_path, frame_index, state)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int block = 0; block < num_blocks; ++block) {
        std::cout << "Running encoder block " << block << " / " << num_blocks << std::endl;
        EncoderBlockParams p;
        if (!load_encoder_block(params_root, block, p)) {
            return EXIT_FAILURE;
        }

        vivit_temp::layernorm_sw(state, norm, p.ln0_scale.data(), p.ln0_bias.data());
        
        vivit_temp::attention_sw(
            norm,
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wq.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wk.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wv.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wo.data()),
            p.q_scale.data(), p.k_scale.data(), p.v_scale.data(), p.o_scale.data(),
            p.q_bias.data(), p.k_bias.data(), p.v_bias.data(), p.o_bias.data(),
            attn);
        
        
        vivit_temp::residual_add(state, attn, tmp);
        
        vivit_temp::layernorm_sw(tmp, norm, p.ln1_scale.data(), p.ln1_bias.data());
        
        vivit_temp::quantize_activation_sw(norm, mlp_in, mlp_in_scale);
        
        vivit_temp::mlp_sw(
            mlp_in, mlp_in_scale,
            reinterpret_cast<const int8_t (*)[MLP_DIM]>(p.mlp_w1.data()),
            p.mlp_b1.data(), p.mlp_s1.data(),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.mlp_w2.data()),
            p.mlp_b2.data(), p.mlp_s2.data(),
            mlp);
        
        vivit_temp::residual_add(tmp, mlp, state);
    }
    const auto stop = std::chrono::steady_clock::now();

    //if (block > 0){
        std::ofstream fout(output_path);
        if (!fout.is_open()) {
            std::cerr << "ERROR: cannot write " << output_path << '\n';
            return EXIT_FAILURE;
        }
        fout << std::setprecision(9);
        for (int i = 0; i < TEMP_LEN; i++){
            for (int d = 0; d < EMBED_DIM; ++d) {
                fout << state[i][d] << ((d + 1 == EMBED_DIM) ? '\n' : ' ');
            }
        }
    //}

    double elapsed = std::chrono::duration<double>(stop - start).count();
    std::cout << "Wrote CLS token embedding to " << output_path << '\n'
              << "Processed frame " << frame_index << " with " << num_blocks
              << " encoder blocks in " << elapsed << " seconds\n";
    return EXIT_SUCCESS;
}
#endif
*/

#include "temporal_layer.h"
#include "host.h"
#include <omp.h>

namespace vivit_temp {

float gelu_int_sw(float x)
{
    const float kAlpha = 0.79788456f;
    return 0.5f * x * (1.0f + std::tanh(kAlpha * (x + 0.044715f * x * x * x)));
}

static int8_t quantize_to_int8(float value, float scale)
{
    if (scale == 0.0f || !std::isfinite(scale)) {
        return 0;
    }
    int q = static_cast<int>(std::lrint(value / scale)); // lrint: float/double rounding to long type
    q = std::max(-128, std::min(127, q));
    return static_cast<int8_t>(q);
}

void quantize_activation_sw(const float in[TEMP_LEN][EMBED_DIM],
                            int8_t out[TEMP_LEN][EMBED_DIM],
                            float scale[TEMP_LEN])
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TEMP_LEN; ++i) {
        float max_abs = 0.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            max_abs = std::max(max_abs, std::fabs(in[i][d]));
        }
        scale[i] = (max_abs > 0.0f) ? (max_abs / 127.5f) : 1.0f;
        for (int d = 0; d < EMBED_DIM; ++d) {
            out[i][d] = quantize_to_int8(in[i][d], scale[i]);
        }
    }
}

void layernorm_sw(const float in[TEMP_LEN][EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM],
                  const float scale[EMBED_DIM],
                  const float bias[EMBED_DIM])
{
    constexpr double kEps = 1.0e-5;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TEMP_LEN; ++i) {
        double mean = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            mean += static_cast<double>(in[i][d]);
        }
        mean /= static_cast<double>(EMBED_DIM);

        double var = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double diff = static_cast<double>(in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(EMBED_DIM);

        const double inv_std = 1.0 / std::sqrt(var + kEps);
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double norm = (static_cast<double>(in[i][d]) - mean) * inv_std;
            out[i][d] = static_cast<float>(static_cast<double>(scale[d]) * norm +
                                           static_cast<double>(bias[d]));
        }
    }
}

void linear_quant_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
                     const float in_scale[TEMP_LEN],
                     const int8_t weight[EMBED_DIM][EMBED_DIM],
                     const float weight_scale[EMBED_DIM],
                     const float bias[EMBED_DIM],
                     float out[TEMP_LEN][EMBED_DIM])
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < TEMP_LEN; ++i) {
        for (int o = 0; o < EMBED_DIM; ++o) {
            double acc = static_cast<double>(bias[o]);
            const double scale = static_cast<double>(in_scale[i]) *
                                 static_cast<double>(weight_scale[o]);
            for (int d = 0; d < EMBED_DIM; ++d) {
                acc += static_cast<double>(in[i][d]) * static_cast<double>(weight[d][o]) * scale;
            }
            out[i][o] = static_cast<float>(acc);
        }
    }
}


void attention_sw(const float in[TEMP_LEN][EMBED_DIM],
                  const int8_t Wq[EMBED_DIM][EMBED_DIM],
                  const int8_t Wk[EMBED_DIM][EMBED_DIM],
                  const int8_t Wv[EMBED_DIM][EMBED_DIM],
                  const int8_t Wo[EMBED_DIM][EMBED_DIM],
                  const float q_weight_scale[EMBED_DIM],
                  const float k_weight_scale[EMBED_DIM],
                  const float v_weight_scale[EMBED_DIM],
                  const float o_weight_scale[EMBED_DIM],
                  const float q_bias[EMBED_DIM],
                  const float k_bias[EMBED_DIM],
                  const float v_bias[EMBED_DIM],
                  const float o_bias[EMBED_DIM],
                  float out[TEMP_LEN][EMBED_DIM])
{
    std::vector<int8_t> in_q_storage(TEMP_LEN * EMBED_DIM);
    int8_t (*in_q)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(in_q_storage.data());
    float in_scale[TEMP_LEN];
    quantize_activation_sw(in, in_q, in_scale);

    std::vector<float> q_storage(TEMP_LEN * EMBED_DIM), k_storage(TEMP_LEN * EMBED_DIM), v_storage(TEMP_LEN * EMBED_DIM);
    float (*Q)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(q_storage.data());
    float (*K)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(k_storage.data());
    float (*V)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(v_storage.data());
    linear_quant_sw(in_q, in_scale, Wq, q_weight_scale, q_bias, Q);
    linear_quant_sw(in_q, in_scale, Wk, k_weight_scale, k_bias, K);
    linear_quant_sw(in_q, in_scale, Wv, v_weight_scale, v_bias, V);

    std::vector<float> context(TEMP_LEN * EMBED_DIM, 0.0f);
    constexpr float kInvSqrtHeadDim = 1.0f / 8.0f;
    // Parallelize over (head, query-position) pairs: each (h, i) is fully independent.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < NUM_HEADS; ++h) {
        for (int i = 0; i < TEMP_LEN; ++i) {
            const int base = h * HEAD_DIM;
            float max_score = -std::numeric_limits<float>::infinity();
            float scores[TEMP_LEN];
            for (int j = 0; j < TEMP_LEN; ++j) {
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += static_cast<double>(Q[i][base + d]) * static_cast<double>(K[j][base + d]);
                }
                scores[j] = static_cast<float>(dot) * kInvSqrtHeadDim;
                max_score = std::max(max_score, scores[j]);
            }

            double denom = 0.0;
            for (int j = 0; j < TEMP_LEN; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                denom += scores[j];
            }

            for (int d = 0; d < HEAD_DIM; ++d) { // 乘V
                double acc = 0.0;
                for (int j = 0; j < TEMP_LEN; ++j) {
                    acc += (static_cast<double>(scores[j]) / denom) * static_cast<double>(V[j][base + d]);
                }
                context[i * EMBED_DIM + base + d] = static_cast<float>(acc);
            }
        }
    }

    float (*context_2d)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(context.data());
    std::vector<int8_t> context_q_storage(TEMP_LEN * EMBED_DIM);
    int8_t (*context_q)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(context_q_storage.data());
    float context_scale[TEMP_LEN];
    quantize_activation_sw(context_2d, context_q, context_scale);
    linear_quant_sw(context_q, context_scale, Wo, o_weight_scale, o_bias, out);
}


void mlp_sw(const int8_t in[TEMP_LEN][EMBED_DIM],
            const float scale_input[TEMP_LEN],
            const int8_t W1[EMBED_DIM][MLP_DIM],
            const float b1[MLP_DIM],
            const float scale_w1[MLP_DIM],
            const int8_t W2[MLP_DIM][EMBED_DIM],
            const float b2[EMBED_DIM],
            const float scale_w2[EMBED_DIM],
            float out[TEMP_LEN][EMBED_DIM])
{
    // Each row i is fully independent. hidden / hidden_q / hidden_scale must be
    // private per thread, so they are declared *inside* the parallel loop body
    // (each thread gets its own local copy) instead of being shared/reused
    // across iterations as in the original serial version.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TEMP_LEN; ++i) {
        std::vector<float> hidden(MLP_DIM);
        std::vector<int8_t> hidden_q(MLP_DIM);
        float hidden_scale;

        for (int h = 0; h < MLP_DIM; ++h) {
            double acc = static_cast<double>(b1[h]);
            const double scale = static_cast<double>(scale_input[i]) * static_cast<double>(scale_w1[h]);
            for (int d = 0; d < EMBED_DIM; ++d) {
                acc += static_cast<double>(in[i][d]) * static_cast<double>(W1[d][h]) * scale;
            }
            hidden[h] = gelu_int_sw(static_cast<float>(acc));
        }


        float max_abs = 0.0f;
        for (int d = 0; d < MLP_DIM; ++d) {
            max_abs = std::max(max_abs, std::fabs(hidden[d]));
        }
        hidden_scale = (max_abs > 0.0f) ? (max_abs / 127.5f) : 1.0f;
        for (int d = 0; d < MLP_DIM; ++d) {
            hidden_q[d] = quantize_to_int8(hidden[d], hidden_scale);
        }


        for (int d = 0; d < EMBED_DIM; ++d) {
            double acc = static_cast<double>(b2[d]);
            for (int h = 0; h < MLP_DIM; ++h) {
                acc += static_cast<double>(hidden_q[h]) * static_cast<double>(W2[h][d]) *
                       static_cast<double>(scale_w2[d]) * static_cast<double>(hidden_scale);
            }
            out[i][d] = static_cast<float>(acc);
        }
    }
}

void residual_add(const float a[TEMP_LEN][EMBED_DIM],
                const float b[TEMP_LEN][EMBED_DIM],
                float out[TEMP_LEN][EMBED_DIM])
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < TEMP_LEN; ++i) {
        for (int d = 0; d < EMBED_DIM; ++d) {
            out[i][d] = a[i][d] + b[i][d];
        }
    }
}

template <typename T>
bool load_txt_values(const std::string &path, T *dst, std::size_t count)
{
    // I/O-bound sequential parsing; not parallelized.
    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "ERROR: cannot open " << path << '\n';
        return false;
    }
    for (std::size_t i = 0; i < count; ++i) {
        double value = 0.0;
        if (!(fin >> value)) {
            std::cerr << "ERROR: expected " << count << " values in " << path
                      << ", stopped at " << i << '\n';
            return false;
        }
        if constexpr (std::is_same<T, int8_t>::value) {
            int q = static_cast<int>(std::lrint(value));
            q = std::max(-128, std::min(127, q));
            dst[i] = static_cast<int8_t>(q);
        } else {
            dst[i] = static_cast<T>(value);
        }
    }
    return true;
}

bool load_Temporal_encoder_block(const std::string &root, int block, EncoderBlockParams &p)
{
    const std::string dir = root + "/encoderblock_" + std::to_string(block) + "/";
    return load_txt_values(dir + "LayerNorm_0_scale.txt", p.ln0_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_0_bias.txt", p.ln0_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_1_scale.txt", p.ln1_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "LayerNorm_1_bias.txt", p.ln1_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_kernel_qvalue.txt", p.wq.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_kernel_qvalue.txt", p.wk.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_kernel_qvalue.txt", p.wv.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_kernel_qvalue.txt", p.wo.data(), EMBED_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_kernel_scale.txt", p.q_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_kernel_scale.txt", p.k_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_kernel_scale.txt", p.v_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_kernel_scale.txt", p.o_scale.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_query_bias.txt", p.q_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_key_bias.txt", p.k_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_value_bias.txt", p.v_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MultiHeadDotProductAttention_0_out_bias.txt", p.o_bias.data(), EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_kernel_qvalue.txt", p.mlp_w1.data(), EMBED_DIM * MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_kernel_qvalue.txt", p.mlp_w2.data(), MLP_DIM * EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_bias.txt", p.mlp_b1.data(), MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_bias.txt", p.mlp_b2.data(), EMBED_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_0_kernel_scale.txt", p.mlp_s1.data(), MLP_DIM) &&
           load_txt_values(dir + "MlpBlock_0_Dense_1_kernel_scale.txt", p.mlp_s2.data(), EMBED_DIM);
}

} // namespace vivit_temp


namespace {



bool load_frame_input(const std::string &path, int frame_index, float out[TEMP_LEN][EMBED_DIM])
{
    if (frame_index < 0 || frame_index >= FRAME) {
        std::cerr << "ERROR: frame index must be in [0, " << (FRAME - 1) << "]\n";
        return false;
    }

    std::ifstream fin(path);
    if (!fin.is_open()) {
        std::cerr << "ERROR: cannot open " << path << '\n';
        return false;
    }

    const std::size_t frame_values = static_cast<std::size_t>(TEMP_LEN) * EMBED_DIM;
    const std::size_t skip_values = static_cast<std::size_t>(frame_index) * frame_values;
    double value = 0.0;
    for (std::size_t i = 0; i < skip_values; ++i) {
        if (!(fin >> value)) {
            std::cerr << "ERROR: input file ended while skipping to frame " << frame_index << '\n';
            return false;
        }
    }
    for (std::size_t i = 0; i < frame_values; ++i) {
        if (!(fin >> value)) {
            std::cerr << "ERROR: expected " << frame_values << " values for frame " << frame_index
                      << ", stopped at " << i << '\n';
            return false;
        }
        out[i / EMBED_DIM][i % EMBED_DIM] = static_cast<float>(value);
    }
    return true;
}


void print_usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " [params_root] [input_txt] [output_txt] [frame_index] [num_blocks]\n"
              << "  params_root : default params/SpatialTransformer\n"
              << "  input_txt   : default params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt\n"
              << "  output_txt  : default /tmp/vivit_sw_frame0_cls.txt\n"
              << "  frame_index : default 0\n"
              << "  num_blocks  : default 12\n";
}

} // namespace


#if 0
int main(int argc, char **argv)
{
    if (argc > 1 && (std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help")) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }

    const std::string params_root = (argc > 1) ? argv[1] : "/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer";
    const std::string input_path = (argc > 2) ? argv[2] : "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0619_result/sample10_intermediate_TemporalTransformer__encoderblock_0__layer_norm_input_1.txt";
    const std::string output_path = (argc > 3) ? argv[3] : "/home/undergraduate/rjchen25/test/vivit_sw_temporal_frame0_cls.txt";
    // "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0619_result/sample10_intermediate_TemporalTransformer__encoderblock_0__encoder_output.txt"
    const int frame_index = (argc > 4) ? std::atoi(argv[4]) : 0;
    int num_blocks = (argc > 5) ? std::atoi(argv[5]) : NUM_ENCODER_BLOCKS;
    if (num_blocks < 0 || num_blocks > NUM_ENCODER_BLOCKS) {
        std::cerr << "ERROR: num_blocks must be in [0, " << NUM_ENCODER_BLOCKS << "]\n";
        return EXIT_FAILURE;
    }

    std::vector<float> state_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> norm_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> attn_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> mlp_storage(TEMP_LEN * EMBED_DIM);
    std::vector<float> tmp_storage(TEMP_LEN * EMBED_DIM);
    std::vector<int8_t> mlp_in_storage(TEMP_LEN * EMBED_DIM);
    float (*state)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(state_storage.data());
    float (*norm)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(norm_storage.data());
    float (*attn)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(attn_storage.data());
    float (*tmp_out)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(attn_storage.data());
    float (*mlp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(mlp_storage.data());
    float (*tmp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(tmp_storage.data());
    int8_t (*mlp_in)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(mlp_in_storage.data());
    float mlp_in_scale[TEMP_LEN];

    if (!load_frame_input(input_path, frame_index, state)) {
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const auto start = std::chrono::steady_clock::now();
    for (int block = 0; block < num_blocks; ++block) {
        std::cout << "Running encoder block " << block << " / " << num_blocks << std::endl;
        EncoderBlockParams p;
        if (!load_encoder_block(params_root, block, p)) {
            return EXIT_FAILURE;
        }

        vivit_temp::layernorm_sw(state, norm, p.ln0_scale.data(), p.ln0_bias.data());
        
        vivit_temp::attention_sw(
            norm,
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wq.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wk.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wv.data()),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.wo.data()),
            p.q_scale.data(), p.k_scale.data(), p.v_scale.data(), p.o_scale.data(),
            p.q_bias.data(), p.k_bias.data(), p.v_bias.data(), p.o_bias.data(),
            attn);
        
        
        vivit_temp::residual_add(state, attn, tmp);
        
        vivit_temp::layernorm_sw(tmp, norm, p.ln1_scale.data(), p.ln1_bias.data());
        
        vivit_temp::quantize_activation_sw(norm, mlp_in, mlp_in_scale);
        
        vivit_temp::mlp_sw(
            mlp_in, mlp_in_scale,
            reinterpret_cast<const int8_t (*)[MLP_DIM]>(p.mlp_w1.data()),
            p.mlp_b1.data(), p.mlp_s1.data(),
            reinterpret_cast<const int8_t (*)[EMBED_DIM]>(p.mlp_w2.data()),
            p.mlp_b2.data(), p.mlp_s2.data(),
            mlp);
        
        vivit_temp::residual_add(tmp, mlp, state);
    }
    const auto stop = std::chrono::steady_clock::now();

    //if (block > 0){
        std::ofstream fout(output_path);
        if (!fout.is_open()) {
            std::cerr << "ERROR: cannot write " << output_path << '\n';
            return EXIT_FAILURE;
        }
        fout << std::setprecision(9);
        for (int i = 0; i < TEMP_LEN; i++){
            for (int d = 0; d < EMBED_DIM; ++d) {
                fout << state[i][d] << ((d + 1 == EMBED_DIM) ? '\n' : ' ');
            }
        }
    //}

    double elapsed = std::chrono::duration<double>(stop - start).count();
    std::cout << "Wrote CLS token embedding to " << output_path << '\n'
              << "Processed frame " << frame_index << " with " << num_blocks
              << " encoder blocks in " << elapsed << " seconds\n";
    return EXIT_SUCCESS;
}
#endif