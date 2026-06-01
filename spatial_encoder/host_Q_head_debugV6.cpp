/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
** HOST Code
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

// openmp
#include <omp.h>

using namespace std;

#include <CL/cl.h>
#include "help_functions.h"
#include "host.h"
// #include "layer.h"
#include "utility.h"

#define ALL_MESSAGES
#define HOST_DEBUG_SERIAL_STAGES 1
#define HOST_DEBUG_LN_SELF_TEST 0
#define HOST_DEBUG_READBACK 0

constexpr int MLP_KERNEL_SEQ_LEN = SEQ_LEN;
constexpr int MLP_CHUNK_U32 = 8;
constexpr int MLP_OUTPUT_CHUNKS = (SEQ_LEN * EMBED_DIM) / MLP_CHUNK_U32;
constexpr int MLP_W1_CHUNKS = (MLP_DIM * EMBED_DIM) / 32;
constexpr int MLP_W2_CHUNKS = (EMBED_DIM * MLP_DIM) / 32;

static string join_path(const string &dir, const string &name)
{
    if (dir.empty())
        return name;
    const char last = dir[dir.size() - 1];
    if (last == '/' || last == '\\')
        return dir + name;
    return dir + "/" + name;
}

static bool path_exists(const string &path)
{
    struct stat st;
    return stat(path.c_str(), &st) == 0;
}

static string first_existing_path(const vector<string> &candidates)
{
    for (const string &path : candidates)
    {
        if (path_exists(path))
            return path;
    }
    return candidates.empty() ? string() : candidates[0];
}

static const char *event_status_name(cl_int status)
{
    switch (status)
    {
    case CL_QUEUED:
        return "QUEUED";
    case CL_SUBMITTED:
        return "SUBMITTED";
    case CL_RUNNING:
        return "RUNNING";
    case CL_COMPLETE:
        return "COMPLETE";
    default:
        return "ERROR";
    }
}

static void debug_event_log(const string &message)
{
    cout << "[HOST-EVENT] " << message << endl;
    cout.flush();
}

static void debug_event_status(const string &label, cl_event event)
{
    cout << "[HOST-EVENT] " << label << " status: ";
    if (event == nullptr)
    {
        cout << "NULL" << endl;
        cout.flush();
        return;
    }

    cl_int status = 0;
    cl_int err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(status), &status, NULL);
    if (err != CL_SUCCESS)
    {
        cout << "clGetEventInfo ERROR " << err << endl;
    }
    else
    {
        cout << event_status_name(status) << " (" << status << ")" << endl;
    }
    cout.flush();
}

static void debug_event_wait_list(const string &label, cl_uint count, const cl_event *events)
{
    debug_event_log(label + " wait list count=" + to_string(count));
    for (cl_uint i = 0; i < count; i++)
    {
        debug_event_status(label + " wait[" + to_string(i) + "]", events[i]);
    }
}

static void debug_float_tensor_stats(const string &label, const float *data, size_t count)
{
    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t finite_count = 0;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float max_abs = 0.0f;
    size_t min_index = 0;
    size_t max_index = 0;
    size_t max_abs_index = 0;

    for (size_t i = 0; i < count; i++)
    {
        const float value = data[i];
        if (std::isnan(value))
        {
            nan_count++;
            continue;
        }
        if (std::isinf(value))
        {
            inf_count++;
            continue;
        }

        if (finite_count == 0)
        {
            min_val = value;
            max_val = value;
            max_abs = std::fabs(value);
            min_index = i;
            max_index = i;
            max_abs_index = i;
        }
        else
        {
            if (value < min_val)
            {
                min_val = value;
                min_index = i;
            }
            if (value > max_val)
            {
                max_val = value;
                max_index = i;
            }
            const float abs_value = std::fabs(value);
            if (abs_value > max_abs)
            {
                max_abs = abs_value;
                max_abs_index = i;
            }
        }
        finite_count++;
    }

    cout << "[HOST-DATA] " << label
         << " count=" << count
         << " finite=" << finite_count
         << " nan=" << nan_count
         << " inf=" << inf_count;
    if (finite_count > 0)
    {
        cout << " min=" << min_val
             << " min_index=" << min_index
             << " max=" << max_val
             << " max_index=" << max_index
             << " max_abs=" << max_abs
             << " max_abs_index=" << max_abs_index;
    }
    cout << endl;

    const size_t preview_count = (count < 8) ? count : 8;
    cout << "[HOST-DATA] " << label << " first_values:";
    for (size_t i = 0; i < preview_count; i++)
    {
        cout << " " << data[i];
    }
    cout << endl;
    cout.flush();
}

// -----------------------------host side helper functions--------------------------------------
// TODO: modify to load weight from quantized model
/*
static void init_random_int(int *data, size_t count, int min_value, int max_value)
{
    int range = max_value - min_value + 1;
    for (size_t i = 0; i < count; i++)
    {
        data[i] = min_value + (rand() % range);
    }
}
// Modified from `posix_memalign`
// allocate 4096-byte aligned host memory
static void *host_aligned_alloc(size_t size_in_bytes)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, size_in_bytes))
    {
        return nullptr;
    }
    return ptr;
}

static bool load_txt_values_as_int(const string &filename, int *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int>(roundf(value));
    }

    return true;
}

static bool load_txt_values_as_float(const string &filename, float *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        if (!(fin >> dst[i]))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
    }

    return true;
}

static bool load_txt_values_as_int8(const string &filename, int8_t *dst, size_t count)
{
    ifstream fin(filename);
    if (!fin.is_open())
    {
        return false;
    }

    for (size_t i = 0; i < count; i++)
    {
        float value = 0.0f;
        if (!(fin >> value))
        {
            cerr << "HOST-Error: Failed to parse value " << i << " from " << filename << endl;
            return false;
        }
        dst[i] = static_cast<int8_t>(roundf(value));
    }

    return true;
}

static void pack_mlp_weights(
    const int8_t W1[EMBED_DIM][MLP_DIM],
    const int8_t W2[MLP_DIM][EMBED_DIM],
    int8_t W1_packed[MLP_W1_CHUNKS][32],
    int8_t W2_packed[MLP_W2_CHUNKS][32])
{
    memset(W1_packed, 0, MLP_W1_CHUNKS * 32 * sizeof(int8_t));
    memset(W2_packed, 0, MLP_W2_CHUNKS * 32 * sizeof(int8_t));

    for (int h = 0; h < MLP_DIM; h++)
    {
        for (int d = 0; d < EMBED_DIM; d++)
        {
            const int flat = h * EMBED_DIM + d;
            W1_packed[flat / 32][flat % 32] = W1[d][h];
        }
    }

    for (int d = 0; d < EMBED_DIM; d++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            const int flat = d * MLP_DIM + h;
            W2_packed[flat / 32][flat % 32] = W2[h][d];
        }
    }
}

static void pack_linear_layernorm_weight(
    const int8_t weight[EMBED_DIM][EMBED_DIM],
    int8_t packed[EMBED_DIM][EMBED_DIM])
{
    for (int out = 0; out < EMBED_DIM; out++)
    {
        for (int in = 0; in < EMBED_DIM; in++)
        {
            packed[out][in] = weight[in][out];
        }
    }
}

static void unpack_mlp_output_tile(
    const uint32_t in_chunks[MLP_OUTPUT_CHUNKS][MLP_CHUNK_U32],
    int token_start,
    int out[SEQ_LEN][EMBED_DIM])
{
    for (int local_token = 0; local_token < MLP_KERNEL_SEQ_LEN; local_token++)
    {
        const int token = token_start + local_token;
        if (token >= SEQ_LEN)
            break;

        for (int d = 0; d < EMBED_DIM; d++)
        {
            const int flat = local_token * EMBED_DIM + d;
            float value = 0.0f;
            uint32_t bits = in_chunks[flat / MLP_CHUNK_U32][flat % MLP_CHUNK_U32];
            memcpy(&value, &bits, sizeof(float));
            out[token][d] = static_cast<int>(roundf(value));
        }
    }
}

// TODO: not implement here
static int gelu_int(int x)
{
    float xf = static_cast<float>(x);
    float y = 0.5f * xf * (1.0f + tanhf(0.79788456f * (xf + 0.044715f * xf * xf * xf)));
    return static_cast<int>(roundf(y));
}

//  TODO: not implement here
static void layernorm_host(int in[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    const float eps = 1.0e-5f;
    const float output_scale = 16.0f;

    for (int i = 0; i < SEQ_LEN; i++)
    {
        float mean = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
            mean += static_cast<float>(in[i][d]);
        mean /= static_cast<float>(EMBED_DIM);

        float var = 0.0f;
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float diff = static_cast<float>(in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<float>(EMBED_DIM);

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float norm = (static_cast<float>(in[i][d]) - mean) * inv_std;
            out[i][d] = static_cast<int>(roundf(norm * output_scale));
        }
    }
}

// Quantize attention outputs with one activation scale per token across all heads.
static void quantize_attention_heads(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN])
{

    for (int i = 0; i < SEQ_LEN; i++)
    {
        float max_abs = 0.0f;
        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float value = fabsf(attention_heads[h][i][d]);
                if (value > max_abs)
                    max_abs = value;
            }
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        attention_scales[i] = scale;

        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float scaled = attention_heads[h][i][d] / scale;
                int q = static_cast<int>(roundf(scaled));
                if (q > 127)
                    q = 127;
                else if (q < -128)
                    q = -128;
                attention_heads_q[h][i][d] = static_cast<int8_t>(q);
            }
        }
    }

}

static void quantize_attention_heads_mp(
    float attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t attention_heads_q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    float attention_scales[SEQ_LEN])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        float max_abs = 0.0f;

        // =========================
        // Find max abs (reduction)
        // =========================
        for (int h = 0; h < NUM_HEADS; h++)
        {
            #pragma omp simd reduction(max:max_abs)
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float value = fabsf(attention_heads[h][i][d]);
                if (value > max_abs)
                    max_abs = value;
            }
        }

        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        attention_scales[i] = scale;

        // =========================
        // Quantization
        // =========================
        for (int h = 0; h < NUM_HEADS; h++)
        {
            #pragma omp simd
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float scaled = attention_heads[h][i][d] / scale;
                int q = static_cast<int>(roundf(scaled));

                if (q > 127)
                    q = 127;
                else if (q < -128)
                    q = -128;

                attention_heads_q[h][i][d] = static_cast<int8_t>(q);
            }
        }
    }
}

// [12][197][64] -> [197][768]
static void concat_attention_heads(int attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
                                   int attention_concat[SEQ_LEN][EMBED_DIM])
{
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < SEQ_LEN; i++)
            for (int d = 0; d < HEAD_DIM; d++)
                attention_concat[i][h * HEAD_DIM + d] = attention_heads[h][i][d];
}

static void concat_attention_heads_mp(int8_t attention_heads[NUM_HEADS][SEQ_LEN][HEAD_DIM],
                                   int8_t attention_concat[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for collapse(2)
    for (int h = 0; h < NUM_HEADS; h++)
        for (int i = 0; i < SEQ_LEN; i++)
            memcpy(&attention_concat[i][h * HEAD_DIM], &attention_heads[h][i][0], sizeof(int8_t) * HEAD_DIM);
            // error may caused by here
            // original error code: memcpy(attention_concat + i * EMBED_DIM + h * HEAD_DIM, attention_heads + h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM, sizeof(int) * HEAD_DIM);
            //for (int d = 0; d < HEAD_DIM; d++)
            //attention_concat[i][h * HEAD_DIM + d] = attention_heads[h][i][d];
}

//  TODO: not implement here?
static void residual_add(int a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = a[i][d] + b[i][d];
}

static void residual_add(float a[SEQ_LEN][EMBED_DIM], int b[SEQ_LEN][EMBED_DIM], int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<int>(roundf(a[i][d] + static_cast<float>(b[i][d])));
}

static void residual_add_mp(
    const int a[SEQ_LEN][EMBED_DIM],
    const int b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            out[i][d] = a[i][d] + b[i][d];
        }
    }
}

static void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const int   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float sum = a[i][d] +  static_cast<float>(b[i][d]);
            out[i][d] = static_cast<int>(roundf(sum));
        }
    }
}

static void residual_add_mp(
    const float a[SEQ_LEN][EMBED_DIM],
    const float   b[SEQ_LEN][EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            float sum = a[i][d] +  static_cast<float>(b[i][d]);
            out[i][d] = static_cast<int>(roundf(sum));
        }
    }
}

// block with int convert into float
static void copy_int_to_float(int in[SEQ_LEN][EMBED_DIM], float out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            out[i][d] = static_cast<float>(in[i][d]);
}
static void copy_int_to_float_mp(
    const int in[SEQ_LEN][EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        #pragma omp simd
        for (int d = 0; d < EMBED_DIM; d++)
        {
            out[i][d] = static_cast<float>(in[i][d]);
        }
    }
}

// TODO: delete this?
static void linear_quant_host_int_input(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float input_scale = scale_input[i] * scale_weight[o];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * input_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }
}

// attention output projection
static void linear_quant_host_fp32_input(
    int in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float weight_scale = scale_weight[o] * scale_input[i];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * weight_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }

}

static void linear_quant_host_fp32_input_mp(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    float out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel for
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float weight_scale = scale_weight[o] * scale_input[i];

            #pragma omp simd reduction(+:acc)
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += (float)in[i][d] *
                       (float)weight[d][o] *
                       weight_scale;
            }

            out[i][o] = (int)roundf(acc);
        }
    }
}

static void mlp_quant_host(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t W1[EMBED_DIM][MLP_DIM],
    float b1[MLP_DIM],
    float scale_w1[MLP_DIM],
    int8_t W2[MLP_DIM][EMBED_DIM],
    float b2[EMBED_DIM],
    float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    vector<float> hidden(MLP_DIM);

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            float acc = b1[h];
            const float scale = scale_input[i] * scale_w1[h];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(W1[d][h]) * scale;
            }
            float gelu = 0.5f * acc * (1.0f + tanhf(0.79788456f * (acc + 0.044715f * acc * acc * acc)));
            hidden[h] = gelu;
        }

        for (int d = 0; d < EMBED_DIM; d++)
        {
            float acc = b2[d];
            const float scale = scale_w2[d];
            for (int h = 0; h < MLP_DIM; h++)
            {
                acc += hidden[h] * static_cast<float>(W2[h][d]) * scale;
            }
            out[i][d] = static_cast<int>(roundf(acc));
        }
    }

}

static void mlp_quant_host_mp(
    const int8_t in[SEQ_LEN][EMBED_DIM],
    const float scale_input[SEQ_LEN],
    const int8_t W1[EMBED_DIM][MLP_DIM],
    const float b1[MLP_DIM],
    const float scale_w1[MLP_DIM],
    const int8_t W2[MLP_DIM][EMBED_DIM],
    const float b2[EMBED_DIM],
    const float scale_w2[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    #pragma omp parallel
    {
        std::vector<float> hidden(MLP_DIM);

        #pragma omp for
        for (int i = 0; i < SEQ_LEN; i++)
        {
            // Layer 1
            for (int h = 0; h < MLP_DIM; h++)
            {
                float acc = b1[h];
                const float scale = scale_input[i] * scale_w1[h];

                #pragma omp simd reduction(+:acc)
                for (int d = 0; d < EMBED_DIM; d++)
                {
                    acc += static_cast<float>(in[i][d]) *
                        static_cast<float>(W1[d][h]) *
                        scale;
                }

                float x = acc;
                hidden[h] = 0.5f * x *
                    (1.0f + tanhf(0.79788456f *
                    (x + 0.044715f * x * x * x)));
            }

            // Layer 2
            for (int d = 0; d < EMBED_DIM; d++)
            {
                float acc = b2[d];
                const float scale = scale_w2[d];

                #pragma omp simd reduction(+:acc)
                for (int h = 0; h < MLP_DIM; h++)
                {
                    acc += hidden[h] *
                        static_cast<float>(W2[h][d]) *
                        scale;
                }

                out[i][d] = static_cast<int>(roundf(acc));
            }
        }
    }
}

// TODO: not implement here, not used?
static void mlp_host(int in[SEQ_LEN][EMBED_DIM],
                     int W1[EMBED_DIM][MLP_DIM], int b1[MLP_DIM],
                     int W2[MLP_DIM][EMBED_DIM], int b2[EMBED_DIM],
                     int out[SEQ_LEN][EMBED_DIM])
{
    vector<int> hidden(MLP_DIM);
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            long long acc = b1[h];
            for (int d = 0; d < EMBED_DIM; d++)
                acc += static_cast<long long>(in[i][d]) * W1[d][h];
            hidden[h] = gelu_int(static_cast<int>(acc >> 8));
        }
        for (int d = 0; d < EMBED_DIM; d++)
        {
            long long acc = b2[d];
            for (int h = 0; h < MLP_DIM; h++)
                acc += static_cast<long long>(hidden[h]) * W2[h][d];
            out[i][d] = static_cast<int>(acc >> 8);
        }
    }
}
*/

static void pack_mlp_weights(
    const int8_t W1[EMBED_DIM][MLP_DIM],
    const int8_t W2[MLP_DIM][EMBED_DIM],
    int8_t W1_packed[MLP_W1_CHUNKS][32],
    int8_t W2_packed[MLP_W2_CHUNKS][32])
{
    memset(W1_packed, 0, MLP_W1_CHUNKS * 32 * sizeof(int8_t));
    memset(W2_packed, 0, MLP_W2_CHUNKS * 32 * sizeof(int8_t));

    for (int h = 0; h < MLP_DIM; h++)
    {
        for (int d = 0; d < EMBED_DIM; d++)
        {
            const int flat = h * EMBED_DIM + d;
            W1_packed[flat / 32][flat % 32] = W1[d][h];
        }
    }

    for (int d = 0; d < EMBED_DIM; d++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            const int flat = d * MLP_DIM + h;
            W2_packed[flat / 32][flat % 32] = W2[h][d];
        }
    }
}

static void pack_linear_layernorm_weight(
    const int8_t weight[EMBED_DIM][EMBED_DIM],
    int8_t packed[EMBED_DIM][EMBED_DIM])
{
    for (int out = 0; out < EMBED_DIM; out++)
    {
        for (int in = 0; in < EMBED_DIM; in++)
        {
            packed[out][in] = weight[in][out];
        }
    }
}
//-------------------------------------------------------------------------------------

// ----------------------------------Same as previous version of host.cpp----------------------------------
// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char *argv[])
{
    cout << endl;

// ============================================================================
// Step 1: Check Command Line Arguments
// ============================================================================
//    o) argv[1] Platfrom Vendor
//    o) argv[2] Device Name
//    o) argv[3] XCLBIN file
// ============================================================================
// INFO: sample usage:
//   ./host Xilinx xilinx_u280_xdma_201920_3 spatial_encoder.xclbin
//   ./host Xilinx xilinx_u280_xdma_201920_3 spatial_encoder.xclbin ../params/all_params/SpatialTransformer ../params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    if (argc < 4 || argc > 6)
    {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File> [SpatialTransformer_Params_Dir] [Input_X_File]" << endl
             << endl;
        return EXIT_FAILURE;
    }

    const char *Target_Platform_Vendor = argv[1];
    const char *Target_Device_Name = argv[2];
    const char *xclbinFilename = argv[3];
    const string spatial_params_root = (argc >= 5)
                                           ? string(argv[4])
                                           : first_existing_path({"../params/all_params/SpatialTransformer",
                                                                  "ViViT/params/all_params/SpatialTransformer",
                                                                  "params/all_params/SpatialTransformer",
                                                                  "/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer",
                                                                  "/home/undergraduate/rjchen25/ViViT/params/SpatialTransformer"});
    const string x_input_file = (argc >= 6)
                                    ? string(argv[5])
                                    : first_existing_path({"../params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "ViViT/params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "/home/undergraduate/rjchen25/ViViT/params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt"});
    cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
    cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;
    cout << "HOST-Info: Params_root       : " << spatial_params_root << endl;
    cout << "HOST-Info: Input_X_file      : " << x_input_file << endl;
    if (!path_exists(spatial_params_root))
    {
        cout << "HOST-Error: SpatialTransformer params directory does not exist: " << spatial_params_root << endl
             << endl;
        return EXIT_FAILURE;
    }
    if (!path_exists(x_input_file))
    {
        cout << "HOST-Error: Input X file does not exist: " << x_input_file << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ============================================================================
    // Step 2: Detect Target Platform and Target Device in a system.
    //         Create Context and Command Queue.
    // ============================================================================
    // Variables:
    //   o) Target_Platform_Vendor[] - defined as main() input argument
    //   o) Target_Device_Name[]     - defined as main() input argument
    //
    // After that
    //   o) Create a Context
    //   o) Create a Command Queue
    // ============================================================================
    cout << endl;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
    cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    cl_uint ui;

    cl_platform_id *Platform_IDs;
    cl_uint Nb_Of_Platforms;
    cl_platform_id Target_Platform_ID;
    bool Platform_Detected;
    char *platform_info;

    cl_device_id *Device_IDs;
    cl_uint Nb_Of_Devices;
    cl_device_id Target_Device_ID;
    bool Device_Detected;
    char *device_info;

    cl_context Context;
    cl_command_queue Command_Queue;

    cl_int errCode;
    size_t size;

    // ------------------------------------------------------------------------------------
    // Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
    // ------------------------------------------------------------------------------------

    // Get the number of platforms
    // ..................................................
    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
#endif

    // Allocate memory to store platforms
    // ..................................................
    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
    if (!Platform_IDs)
    {
        cout << endl
             << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Get and store all PLATFORMS
    // ..................................................
    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
    // ....................................................................................
    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++)
    {

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the size of the Platofrm parameter " << "CL_PLATFORM_VENDOR" << " value " << endl
                 << endl;
            return EXIT_FAILURE;
        }

        platform_info = new char[size];
        if (!platform_info)
        {
            cout << endl
                 << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        // Check if the current platform matches Target_Platform_Vendor
        // .............................................................
        if (strcmp(platform_info, Target_Platform_Vendor) == 0)
        {
            Platform_Detected = true;
            Target_Platform_ID = Platform_IDs[ui];
#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl
                 << endl;
#endif
        }
    }

    if (Platform_Detected == false)
    {
        cout << endl
             << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------------------
    // Step 2.2:  Get All Devices for selected platform Target_Platform_ID
    //            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
    // ------------------------------------------------------------------------------------

    // Get the Number of Devices
    // ............................................................................
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available Devices" << endl
             << endl;
        return EXIT_FAILURE;
    }
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of available devices  : " << Nb_Of_Devices << endl;
#endif

    Device_IDs = new cl_device_id[Nb_Of_Devices];
    if (!Device_IDs)
    {
        cout << endl
             << "HOST-Error: Out of Memory during memory allocation for Device_IDs" << endl
             << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get available Devices" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Search for CL_DEVICE_NAME = Target_Device_Name
    // ............................................................................
    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++)
    {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        device_info = new char[size];
        if (!device_info)
        {
            cout << endl
                 << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl
                 << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        // Check if the current device matches Target_Device_Name
        // ............................................................................
        if (strcmp(device_info, Target_Device_Name) == 0)
        {
            Device_Detected = true;
            Target_Device_ID = Device_IDs[ui];
        }
    }

    if (Device_Detected == false)
    {
        cout << endl
             << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl
             << endl;
        return EXIT_FAILURE;
    }
    else
    {
#ifdef ALL_MESSAGES
        cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl
             << endl;
#endif
    }

// ------------------------------------------------------------------------------------
// Step 2.3: Create Context
// ------------------------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Context ... " << endl;
#endif
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Context" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ------------------------------------------------------------------------------------
// Step 2.4: Create Command Queue (commands are executed in-order)
// ------------------------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Command Queue ... " << endl;
#endif
    Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Command Queue" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ============================================================================
// Step 3: Create Program and Kernel
// ============================================================================
//   o) Create a Program from a Binary File and Build it
//   o) Create a Kernel
// ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------------------
    // Step 3.1: Load Binary File from a disk to Memory
    // ------------------------------------------------------------------
    unsigned char *xclbin_Memory;
    int program_length;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
#endif

    program_length = loadFile2Memory(xclbinFilename, (char **)&xclbin_Memory);
    if (program_length < 0)
    {
        cout << endl
             << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------
    // Step 3.2: Create a program using a Binary File
    // ------------------------------------------------------------
    size_t Program_Length_in_Bytes;
    cl_program Program;
    cl_int Binary_Status;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Program with Binary ..." << endl;
#endif
    Program_Length_in_Bytes = program_length;
    Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
                                        (const unsigned char **)&xclbin_Memory, &Binary_Status, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Program from a Binary" << endl
             << endl;
        return EXIT_FAILURE;
    }

// ----------------------------------------------------------------------
// Step 3.3: Build (compiles and links) a program executable from binary
// ----------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Building the Program ..." << endl;
#endif

    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to build a Program Executable" << endl
             << endl;
        return EXIT_FAILURE;
    }
    // ------------------------Start to be defferent from sample host.cpp------------------------------
    // -------------------------------------------------------------
    // Step 3.4: Create Kernels
    // -------------------------------------------------------------
    cl_kernel K_layernorm[2], K_linear_qkv_multihead, K_attention_opt[NUM_HEADS],
        K_projection_out, K_residual, K_mlp_q;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
#endif
    for (int idx = 0; idx < 2; idx++)
    {
        K_layernorm[idx] = clCreateKernel(Program, "layernorm", &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to create K_layernorm" << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: linear_qkv_multihead ..." << endl;
#endif
    K_linear_qkv_multihead = clCreateKernel(Program, "linear_qkv_multihead", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_linear_qkv_multihead" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Kernels: attention_opt[0.." << (FRAME * NUM_HEADS - 1) << "] ..." << endl;
#endif
    for (int idx = 0; idx < NUM_HEADS; idx++)
    {
        K_attention_opt[idx] = clCreateKernel(Program, "attention_opt", &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to create K_attention_opt[" << idx << "]" << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: projection_out ..." << endl;
#endif
    K_projection_out = clCreateKernel(Program, "projection_out", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_projection_out" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: residual ..." << endl;
#endif
    K_residual = clCreateKernel(Program, "residual", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_residual" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: mlp_q ..." << endl;
#endif
    K_mlp_q = clCreateKernel(Program, "mlp_q", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_mlp_q" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    //   o) Allocate and initialize spatial encoder input and weights
    //   o) Allocate Q/K/V, attention, MLP, and final output arrays
    //   o) Create Buffers in Global Memory to store FPGA data
    // ================================================================
    // allocate host memory for spatial encoder input, weights, and outputs
    float (*X)[EMBED_DIM]; // FRAME * SEQ_LEN * EMBED_DIM

    // model params
    float *LayerNorm_gamma;                // EMBED_DIM
    float *LayerNorm_beta;                 // EMBED_DIM
    int8_t (*Wq)[EMBED_DIM];               // EMBED_DIM * EMBED_DIM (EMBED_DIM * NUM_HEADS * HEAD_DIM)
    int8_t (*Wq_Linear_Packed)[EMBED_DIM]; // EMBED_DIM * EMBED_DIM, output-major for fused LN+QKV
    float *bq;                             // HEAD_NUM * HEAD_DIM
    float *scale_wq;                       // HEAD_NUM * HEAD_DIM
    int8_t (*Wk)[EMBED_DIM];               // EMBED_DIM * EMBED_DIM (EMBED_DIM * NUM_HEADS * HEAD_DIM)
    int8_t (*Wk_Linear_Packed)[EMBED_DIM]; // EMBED_DIM * EMBED_DIM, output-major for fused LN+QKV
    float *bk;                             // HEAD_NUM * HEAD_DIM
    float *scale_wk;                       // HEAD_NUM * HEAD_DIM
    int8_t (*Wv)[EMBED_DIM];               // EMBED_DIM * EMBED_DIM (EMBED_DIM * NUM_HEADS * HEAD_DIM)
    int8_t (*Wv_Linear_Packed)[EMBED_DIM]; // EMBED_DIM * EMBED_DIM, output-major for fused LN+QKV
    float *bv;                             // HEAD_NUM * HEAD_DIM
    float *scale_wv;                       // HEAD_NUM * HEAD_DIM
    // attention ouput projection weight
    int8_t (*Wout)[EMBED_DIM];   // EMBED_DIM * EMBED_DIM (NUM_HEADS * HEAD_DIM * EMBED_DIM)
    float *bout;                 // HEAD_NUM * HEAD_DIM
    float *scale_wout;           // HEAD_NUM * HEAD_DIM
    float *LayerNorm1_gamma;     // EMBED_DIM
    float *LayerNorm1_beta;      // EMBED_DIM
    int8_t (*MLP_W1)[MLP_DIM];   // EMBED_DIM * MLP_DIM
    float *MLP_b1;               // MLP_DIM
    float *MLP_scale_w1;         // MLP_DIM
    int8_t (*MLP_W2)[EMBED_DIM]; // MLP_DIM * EMBED_DIM
    float *MLP_b2;               // EMBED_DIM
    float *MLP_scale_w2;         // EMBED_DIM

    // intermediate value
    float *LayerNorm_scaler;           // FRAME * SEQ_LEN
    int8_t (*X_norm)[EMBED_DIM];       // FRAME * SEQ_LEN * EMBED_DIM
    float (*Q)[SEQ_LEN][HEAD_DIM];     // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    float (*K_out)[SEQ_LEN][HEAD_DIM]; // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    float (*V)[SEQ_LEN][HEAD_DIM];     // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM

    float (*Attention_Out)[SEQ_LEN][HEAD_DIM];          // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    int8_t (*Attention_Out_Quant)[SEQ_LEN][HEAD_DIM];   // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    float *Attention_Quant_Scales;                      // FRAME * SEQ_LEN
    int8_t (*Attention_Concat)[EMBED_DIM];              // FRAME * SEQ_LEN * EMBED_DIM
    float (*Attention_Projected)[EMBED_DIM];            // FRAME * SEQ_LEN * EMBED_DIM
    float (*After_Attention_Residual_Float)[EMBED_DIM]; // FRAME * SEQ_LEN * EMBED_DIM
    uint32_t (*MLP_Output_Packed)[MLP_CHUNK_U32];       // SEQ_LEN * EMBED_DIM fp32 bits
    int8_t (*MLP_W1_Packed)[32];                        // MLP_DIM * EMBED_DIM int8
    int8_t (*MLP_W2_Packed)[32];                        // EMBED_DIM * MLP_DIM int8

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------------------
    // Step 4.1: Allocate and initialize spatial encoder arrays
    // ------------------------------------------------------------------
    void *ptr = nullptr;
    // config memory
    X = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));

    LayerNorm_gamma = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm_beta = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wq = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    Wq_Linear_Packed = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bq = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wq = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wk = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    Wk_Linear_Packed = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bk = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wk = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wv = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    Wv_Linear_Packed = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bv = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wv = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    Wout = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
    bout = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    scale_wout = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm1_gamma = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    LayerNorm1_beta = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    MLP_W1 = reinterpret_cast<int8_t (*)[MLP_DIM]>(host_aligned_alloc(EMBED_DIM * MLP_DIM * sizeof(int8_t)));
    MLP_b1 = reinterpret_cast<float *>(host_aligned_alloc(MLP_DIM * sizeof(float)));
    MLP_scale_w1 = reinterpret_cast<float *>(host_aligned_alloc(MLP_DIM * sizeof(float)));
    MLP_W2 = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(MLP_DIM * EMBED_DIM * sizeof(int8_t)));
    MLP_b2 = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    MLP_scale_w2 = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));

    X_norm = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(int8_t)));
    LayerNorm_scaler = reinterpret_cast<float *>(host_aligned_alloc(FRAME * SEQ_LEN * sizeof(float)));
    Q = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    K_out = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    V = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    Attention_Out = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float)));
    Attention_Out_Quant = reinterpret_cast<int8_t (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int8_t)));
    Attention_Quant_Scales = reinterpret_cast<float *>(host_aligned_alloc(FRAME * SEQ_LEN * sizeof(float)));

    Attention_Concat = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(int8_t)));
    Attention_Projected = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    After_Attention_Residual_Float = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    MLP_Output_Packed = reinterpret_cast<uint32_t (*)[MLP_CHUNK_U32]>(host_aligned_alloc(MLP_OUTPUT_CHUNKS * MLP_CHUNK_U32 * sizeof(uint32_t)));
    MLP_W1_Packed = reinterpret_cast<int8_t (*)[32]>(host_aligned_alloc(MLP_W1_CHUNKS * 32 * sizeof(int8_t)));
    MLP_W2_Packed = reinterpret_cast<int8_t (*)[32]>(host_aligned_alloc(MLP_W2_CHUNKS * 32 * sizeof(int8_t)));

    if (!X || !LayerNorm_gamma || !LayerNorm_beta || !X_norm || !LayerNorm_scaler ||
        !Wq || !bq || !scale_wq || !Wk || !bk || !scale_wk || !Wv || !bv || !scale_wv ||
        !Wq_Linear_Packed || !Wk_Linear_Packed || !Wv_Linear_Packed ||
        !Wout || !bout || !scale_wout || !LayerNorm1_gamma || !LayerNorm1_beta ||
        !Q || !K_out || !V || !Attention_Out || !Attention_Out_Quant || !Attention_Quant_Scales ||
        !Attention_Concat || !Attention_Projected ||
        !After_Attention_Residual_Float ||
        !MLP_Output_Packed || !MLP_W1_Packed || !MLP_W2_Packed ||
        !MLP_W1 || !MLP_b1 || !MLP_scale_w1 || !MLP_W2 || !MLP_b2 || !MLP_scale_w2)
    {
        cout << endl
             << "HOST-Error: Out of Memory during spatial encoder allocation" << endl
             << endl;
        return EXIT_FAILURE;
    }

    // Keep the original random initialization flow for fallback/debug reference.
    /*
    srand(0);
    init_random_int(&X[0][0], SEQ_LEN * EMBED_DIM, -2, 2);
    init_random_int(&Wq[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bq, EMBED_DIM, -1, 1);
    init_random_int(&Wk[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bk, EMBED_DIM, -1, 1);
    init_random_int(&Wv[0][0], EMBED_DIM * EMBED_DIM, -1, 1);
    init_random_int(bv, EMBED_DIM, -1, 1);
    init_random_int(&MLP_W1[0][0], EMBED_DIM * MLP_DIM, -1, 1);
    init_random_int(MLP_b1, MLP_DIM, -1, 1);
    init_random_int(&MLP_W2[0][0], MLP_DIM * EMBED_DIM, -1, 1);
    init_random_int(MLP_b2, EMBED_DIM, -1, 1);
    */
    if (!load_txt_values_as_float(x_input_file, &X[0][0], FRAME * SEQ_LEN * EMBED_DIM))
    {
        cout << endl
             << "HOST-Error: Failed to load encoder input X from " << x_input_file << endl
             << endl;
        return EXIT_FAILURE;
    }
    cout << "HOST-Info: Loaded spatial encoder input X from " << x_input_file << endl;

    cout << endl;

// ------------------------------------------------------------------
// Step 4.2: Create Buffers in Global Memory to store data
//             o) GlobMem_BUF_X                 - stores fp32 attention input X (R)
//             o) GlobMem_BUF_LayerNorm_Gamma   - stores LayerNorm gamma (R)
//             o) GlobMem_BUF_LayerNorm_Beta    - stores LayerNorm beta (R)
//             o) GlobMem_BUF_X_norm            - stores int8 LayerNorm output (R/W)
//             o) GlobMem_BUF_LayerNorm_Scaler  - stores LayerNorm scaler[SEQ_LEN] (R/W)
//             o) GlobMem_BUF_Wq/bq/scale_wq    - stores Q quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Wk/bk/scale_wk    - stores K quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Wv/bv/scale_wv    - stores V quantized weights, bias, and scaler (R)
//             o) GlobMem_BUF_Q/K/V             - stores fp32 Q/K/V outputs (R/W)
//             o) GlobMem_BUF_Attention_Out     - stores attention output from all heads (R/W)
// ------------------------------------------------------------------
// TODO: Add mlp weights and bias buffer
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
#endif

    // buffer relative to X
    cl_mem GlobMem_BUF_X, GlobMem_BUF_X_norm, GlobMem_BUF_Attention_Out,
        GlobMem_BUF_Attention_Projected,
        GlobMem_BUF_After_Attention_Residual_Float,
        GlobMem_BUF_LayerNorm_Scaler, GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V;

    cl_mem GlobMem_BUF_frame_X[FRAME], GlobMem_BUF_frame_X_norm[FRAME], GlobMem_BUF_frame_LayerNorm_Scaler[FRAME],
        GlobMem_BUF_frame_Q[FRAME], GlobMem_BUF_frame_K[FRAME], GlobMem_BUF_frame_V[FRAME],
        GlobMem_BUF_frame_Attention_Out[FRAME], GlobMem_BUF_frame_Attention_Projected[FRAME],
        GlobMem_BUF_frame_After_Attention_Residual_Float[FRAME];

    cl_mem GlobMem_BUF_LayerNorm_Gamma, GlobMem_BUF_LayerNorm_Beta,
        GlobMem_BUF_LayerNorm1_Gamma, GlobMem_BUF_LayerNorm1_Beta,
        GlobMem_BUF_Wq, GlobMem_BUF_bq, GlobMem_BUF_scale_wq,
        GlobMem_BUF_Wk, GlobMem_BUF_bk, GlobMem_BUF_scale_wk,
        GlobMem_BUF_Wv, GlobMem_BUF_bv, GlobMem_BUF_scale_wv,
        GlobMem_BUF_Wout, GlobMem_BUF_bout, GlobMem_BUF_scale_wout,
        GlobMem_BUF_MLP_Output,
        GlobMem_BUF_MLP_W1, GlobMem_BUF_MLP_W2,
        GlobMem_BUF_MLP_b1, GlobMem_BUF_MLP_b2,
        GlobMem_BUF_MLP_scale_w1, GlobMem_BUF_MLP_scale_w2;

    GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), X, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_X" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_X_norm = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * SEQ_LEN * EMBED_DIM * sizeof(int8_t), X_norm, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_X_norm" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_LayerNorm_Scaler = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * SEQ_LEN * sizeof(float), LayerNorm_scaler, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Scaler" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Q = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), Q, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Q" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_K = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), K_out, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_K" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_V = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), V, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_V" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Attention_Out = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), Attention_Out, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Attention_Out" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Attention_Projected = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), Attention_Projected, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Attention_Projected" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_After_Attention_Residual_Float = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), After_Attention_Residual_Float, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_After_Attention_Residual_Float" << endl
             << endl;
        return EXIT_FAILURE;
    }
    // ================================================
    //                  Ceate SubBuffer
    // ================================================
    size_t frame_size_X = SEQ_LEN * EMBED_DIM * sizeof(float);
    size_t frame_size_X_norm = SEQ_LEN * EMBED_DIM * sizeof(int8_t);
    size_t frame_size_scaler = SEQ_LEN * sizeof(float);
    size_t frame_size_QKV = NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t frame_size_Attention_Out = NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
    size_t frame_size_Attention_Projected = SEQ_LEN * EMBED_DIM * sizeof(float);
    size_t frame_size_After_Attention_Residual_Float = SEQ_LEN * EMBED_DIM * sizeof(float);
    for (int f = 0; f < FRAME; f++)
    {
        cl_buffer_region region;

        // ---------------- X ----------------
        region.origin = f * frame_size_X;
        region.size = frame_size_X;
        GlobMem_BUF_frame_X[f] = clCreateSubBuffer(
            GlobMem_BUF_X, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_X creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- X_norm ----------------
        region.origin = f * frame_size_X_norm;
        region.size = frame_size_X_norm;
        GlobMem_BUF_frame_X_norm[f] = clCreateSubBuffer(
            GlobMem_BUF_X_norm, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_X_norm creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- LayerNorm scaler ----------------
        region.origin = f * frame_size_scaler;
        region.size = frame_size_scaler;
        GlobMem_BUF_frame_LayerNorm_Scaler[f] = clCreateSubBuffer(
            GlobMem_BUF_LayerNorm_Scaler, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_LayerNorm_Scaler creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- Q/K/V full frame ----------------
        region.origin = f * frame_size_QKV;
        region.size = frame_size_QKV;
        GlobMem_BUF_frame_Q[f] = clCreateSubBuffer(
            GlobMem_BUF_Q, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_Q creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        GlobMem_BUF_frame_K[f] = clCreateSubBuffer(
            GlobMem_BUF_K, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_K creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        GlobMem_BUF_frame_V[f] = clCreateSubBuffer(
            GlobMem_BUF_V, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_V creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- Attention Out ----------------
        region.origin = f * frame_size_Attention_Out;
        region.size = frame_size_Attention_Out;
        GlobMem_BUF_frame_Attention_Out[f] = clCreateSubBuffer(
            GlobMem_BUF_Attention_Out, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_Attention_Out creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- Attention Projected Float ----------------
        region.origin = f * frame_size_Attention_Projected;
        region.size = frame_size_Attention_Projected;
        GlobMem_BUF_frame_Attention_Projected[f] = clCreateSubBuffer(
            GlobMem_BUF_Attention_Projected, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_Attention_Projected creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------- Attention Residual Float ----------------
        region.origin = f * frame_size_After_Attention_Residual_Float;
        region.size = frame_size_After_Attention_Residual_Float;
        GlobMem_BUF_frame_After_Attention_Residual_Float[f] = clCreateSubBuffer(
            GlobMem_BUF_After_Attention_Residual_Float, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_After_Attention_Residual_Float creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ===================== below can be removed =======================
        /*
        // ---------------- Q ----------------
        region.origin = f * frame_size_QKV;
        region.size   = frame_size_QKV;
        GlobMem_BUF_frame_Q[f] = clCreateSubBuffer(
            GlobMem_BUF_Q, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        // ---------------- K ----------------
        region.origin = f * frame_size_QKV;
        region.size   = frame_size_QKV;
        GlobMem_BUF_frame_K[f] = clCreateSubBuffer(
            GlobMem_BUF_K, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        // ---------------- V ----------------
        region.origin = f * frame_size_QKV;
        region.size   = frame_size_QKV;
        GlobMem_BUF_frame_V[f] = clCreateSubBuffer(
            GlobMem_BUF_V, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        // ---------------- Attention Out ----------------
        region.origin = f * frame_size_Attention_Out;
        region.size   = frame_size_Attention_Out;
        GlobMem_BUF_frame_Attention_Out[f] = clCreateSubBuffer(
            GlobMem_BUF_Attention_Out, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);
        */
        // ---------------- error check ----------------
    }

    // ref: clCreateBuffer method
    // https://www.cnblogs.com/willhua/p/9463515.html
    // https://zhuanlan.zhihu.com/p/684133601
    // don't care loading data at later
    GlobMem_BUF_LayerNorm_Gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm_gamma, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Gamma" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_LayerNorm_Beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm_beta, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Beta" << endl
             << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_LayerNorm1_Gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm1_gamma, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm1_Gamma" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_LayerNorm1_Beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), LayerNorm1_beta, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm1_Beta" << endl
             << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wq_Linear_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bq, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wq, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wq" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wk_Linear_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bk, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wk, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wk" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wv_Linear_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bv, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wv, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wv" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_Wout = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wout, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_Wout" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_bout = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bout, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_bout" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_scale_wout = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wout, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_scale_wout" << endl
             << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_MLP_Output = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, MLP_OUTPUT_CHUNKS * MLP_CHUNK_U32 * sizeof(uint32_t), MLP_Output_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_Output" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_W1 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MLP_W1_CHUNKS * 32 * sizeof(int8_t), MLP_W1_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_W1" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_W2 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MLP_W2_CHUNKS * 32 * sizeof(int8_t), MLP_W2_Packed, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_W2" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_b1 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MLP_DIM * sizeof(float), MLP_b1, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_b1" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_b2 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), MLP_b2, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_b2" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_scale_w1 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, MLP_DIM * sizeof(float), MLP_scale_w1, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_scale_w1" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_MLP_scale_w2 = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), MLP_scale_w2, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_MLP_scale_w2" << endl
             << endl;
        return EXIT_FAILURE;
    }

#if HOST_DEBUG_LN_SELF_TEST
    // Host-only debug: verify the layernorm CU inside the integrated xclbin
    // can complete before it is connected to the full encoder pipeline.
    {
        debug_event_log("BEGIN integrated xclbin standalone LN self-test");
        float (*ln_test_x)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(float)));
        float *ln_test_gamma = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
        float *ln_test_beta = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
        int8_t (*ln_test_out)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(SEQ_LEN * EMBED_DIM * sizeof(int8_t)));
        float *ln_test_scaler = reinterpret_cast<float *>(host_aligned_alloc(SEQ_LEN * sizeof(float)));

        if (!ln_test_x || !ln_test_gamma || !ln_test_beta || !ln_test_out || !ln_test_scaler)
        {
            cout << endl
                 << "HOST-Error: Out of Memory during integrated LN self-test allocation" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        for (int i = 0; i < SEQ_LEN; i++)
        {
            for (int d = 0; d < EMBED_DIM; d++)
            {
                ln_test_x[i][d] = 0.01f * static_cast<float>((d % 17) - 8) + 0.001f * static_cast<float>(i);
                ln_test_out[i][d] = 0;
            }
            ln_test_scaler[i] = 0.0f;
        }
        for (int d = 0; d < EMBED_DIM; d++)
        {
            ln_test_gamma[d] = 1.0f;
            ln_test_beta[d] = 0.0f;
        }

        cl_mem ln_test_buf_x = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                              SEQ_LEN * EMBED_DIM * sizeof(float), ln_test_x, &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to allocate ln_test_buf_x" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        cl_mem ln_test_buf_gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                  EMBED_DIM * sizeof(float), ln_test_gamma, &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to allocate ln_test_buf_gamma" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        cl_mem ln_test_buf_beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                                 EMBED_DIM * sizeof(float), ln_test_beta, &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to allocate ln_test_buf_beta" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        cl_mem ln_test_buf_out = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                SEQ_LEN * EMBED_DIM * sizeof(int8_t), ln_test_out, &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to allocate ln_test_buf_out" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        cl_mem ln_test_buf_scaler = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                                   SEQ_LEN * sizeof(float), ln_test_scaler, &errCode);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to allocate ln_test_buf_scaler" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        cl_mem ln_test_output_buffers[] = {ln_test_buf_out, ln_test_buf_scaler};
        cl_event ln_test_x_event = NULL;
        cl_event ln_test_gamma_event = NULL;
        cl_event ln_test_beta_event = NULL;
        cl_event ln_test_out_event = NULL;
        cl_event ln_test_scaler_event = NULL;
        cl_event ln_test_kernel_event = NULL;
        cl_event ln_test_read_event = NULL;

        errCode = false;
        errCode |= clSetKernelArg(K_layernorm[1], 0, sizeof(cl_mem), &ln_test_buf_x);
        errCode |= clSetKernelArg(K_layernorm[1], 1, sizeof(cl_mem), &ln_test_buf_gamma);
        errCode |= clSetKernelArg(K_layernorm[1], 2, sizeof(cl_mem), &ln_test_buf_beta);
        errCode |= clSetKernelArg(K_layernorm[1], 3, sizeof(cl_mem), &ln_test_buf_out);
        errCode |= clSetKernelArg(K_layernorm[1], 4, sizeof(cl_mem), &ln_test_buf_scaler);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to set LN self-test kernel arguments" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        debug_event_log("BEFORE enqueue LN self-test x migrate");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &ln_test_buf_x, 0, 0, NULL, &ln_test_x_event);
        debug_event_status("AFTER enqueue LN self-test x migrate", ln_test_x_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate LN self-test x buffer" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test x migrate", ln_test_x_event);
        clWaitForEvents(1, &ln_test_x_event);
        debug_event_status("AFTER wait LN self-test x migrate", ln_test_x_event);

        debug_event_log("BEFORE enqueue LN self-test gamma migrate");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &ln_test_buf_gamma, 0, 0, NULL, &ln_test_gamma_event);
        debug_event_status("AFTER enqueue LN self-test gamma migrate", ln_test_gamma_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate LN self-test gamma buffer" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test gamma migrate", ln_test_gamma_event);
        clWaitForEvents(1, &ln_test_gamma_event);
        debug_event_status("AFTER wait LN self-test gamma migrate", ln_test_gamma_event);

        debug_event_log("BEFORE enqueue LN self-test beta migrate");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &ln_test_buf_beta, 0, 0, NULL, &ln_test_beta_event);
        debug_event_status("AFTER enqueue LN self-test beta migrate", ln_test_beta_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate LN self-test beta buffer" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test beta migrate", ln_test_beta_event);
        clWaitForEvents(1, &ln_test_beta_event);
        debug_event_status("AFTER wait LN self-test beta migrate", ln_test_beta_event);

        debug_event_log("BEFORE enqueue LN self-test out migrate");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &ln_test_buf_out,
                                             CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &ln_test_out_event);
        debug_event_status("AFTER enqueue LN self-test out migrate", ln_test_out_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate LN self-test out buffer" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test out migrate", ln_test_out_event);
        clWaitForEvents(1, &ln_test_out_event);
        debug_event_status("AFTER wait LN self-test out migrate", ln_test_out_event);

        debug_event_log("BEFORE enqueue LN self-test scaler migrate");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &ln_test_buf_scaler,
                                             CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &ln_test_scaler_event);
        debug_event_status("AFTER enqueue LN self-test scaler migrate", ln_test_scaler_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate LN self-test scaler buffer" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test scaler migrate", ln_test_scaler_event);
        clWaitForEvents(1, &ln_test_scaler_event);
        debug_event_status("AFTER wait LN self-test scaler migrate", ln_test_scaler_event);

        debug_event_log("BEFORE enqueue LN self-test kernel with completed migrations");
        errCode = clEnqueueTask(Command_Queue, K_layernorm[1], 0, NULL, &ln_test_kernel_event);
        debug_event_status("AFTER enqueue LN self-test kernel", ln_test_kernel_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to submit LN self-test kernel" << endl
                 << endl;
            return EXIT_FAILURE;
        }

        debug_event_status("BEFORE wait LN self-test kernel", ln_test_kernel_event);
        clWaitForEvents(1, &ln_test_kernel_event);
        debug_event_status("AFTER wait LN self-test kernel", ln_test_kernel_event);

        debug_event_log("BEFORE enqueue LN self-test readback");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 2, ln_test_output_buffers,
                                             CL_MIGRATE_MEM_OBJECT_HOST, 1, &ln_test_kernel_event, &ln_test_read_event);
        debug_event_status("AFTER enqueue LN self-test readback", ln_test_read_event);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to read back LN self-test outputs" << endl
                 << endl;
            return EXIT_FAILURE;
        }
        debug_event_status("BEFORE wait LN self-test readback", ln_test_read_event);
        clWaitForEvents(1, &ln_test_read_event);
        debug_event_status("AFTER wait LN self-test readback", ln_test_read_event);
        debug_float_tensor_stats("LN self-test scaler", ln_test_scaler, SEQ_LEN);
        cout << "[HOST-DATA] LN self-test out first_values:";
        for (int i = 0; i < 8 && i < EMBED_DIM; i++)
        {
            cout << " " << static_cast<int>(ln_test_out[0][i]);
        }
        cout << endl;
        cout.flush();

        clReleaseEvent(ln_test_x_event);
        clReleaseEvent(ln_test_gamma_event);
        clReleaseEvent(ln_test_beta_event);
        clReleaseEvent(ln_test_out_event);
        clReleaseEvent(ln_test_scaler_event);
        clReleaseEvent(ln_test_kernel_event);
        clReleaseEvent(ln_test_read_event);
        clReleaseMemObject(ln_test_buf_x);
        clReleaseMemObject(ln_test_buf_gamma);
        clReleaseMemObject(ln_test_buf_beta);
        clReleaseMemObject(ln_test_buf_out);
        clReleaseMemObject(ln_test_buf_scaler);
        free(ln_test_x);
        free(ln_test_gamma);
        free(ln_test_beta);
        free(ln_test_out);
        free(ln_test_scaler);
        debug_event_log("END integrated xclbin standalone LN self-test");
    }
#endif

    // ============================================================================
    // Step 5: Run configured Spatial Transformer encoder blocks in sequence
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run " << NUM_ENCODER_BLOCKS << " Spatial Encoder Blocks" << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    double Host_Post_Time_ms = 0.0;
    int Total_Kernel_Events = 0;
    int Total_Memory_Events = 0;
    vector<cl_event> Profiling_Kernel_Events;
    vector<cl_event> Profiling_Memory_Events;
    vector<string> Profiling_Kernel_Names;

    const int Nb_Of_Mem_Events = 13;
    const int Nb_Of_Exe_Events = 5;
    const int Nb_Of_Att_Events = NUM_HEADS;
    cl_event Mem_op_event[Nb_Of_Mem_Events];
    cl_event K_exe_event[Nb_Of_Exe_Events];
    cl_event A_exe_event[Nb_Of_Att_Events];
    // ============================================================================
    // EVENT EXECUTION PIPELINE (PER FRAME, PER BLOCK)
    // ============================================================================
    //   [MEMORY STAGE]
    //   1. Migrate fused LayerNorm + QKV input/output buffers to device
    //        -> Mem_op_event[0 ~ 2]
    //   [STAGE 1: FUSED LAYERNORM 0 + QKV PROJECTION]
    //   2. K_linear_qkv_multihead
    //        waits: Mem_op_event[0], Mem_op_event[1]
    //        produces: K_exe_event[0]
    //   [STAGE 3: MULTI-HEAD ATTENTION]
    //   4. K_attention_opt[h] for each head
    //        waits: K_exe_event[0], Mem_op_event[2]
    //        produces: A_exe_event[h] for h in [0..NUM_HEADS-1]
    //   [STAGE 4: ATTENTION OUTPUT RETURN]
    //   5. Migrate Attention_Out back to host
    //        MUST wait: ALL A_exe_event[]/
    //   [HOST POST-PROCESS]
    //   6. concat_attention_heads()
    //   7. output projection (Wout)
    //   8. residual add
    //   [STAGE 5: FUSED LAYERNORM 1 + MLP]
    //   9. K_mlp_q
    //        waits: previous K_exe_event[2]
    //   [STAGE 6: FINAL RESIDUAL]
    // ============================================================================

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++)
    {
        memset(Mem_op_event, 0, sizeof(Mem_op_event));
        memset(K_exe_event, 0, sizeof(K_exe_event));
        memset(A_exe_event, 0, sizeof(A_exe_event));
        const string spatial_encoder_dir = join_path(spatial_params_root, "encoderblock_" + to_string(block));
        cout << "HOST-Info: Loading parameters for encoder block " << block << " from " << spatial_encoder_dir << endl;

        // loading weight per block
        if (!load_txt_values_as_float(join_path(spatial_encoder_dir, "LayerNorm_0_scale.txt"), LayerNorm_gamma, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "LayerNorm_0_bias.txt"), LayerNorm_beta, EMBED_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_query_kernel_qvalue.txt"), &Wq[0][0], EMBED_DIM * EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_query_bias.txt"), bq, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_query_kernel_scale.txt"), scale_wq, EMBED_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_key_kernel_qvalue.txt"), &Wk[0][0], EMBED_DIM * EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_key_bias.txt"), bk, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_key_kernel_scale.txt"), scale_wk, EMBED_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_value_kernel_qvalue.txt"), &Wv[0][0], EMBED_DIM * EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_value_bias.txt"), bv, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_value_kernel_scale.txt"), scale_wv, EMBED_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_out_kernel_qvalue.txt"), &Wout[0][0], EMBED_DIM * EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_out_bias.txt"), bout, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MultiHeadDotProductAttention_0_out_kernel_scale.txt"), scale_wout, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "LayerNorm_1_scale.txt"), LayerNorm1_gamma, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "LayerNorm_1_bias.txt"), LayerNorm1_beta, EMBED_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_0_kernel_qvalue.txt"), &MLP_W1[0][0], EMBED_DIM * MLP_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_0_bias.txt"), MLP_b1, MLP_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_0_kernel_scale.txt"), MLP_scale_w1, MLP_DIM) ||
            !load_txt_values_as_int8(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_1_kernel_qvalue.txt"), &MLP_W2[0][0], MLP_DIM * EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_1_bias.txt"), MLP_b2, EMBED_DIM) ||
            !load_txt_values_as_float(join_path(spatial_encoder_dir, "MlpBlock_0_Dense_1_kernel_scale.txt"), MLP_scale_w2, EMBED_DIM))
        {
            cout << endl
                 << "HOST-Error: Failed to load one or more parameter files for encoder block " << block << endl
                 << endl;
            return EXIT_FAILURE;
        }

        pack_linear_layernorm_weight(Wq, Wq_Linear_Packed);
        pack_linear_layernorm_weight(Wk, Wk_Linear_Packed);
        pack_linear_layernorm_weight(Wv, Wv_Linear_Packed);
        pack_mlp_weights(MLP_W1, MLP_W2, MLP_W1_Packed, MLP_W2_Packed);

        // create subbuffer for KQV first
        // already consider frame term (create all buffer)
        cl_mem GlobMem_BUF_head_Q[FRAME * NUM_HEADS] = {NULL};
        cl_mem GlobMem_BUF_head_K[FRAME * NUM_HEADS] = {NULL};
        cl_mem GlobMem_BUF_head_V[FRAME * NUM_HEADS] = {NULL};
        cl_mem GlobMem_BUF_head_Out[FRAME * NUM_HEADS] = {NULL};
        const size_t HEAD_BYTES = SEQ_LEN * HEAD_DIM * sizeof(float);
        const size_t HEAD_BYTES_OUT = SEQ_LEN * HEAD_DIM * sizeof(float);

        for (int frame = 0; frame < FRAME; frame++)
        {
            cout << "----- running block " << block << ", frame " << frame << " -----\n";
            for (int h = 0; h < NUM_HEADS; h++)
            {
                cl_buffer_region head_region, head_region_out;
                int idx = frame * NUM_HEADS + h;
                head_region.origin = static_cast<size_t>(idx) * HEAD_BYTES;
                head_region.size = HEAD_BYTES;
                head_region_out.origin = static_cast<size_t>(idx) * HEAD_BYTES_OUT;
                head_region_out.size = HEAD_BYTES_OUT;

                // TODO: FLAG is right?
                // TODO: QKV can be set as INT? (right now is FLOAT)
                GlobMem_BUF_head_Q[idx] = clCreateSubBuffer(GlobMem_BUF_Q, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                GlobMem_BUF_head_K[idx] = clCreateSubBuffer(GlobMem_BUF_K, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                GlobMem_BUF_head_V[idx] = clCreateSubBuffer(GlobMem_BUF_V, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
                // Attention_Out is stored as fp32 per head.
                GlobMem_BUF_head_Out[idx] = clCreateSubBuffer(GlobMem_BUF_Attention_Out, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &head_region_out, &errCode);
                if (errCode != CL_SUCCESS)
                    return EXIT_FAILURE;
            }
        }

        for (int frame = 0; frame < FRAME; frame++)
        {
            memset(Mem_op_event, 0, sizeof(Mem_op_event));
            memset(K_exe_event, 0, sizeof(K_exe_event));
            memset(A_exe_event, 0, sizeof(A_exe_event));

            // ========================== create subbuffer for KQV finish ===================================

            // setting kenel Args

            cl_mem LayerNorm1_Input_Buffers[] = {
                GlobMem_BUF_frame_After_Attention_Residual_Float[frame],
                GlobMem_BUF_LayerNorm1_Gamma,
                GlobMem_BUF_LayerNorm1_Beta};
            cl_mem LinearLayerNorm_Input_Buffers[] = {
                GlobMem_BUF_frame_X[frame],
                GlobMem_BUF_LayerNorm_Gamma, GlobMem_BUF_LayerNorm_Beta,
                GlobMem_BUF_Wk, GlobMem_BUF_Wv, GlobMem_BUF_Wq,
                GlobMem_BUF_bq, GlobMem_BUF_bk, GlobMem_BUF_bv,
                GlobMem_BUF_scale_wq, GlobMem_BUF_scale_wk, GlobMem_BUF_scale_wv};
            cl_mem QKV_Output_Buffers[] = {
                GlobMem_BUF_frame_Q[frame], GlobMem_BUF_frame_K[frame], GlobMem_BUF_frame_V[frame]};
            cl_mem Projection_Input_Buffers[] = {
                GlobMem_BUF_Wout, GlobMem_BUF_bout, GlobMem_BUF_scale_wout};

            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------

            // setting fused LN0 + linear_qkv_multihead Args
            errCode = false;
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 0, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 3, sizeof(cl_mem), &GlobMem_BUF_Wk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 4, sizeof(cl_mem), &GlobMem_BUF_Wv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 5, sizeof(cl_mem), &GlobMem_BUF_Wq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 6, sizeof(cl_mem), &GlobMem_BUF_bq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 7, sizeof(cl_mem), &GlobMem_BUF_bk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 8, sizeof(cl_mem), &GlobMem_BUF_bv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 9, sizeof(cl_mem), &GlobMem_BUF_scale_wq);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 10, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 11, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 12, sizeof(cl_mem), &GlobMem_BUF_frame_Q[frame]);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 13, sizeof(cl_mem), &GlobMem_BUF_frame_K[frame]);
            errCode |= clSetKernelArg(K_linear_qkv_multihead, 14, sizeof(cl_mem), &GlobMem_BUF_frame_V[frame]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set LN0/QKV kernel arguments for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
            for (int h = 0; h < NUM_HEADS; h++)
            {
                int idx = frame * NUM_HEADS + h;
                // setting(update) K_attention_opt Args
                errCode = false;
                errCode |= clSetKernelArg(K_attention_opt[h], 0, sizeof(cl_mem), &GlobMem_BUF_head_Q[idx]);
                errCode |= clSetKernelArg(K_attention_opt[h], 1, sizeof(cl_mem), &GlobMem_BUF_head_K[idx]);
                errCode |= clSetKernelArg(K_attention_opt[h], 2, sizeof(cl_mem), &GlobMem_BUF_head_V[idx]);
                errCode |= clSetKernelArg(K_attention_opt[h], 3, sizeof(cl_mem), &GlobMem_BUF_head_Out[idx]);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                         << "Host-ERROR: Failed to set attention kernel arguments for frame: " << frame << ", block: " << block << ", head: " << h << endl
                         << endl;
                    return EXIT_FAILURE;
                }
            }

            errCode = false;
            errCode |= clSetKernelArg(K_projection_out, 0, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Out[frame]);
            errCode |= clSetKernelArg(K_projection_out, 1, sizeof(cl_mem), &GlobMem_BUF_Wout);
            errCode |= clSetKernelArg(K_projection_out, 2, sizeof(cl_mem), &GlobMem_BUF_bout);
            errCode |= clSetKernelArg(K_projection_out, 3, sizeof(cl_mem), &GlobMem_BUF_scale_wout);
            errCode |= clSetKernelArg(K_projection_out, 4, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Projected[frame]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set projection_out kernel arguments for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            errCode = false;
            errCode |= clSetKernelArg(K_residual, 0, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]);
            errCode |= clSetKernelArg(K_residual, 1, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Projected[frame]);
            errCode |= clSetKernelArg(K_residual, 2, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float[frame]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set attention residual kernel arguments for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            // ------------------------------------------------------
            // Step 5.2: Copy Input Data from Host to Global Memory
            // ------------------------------------------------------
            // move fused LayerNorm + QKV args to Device
            // move Q/K/V output buffers to Device
            // move Attention Output buffer to Device (TODO: is it useless?)
            // ------------------------------------------------------
            debug_event_log("BEGIN block " + to_string(block) + ", frame " + to_string(frame) + " memory migration events");
            if (block == 0)
            {
                debug_event_log("BEFORE enqueue Mem_op_event[0]: migrate 12 fused LN0/QKV inputs");
                errCode = clEnqueueMigrateMemObjects(Command_Queue, 12, LinearLayerNorm_Input_Buffers, 0, 0, NULL, &Mem_op_event[0]);
            }
            else
            {
                debug_event_log("BEFORE enqueue Mem_op_event[0]: migrate 11 fused LN0/QKV parameter inputs");
                errCode = clEnqueueMigrateMemObjects(Command_Queue, 11, LinearLayerNorm_Input_Buffers + 1, 0, 0, NULL, &Mem_op_event[0]);
            }
            debug_event_status("AFTER enqueue Mem_op_event[0]", Mem_op_event[0]);
            debug_event_log("BEFORE enqueue Mem_op_event[1]: migrate Q/K/V output buffers");
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 3, QKV_Output_Buffers, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[1]);
            debug_event_status("AFTER enqueue Mem_op_event[1]", Mem_op_event[1]);
            debug_event_log("BEFORE enqueue Mem_op_event[2]: migrate frame attention output buffer");
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_frame_Attention_Out[frame], CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[2]);
            debug_event_status("AFTER enqueue Mem_op_event[2]", Mem_op_event[2]);
            debug_event_log("BEFORE enqueue Mem_op_event[3]: migrate projection input buffers");
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 3, Projection_Input_Buffers, 0, 0, NULL, &Mem_op_event[3]);
            debug_event_status("AFTER enqueue Mem_op_event[3]", Mem_op_event[3]);
            debug_event_log("BEFORE enqueue Mem_op_event[4]: migrate projection output buffer");
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_frame_Attention_Projected[frame], CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[4]);
            debug_event_status("AFTER enqueue Mem_op_event[4]", Mem_op_event[4]);
            // origin error: errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_head_Out[frame * SEQ_LEN], CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[3]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate LN0/QKV buffers for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            // ----------------------------------------
            // Step 5.3: Submit Kernels for Execution
            // ----------------------------------------
            cl_event qkv_wait_list[] = {Mem_op_event[0], Mem_op_event[1]};
            debug_event_wait_list("BEFORE enqueue K_exe_event[0] linear_qkv_multihead", 2, qkv_wait_list);
            errCode = clEnqueueTask(Command_Queue, K_linear_qkv_multihead, 2, qkv_wait_list, &K_exe_event[0]);
            debug_event_status("AFTER enqueue K_exe_event[0] linear_qkv_multihead", K_exe_event[0]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit linear_qkv_multihead for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
#if HOST_DEBUG_SERIAL_STAGES
            debug_event_status("BEFORE serial wait K_exe_event[0] linear_qkv_multihead", K_exe_event[0]);
            clWaitForEvents(1, &K_exe_event[0]);
            debug_event_status("AFTER serial wait K_exe_event[0] linear_qkv_multihead", K_exe_event[0]);
#endif

            for (int h = 0; h < NUM_HEADS; h++)
            {

                // setting K_attention_opt Args
                cl_event attention_wait_list[] = {K_exe_event[0], Mem_op_event[2]};
                // Compute Attention
                debug_event_wait_list("BEFORE enqueue A_exe_event[" + to_string(h) + "] attention head " + to_string(h), 2, attention_wait_list);
                errCode = clEnqueueTask(Command_Queue, K_attention_opt[h], 2, attention_wait_list, &A_exe_event[h]);
                debug_event_status("AFTER enqueue A_exe_event[" + to_string(h) + "] attention head " + to_string(h), A_exe_event[h]);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                         << "HOST-Error: Failed to submit attention for frame: " << frame << ", block: " << block << ", head: " << h << endl
                         << endl;
                    return EXIT_FAILURE;
                }
#if HOST_DEBUG_SERIAL_STAGES
                debug_event_status("BEFORE serial wait A_exe_event[" + to_string(h) + "] attention head " + to_string(h), A_exe_event[h]);
                clWaitForEvents(1, &A_exe_event[h]);
                debug_event_status("AFTER serial wait A_exe_event[" + to_string(h) + "] attention head " + to_string(h), A_exe_event[h]);
#endif
            }
            auto Host_Post_Start = chrono::high_resolution_clock::now();

            // concat_attention_heads((Attention_Out + frame * NUM_HEADS), (Attention_Concat + frame * SEQ_LEN));
            // linear_quant_host_fp32_input((Attention_Concat + frame * SEQ_LEN), Wout, bout, scale_wout, (Attention_Projected + frame * SEQ_LEN));
            // residual_add((X + frame * SEQ_LEN), (Attention_Projected + frame * SEQ_LEN), (After_Attention_Residual + frame * SEQ_LEN));
            // copy_int_to_float((After_Attention_Residual + frame * SEQ_LEN), (X + frame * SEQ_LEN));

            cl_event projection_wait_list[NUM_HEADS + 2];
            for (int h = 0; h < NUM_HEADS; h++)
                projection_wait_list[h] = A_exe_event[h];
            projection_wait_list[NUM_HEADS] = Mem_op_event[3];
            projection_wait_list[NUM_HEADS + 1] = Mem_op_event[4];

            debug_event_wait_list("BEFORE enqueue K_exe_event[1] projection_out", NUM_HEADS + 2, projection_wait_list);
            errCode = clEnqueueTask(Command_Queue, K_projection_out, NUM_HEADS + 2, projection_wait_list, &K_exe_event[1]);
            debug_event_status("AFTER enqueue K_exe_event[1] projection_out", K_exe_event[1]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit projection_out for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
#if HOST_DEBUG_SERIAL_STAGES
            debug_event_status("BEFORE serial wait K_exe_event[1] projection_out", K_exe_event[1]);
            clWaitForEvents(1, &K_exe_event[1]);
            debug_event_status("AFTER serial wait K_exe_event[1] projection_out", K_exe_event[1]);
#endif
#if HOST_DEBUG_READBACK
            cl_mem attention_residual_input_debug_buffers[] = {
                GlobMem_BUF_frame_X[frame],
                GlobMem_BUF_frame_Attention_Projected[frame]};
            cl_event attention_residual_inputs_read_event = NULL;
            debug_event_log("BEFORE enqueue attention residual input debug readback: X and projected attention to host");
            errCode = clEnqueueMigrateMemObjects(
                Command_Queue,
                2,
                attention_residual_input_debug_buffers,
                CL_MIGRATE_MEM_OBJECT_HOST,
                1,
                &K_exe_event[1],
                &attention_residual_inputs_read_event);
            debug_event_status("AFTER enqueue attention residual input debug readback", attention_residual_inputs_read_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate attention residual inputs back to host for debug, block " << block << ", frame " << frame << endl
                     << endl;
                return EXIT_FAILURE;
            }
            debug_event_status("BEFORE wait attention residual input debug readback", attention_residual_inputs_read_event);
            clWaitForEvents(1, &attention_residual_inputs_read_event);
            debug_event_status("AFTER wait attention residual input debug readback", attention_residual_inputs_read_event);
            debug_float_tensor_stats(
                "attention residual input X block " + to_string(block) + ", frame " + to_string(frame),
                reinterpret_cast<float *>(X + frame * SEQ_LEN),
                static_cast<size_t>(SEQ_LEN) * EMBED_DIM);
            debug_float_tensor_stats(
                "attention residual input Attention_Projected block " + to_string(block) + ", frame " + to_string(frame),
                reinterpret_cast<float *>(Attention_Projected + frame * SEQ_LEN),
                static_cast<size_t>(SEQ_LEN) * EMBED_DIM);
            debug_event_log("BEFORE release attention residual input debug readback event");
            clReleaseEvent(attention_residual_inputs_read_event);
            debug_event_log("AFTER release attention residual input debug readback event");

            cl_event attention_residual_inputs_restore_event = NULL;
            debug_event_log("BEFORE enqueue attention residual input restore: X and projected attention back to device");
            errCode = clEnqueueMigrateMemObjects(
                Command_Queue,
                2,
                attention_residual_input_debug_buffers,
                0,
                0,
                NULL,
                &attention_residual_inputs_restore_event);
            debug_event_status("AFTER enqueue attention residual input restore", attention_residual_inputs_restore_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to restore attention residual inputs to device after debug readback, block " << block << ", frame " << frame << endl
                     << endl;
                return EXIT_FAILURE;
            }
            debug_event_status("BEFORE wait attention residual input restore", attention_residual_inputs_restore_event);
            clWaitForEvents(1, &attention_residual_inputs_restore_event);
            debug_event_status("AFTER wait attention residual input restore", attention_residual_inputs_restore_event);
            debug_event_log("BEFORE release attention residual input restore event");
            clReleaseEvent(attention_residual_inputs_restore_event);
            debug_event_log("AFTER release attention residual input restore event");
#endif

            debug_event_log("BEFORE enqueue Mem_op_event[6]: migrate attention residual output buffer");
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_frame_After_Attention_Residual_Float[frame],
                                                 CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[6]);
            debug_event_status("AFTER enqueue Mem_op_event[6]", Mem_op_event[6]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate residual buffers for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            cl_event residual_wait_list[] = {K_exe_event[1], Mem_op_event[6]};
            debug_event_wait_list("BEFORE enqueue K_exe_event[2] attention residual", 2, residual_wait_list);
            errCode = clEnqueueTask(Command_Queue, K_residual, 2, residual_wait_list, &K_exe_event[2]);
            debug_event_status("AFTER enqueue K_exe_event[2] attention residual", K_exe_event[2]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit attention residual for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
#if HOST_DEBUG_SERIAL_STAGES
            debug_event_status("BEFORE serial wait K_exe_event[2] attention residual", K_exe_event[2]);
            clWaitForEvents(1, &K_exe_event[2]);
            debug_event_status("AFTER serial wait K_exe_event[2] attention residual", K_exe_event[2]);
#endif

#if HOST_DEBUG_READBACK
            cl_event ln1_input_read_event = NULL;
            debug_event_log("BEFORE enqueue LN1 input debug readback: attention residual output to host");
            errCode = clEnqueueMigrateMemObjects(
                Command_Queue,
                1,
                &GlobMem_BUF_frame_After_Attention_Residual_Float[frame],
                CL_MIGRATE_MEM_OBJECT_HOST,
                1,
                &K_exe_event[2],
                &ln1_input_read_event);
            debug_event_status("AFTER enqueue LN1 input debug readback", ln1_input_read_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate LN1 input back to host for debug, block " << block << ", frame " << frame << endl
                     << endl;
                return EXIT_FAILURE;
            }
            debug_event_status("BEFORE wait LN1 input debug readback", ln1_input_read_event);
            clWaitForEvents(1, &ln1_input_read_event);
            debug_event_status("AFTER wait LN1 input debug readback", ln1_input_read_event);
            debug_float_tensor_stats(
                "LN1 input After_Attention_Residual_Float block " + to_string(block) + ", frame " + to_string(frame),
                reinterpret_cast<float *>(After_Attention_Residual_Float + frame * SEQ_LEN),
                static_cast<size_t>(SEQ_LEN) * EMBED_DIM);
            debug_event_log("BEFORE release LN1 input debug readback event");
            clReleaseEvent(ln1_input_read_event);
            debug_event_log("AFTER release LN1 input debug readback event");

            cl_event ln1_input_restore_event = NULL;
            debug_event_log("BEFORE enqueue LN1 input restore: attention residual output back to device");
            errCode = clEnqueueMigrateMemObjects(
                Command_Queue,
                1,
                &GlobMem_BUF_frame_After_Attention_Residual_Float[frame],
                0,
                0,
                NULL,
                &ln1_input_restore_event);
            debug_event_status("AFTER enqueue LN1 input restore", ln1_input_restore_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to restore LN1 input to device after debug readback, block " << block << ", frame " << frame << endl
                     << endl;
                return EXIT_FAILURE;
            }
            debug_event_status("BEFORE wait LN1 input restore", ln1_input_restore_event);
            clWaitForEvents(1, &ln1_input_restore_event);
            debug_event_status("AFTER wait LN1 input restore", ln1_input_restore_event);
            debug_event_log("BEFORE release LN1 input restore event");
            clReleaseEvent(ln1_input_restore_event);
            debug_event_log("AFTER release LN1 input restore event");
#endif

            auto Host_Post_Mid = chrono::high_resolution_clock::now();
            Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_Post_Mid - Host_Post_Start).count();
            // ---------------------------------------------------------
            // Step 5.4-1: Compute fused LayerNorm1 + MLP at Device
            // ---------------------------------------------------------
            debug_float_tensor_stats(
                "LN1 source gamma block " + to_string(block) + ", frame " + to_string(frame),
                LayerNorm1_gamma,
                EMBED_DIM);
            debug_float_tensor_stats(
                "LN1 source beta block " + to_string(block) + ", frame " + to_string(frame),
                LayerNorm1_beta,
                EMBED_DIM);

            errCode = false;
            debug_event_log("BEFORE enqueue Mem_op_event[10]: migrate fused LN1+MLP gamma/beta buffers");
            errCode |= clEnqueueMigrateMemObjects(Command_Queue, 2, LayerNorm1_Input_Buffers + 1, 0, 0, NULL, &Mem_op_event[10]);
            debug_event_status("AFTER enqueue Mem_op_event[10]", Mem_op_event[10]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate fused LN1+MLP gamma/beta buffers for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            debug_event_status("BEFORE serial wait Mem_op_event[10] fused LN1+MLP gamma/beta buffers", Mem_op_event[10]);
            clWaitForEvents(1, &Mem_op_event[10]);
            debug_event_status("AFTER serial wait Mem_op_event[10] fused LN1+MLP gamma/beta buffers", Mem_op_event[10]);

            errCode = false;
            errCode |= clSetKernelArg(K_mlp_q, 0, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float[frame]);
            errCode |= clSetKernelArg(K_mlp_q, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Gamma);
            errCode |= clSetKernelArg(K_mlp_q, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Beta);
            errCode |= clSetKernelArg(K_mlp_q, 3, sizeof(cl_mem), &GlobMem_BUF_MLP_W1);
            errCode |= clSetKernelArg(K_mlp_q, 4, sizeof(cl_mem), &GlobMem_BUF_MLP_W2);
            errCode |= clSetKernelArg(K_mlp_q, 5, sizeof(cl_mem), &GlobMem_BUF_MLP_b1);
            errCode |= clSetKernelArg(K_mlp_q, 6, sizeof(cl_mem), &GlobMem_BUF_MLP_b2);
            errCode |= clSetKernelArg(K_mlp_q, 7, sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w1);
            errCode |= clSetKernelArg(K_mlp_q, 8, sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w2);
            errCode |= clSetKernelArg(K_mlp_q, 9, sizeof(cl_mem), &GlobMem_BUF_MLP_Output);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set fused LN1+MLP kernel arguments for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            cl_mem MLP_Param_Buffers[] = {
                GlobMem_BUF_MLP_W1, GlobMem_BUF_MLP_W2,
                GlobMem_BUF_MLP_b1, GlobMem_BUF_MLP_b2,
                GlobMem_BUF_MLP_scale_w1, GlobMem_BUF_MLP_scale_w2};
            cl_event mlp_param_event = nullptr;
            debug_event_log("BEFORE enqueue mlp_param_event: migrate MLP parameter buffers");
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 6, MLP_Param_Buffers, 0, 0, NULL, &mlp_param_event);
            debug_event_status("AFTER enqueue mlp_param_event", mlp_param_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate MLP parameter buffers for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
            debug_event_status("BEFORE wait mlp_param_event", mlp_param_event);
            clWaitForEvents(1, &mlp_param_event);
            debug_event_status("AFTER wait mlp_param_event", mlp_param_event);
            debug_event_log("BEFORE release mlp_param_event");
            clReleaseEvent(mlp_param_event);
            debug_event_log("AFTER release mlp_param_event");

            cl_event mlp_output_alloc_event = nullptr;
            cl_mem mlp_output_buffer[] = {GlobMem_BUF_MLP_Output};
            debug_event_log("BEFORE enqueue mlp_output_alloc_event: migrate MLP output buffer");
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, mlp_output_buffer, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &mlp_output_alloc_event);
            debug_event_status("AFTER enqueue mlp_output_alloc_event", mlp_output_alloc_event);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to migrate MLP output buffer for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            vector<cl_event> mlp_tile_events;
            for (int token_start = 0; token_start < SEQ_LEN; token_start += MLP_KERNEL_SEQ_LEN)
            {
                cl_event mlp_kernel_event = nullptr;
                int valid_seq_len = SEQ_LEN;

                errCode = false;
                errCode |= clSetKernelArg(K_mlp_q, 10, sizeof(int), &token_start);
                errCode |= clSetKernelArg(K_mlp_q, 11, sizeof(int), &valid_seq_len);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                         << "Host-ERROR: Failed to set MLP tile arguments for block " << block << endl
                         << endl;
                    return EXIT_FAILURE;
                }

                cl_event mlp_wait_list[] = {K_exe_event[2], mlp_output_alloc_event};
                debug_event_wait_list("BEFORE enqueue mlp_kernel_event token_start=" + to_string(token_start), 2, mlp_wait_list);
                errCode = clEnqueueTask(Command_Queue, K_mlp_q, 2, mlp_wait_list, &mlp_kernel_event);
                debug_event_status("AFTER enqueue mlp_kernel_event token_start=" + to_string(token_start), mlp_kernel_event);
                if (errCode != CL_SUCCESS)
                {
                    cout << endl
                         << "HOST-Error: Failed to submit MLP kernel for block " << block << endl
                         << endl;
                    return EXIT_FAILURE;
                }

                mlp_tile_events.push_back(mlp_kernel_event);
            }

            errCode = false;
            errCode |= clSetKernelArg(K_residual, 0, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float[frame]);
            errCode |= clSetKernelArg(K_residual, 1, sizeof(cl_mem), &GlobMem_BUF_MLP_Output);
            errCode |= clSetKernelArg(K_residual, 2, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set MLP residual kernel arguments for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            vector<cl_event> mlp_residual_wait_list = mlp_tile_events;
            debug_event_wait_list("BEFORE enqueue K_exe_event[4] MLP residual", static_cast<cl_uint>(mlp_residual_wait_list.size()), mlp_residual_wait_list.data());
            errCode = clEnqueueTask(
                Command_Queue,
                K_residual,
                static_cast<cl_uint>(mlp_residual_wait_list.size()),
                mlp_residual_wait_list.data(),
                &K_exe_event[4]);
            debug_event_status("AFTER enqueue K_exe_event[4] MLP residual", K_exe_event[4]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit MLP residual for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            debug_event_log("UPSTREAM SNAPSHOT before waiting for MLP residual, block " + to_string(block) + ", frame " + to_string(frame));
            debug_event_status("UPSTREAM Mem_op_event[0] fused LN0/QKV inputs", Mem_op_event[0]);
            debug_event_status("UPSTREAM Mem_op_event[1] Q/K/V output buffers", Mem_op_event[1]);
            debug_event_status("UPSTREAM Mem_op_event[2] frame attention output buffer", Mem_op_event[2]);
            debug_event_status("UPSTREAM K_exe_event[0] linear_qkv_multihead", K_exe_event[0]);
            for (int h = 0; h < NUM_HEADS; h++)
            {
                debug_event_status("UPSTREAM A_exe_event[" + to_string(h) + "] attention head " + to_string(h), A_exe_event[h]);
            }
            debug_event_status("UPSTREAM Mem_op_event[3] projection input buffers", Mem_op_event[3]);
            debug_event_status("UPSTREAM Mem_op_event[4] projection output buffer", Mem_op_event[4]);
            debug_event_status("UPSTREAM K_exe_event[1] projection_out", K_exe_event[1]);
            debug_event_status("UPSTREAM Mem_op_event[6] attention residual output buffer", Mem_op_event[6]);
            debug_event_status("UPSTREAM K_exe_event[2] attention residual", K_exe_event[2]);
            debug_event_status("UPSTREAM Mem_op_event[10] LN1 gamma/beta buffers", Mem_op_event[10]);
            debug_event_status("UPSTREAM Mem_op_event[10] fused LN1+MLP gamma/beta buffers", Mem_op_event[10]);
            debug_event_status("UPSTREAM mlp_output_alloc_event", mlp_output_alloc_event);
            for (size_t i = 0; i < mlp_tile_events.size(); i++)
            {
                debug_event_status("UPSTREAM mlp_tile_event[" + to_string(i) + "]", mlp_tile_events[i]);
            }
            debug_event_status("UPSTREAM K_exe_event[4] MLP residual", K_exe_event[4]);
            debug_event_log("END UPSTREAM SNAPSHOT");

            debug_event_status("BEFORE wait K_exe_event[4] MLP residual", K_exe_event[4]);
            clWaitForEvents(1, &K_exe_event[4]);
            debug_event_status("AFTER wait K_exe_event[4] MLP residual", K_exe_event[4]);
            cout << "------------MLP compute finish---------------------\n";
            debug_event_log("BEFORE release mlp_output_alloc_event");
            clReleaseEvent(mlp_output_alloc_event);
            debug_event_log("AFTER release mlp_output_alloc_event");
            for (cl_event evt : mlp_tile_events)
            {
                debug_event_status("BEFORE release mlp_tile_event", evt);
                clReleaseEvent(evt);
                debug_event_log("AFTER release mlp_tile_event");
            }

            auto Host_Post_End = chrono::high_resolution_clock::now();
            Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_Post_End - Host_Post_Mid).count();

            // Host event control
            for (int i = 0; i < Nb_Of_Mem_Events; i++)
            {
                if (Mem_op_event[i] != nullptr)
                {
                    Profiling_Memory_Events.push_back(Mem_op_event[i]);
                    Total_Memory_Events++;
                }
            }
            Profiling_Kernel_Events.push_back(K_exe_event[0]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_linear_layernorm_qkv");
            for (int h = 0; h < NUM_HEADS; h++)
            {
                Profiling_Kernel_Events.push_back(A_exe_event[h]);
                Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_attention_head_" + to_string(h));
            }
            Profiling_Kernel_Events.push_back(K_exe_event[1]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_projection_out");
            Profiling_Kernel_Events.push_back(K_exe_event[2]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_attention_residual");
            Profiling_Kernel_Events.push_back(K_exe_event[4]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_mlp_residual");
            Total_Kernel_Events += Nb_Of_Exe_Events - 1;
        }

        debug_event_log("BEFORE clFinish after encoder block " + to_string(block));
        clFinish(Command_Queue);
        debug_event_log("AFTER clFinish after encoder block " + to_string(block));

        for (int frame = 0; frame < FRAME; frame++)
        {
            for (int h = 0; h < NUM_HEADS; h++)
            {
                int idx = frame * NUM_HEADS + h;
                clReleaseMemObject(GlobMem_BUF_head_Q[idx]);
                clReleaseMemObject(GlobMem_BUF_head_K[idx]);
                clReleaseMemObject(GlobMem_BUF_head_V[idx]);
                clReleaseMemObject(GlobMem_BUF_head_Out[idx]);
            }
        }
    }

    cout << endl
         << "HOST_Info: Waiting for application to be completed ..." << endl;
    debug_event_log("BEFORE final clFinish before output read");
    clFinish(Command_Queue);
    debug_event_log("AFTER final clFinish before output read");

    cl_event final_output_read_event = nullptr;
    debug_event_log("BEFORE enqueue final_output_read_event: migrate final spatial encoder output to host");
    cl_mem final_output_buffers[FRAME];
    for (int f = 0; f < FRAME; f++)
    {
        final_output_buffers[f] = GlobMem_BUF_frame_X[f];
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, FRAME, final_output_buffers,
                                         CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, &final_output_read_event);
    debug_event_status("AFTER enqueue final_output_read_event", final_output_read_event);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to copy final spatial encoder output to host, err=" << errCode << endl
             << endl;
        return EXIT_FAILURE;
    }
    debug_event_status("BEFORE wait final_output_read_event", final_output_read_event);
    clWaitForEvents(1, &final_output_read_event);
    debug_event_status("AFTER wait final_output_read_event", final_output_read_event);

// ============================================================================
// Step 6: Processing Output Results
//         o) Store spatial encoder output results to a file
// ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------
    // Step 6.1: Store output Result to the spatial encoder output file
    // ------------------------------------------------------
    char Output_File_Name[] = "spatial_encoder_output.txt";
    cout << "HOST_Info: Store output results in: " << Output_File_Name << endl;

    fstream Output_File;
    Output_File.open(Output_File_Name, ios::out);
    if (!Output_File.is_open())
    {
        cout << endl
             << "HOST-Error: Failed to open the " << Output_File_Name << " file for write" << endl
             << endl;
        return EXIT_FAILURE;
    }
    for (int f = 0; f < FRAME; f++)
    {
        for (int i = 0; i < SEQ_LEN; i++)
        {
            for (int d = 0; d < EMBED_DIM; d++)
            {
                // Output_File << Spatial_Encoder_Output[i][d] << " ";
                Output_File << X[f * SEQ_LEN + i][d] << " ";
            }
            Output_File << endl;
        }
    }
    Output_File.close();
    cout << "HOST-Info: Host post-processing time (concat + residual + MLP) = "
         << Host_Post_Time_ms << " ms" << endl;
    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    int Nb_Of_Kernels = static_cast<int>(Profiling_Kernel_Events.size());
    int Nb_Of_Memory_Tranfers = static_cast<int>(Profiling_Memory_Events.size());
    if (Nb_Of_Kernels > 0 && Nb_Of_Memory_Tranfers > 0)
    {
        run_custom_profiling(
            Nb_Of_Kernels,
            Nb_Of_Memory_Tranfers,
            Profiling_Kernel_Events.data(),
            Profiling_Memory_Events.data(),
            Profiling_Kernel_Names.data());
    }

    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

    for (cl_event evt : Profiling_Memory_Events)
    {
        debug_event_status("BEFORE release profiling memory event", evt);
        clReleaseEvent(evt);
        debug_event_log("AFTER release profiling memory event");
    }
    for (cl_event evt : Profiling_Kernel_Events)
    {
        debug_event_status("BEFORE release profiling kernel event", evt);
        clReleaseEvent(evt);
        debug_event_log("AFTER release profiling kernel event");
    }
    debug_event_status("BEFORE release final_output_read_event", final_output_read_event);
    clReleaseEvent(final_output_read_event);
    debug_event_log("AFTER release final_output_read_event");

    for (int f = 0; f < FRAME; f++)
    {
        clReleaseMemObject(GlobMem_BUF_frame_X[f]);
        clReleaseMemObject(GlobMem_BUF_frame_X_norm[f]);
        clReleaseMemObject(GlobMem_BUF_frame_LayerNorm_Scaler[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Q[f]);
        clReleaseMemObject(GlobMem_BUF_frame_K[f]);
        clReleaseMemObject(GlobMem_BUF_frame_V[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Attention_Out[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Attention_Projected[f]);
        clReleaseMemObject(GlobMem_BUF_frame_After_Attention_Residual_Float[f]);
    }
    clReleaseMemObject(GlobMem_BUF_X);
    clReleaseMemObject(GlobMem_BUF_X_norm);
    clReleaseMemObject(GlobMem_BUF_Attention_Out);
    clReleaseMemObject(GlobMem_BUF_Attention_Projected);
    clReleaseMemObject(GlobMem_BUF_After_Attention_Residual_Float);

    clReleaseMemObject(GlobMem_BUF_LayerNorm_Gamma);
    clReleaseMemObject(GlobMem_BUF_LayerNorm_Beta);
    clReleaseMemObject(GlobMem_BUF_LayerNorm1_Gamma);
    clReleaseMemObject(GlobMem_BUF_LayerNorm1_Beta);
    clReleaseMemObject(GlobMem_BUF_LayerNorm_Scaler);

    clReleaseMemObject(GlobMem_BUF_Wq);
    clReleaseMemObject(GlobMem_BUF_bq);
    clReleaseMemObject(GlobMem_BUF_scale_wq);
    clReleaseMemObject(GlobMem_BUF_Wk);
    clReleaseMemObject(GlobMem_BUF_bk);
    clReleaseMemObject(GlobMem_BUF_scale_wk);
    clReleaseMemObject(GlobMem_BUF_Wv);
    clReleaseMemObject(GlobMem_BUF_bv);
    clReleaseMemObject(GlobMem_BUF_scale_wv);
    clReleaseMemObject(GlobMem_BUF_Wout);
    clReleaseMemObject(GlobMem_BUF_bout);
    clReleaseMemObject(GlobMem_BUF_scale_wout);
    clReleaseMemObject(GlobMem_BUF_MLP_Output);
    clReleaseMemObject(GlobMem_BUF_MLP_W1);
    clReleaseMemObject(GlobMem_BUF_MLP_W2);
    clReleaseMemObject(GlobMem_BUF_MLP_b1);
    clReleaseMemObject(GlobMem_BUF_MLP_b2);
    clReleaseMemObject(GlobMem_BUF_MLP_scale_w1);
    clReleaseMemObject(GlobMem_BUF_MLP_scale_w2);
    clReleaseMemObject(GlobMem_BUF_Q);
    clReleaseMemObject(GlobMem_BUF_K);
    clReleaseMemObject(GlobMem_BUF_V);

    for (int i = 0; i < 2; i++)
        clReleaseKernel(K_layernorm[i]);
    clReleaseKernel(K_linear_qkv_multihead);
    for (int h = 0; h < NUM_HEADS; h++)
        clReleaseKernel(K_attention_opt[h]);
    clReleaseKernel(K_projection_out);
    clReleaseKernel(K_residual);
    clReleaseKernel(K_mlp_q);

    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    free(Platform_IDs);
    free(Device_IDs);
    free(X);
    free(LayerNorm_gamma);
    free(LayerNorm_beta);
    free(X_norm);
    free(LayerNorm_scaler);
    free(Wq);
    free(Wq_Linear_Packed);
    free(bq);
    free(scale_wq);
    free(Wk);
    free(Wk_Linear_Packed);
    free(bk);
    free(scale_wk);
    free(Wv);
    free(Wv_Linear_Packed);
    free(bv);
    free(scale_wv);
    free(Wout);
    free(bout);
    free(scale_wout);
    free(LayerNorm1_gamma);
    free(LayerNorm1_beta);
    free(Q);
    free(K_out);
    free(V);
    free(Attention_Out);
    free(Attention_Concat);
    free(Attention_Projected);
    free(After_Attention_Residual_Float);
    free(MLP_Output_Packed);
    free(MLP_W1_Packed);
    free(MLP_W2_Packed);
    free(MLP_W1);
    free(MLP_b1);
    free(MLP_scale_w1);
    free(MLP_W2);
    free(MLP_b2);
    free(MLP_scale_w2);
    free(xclbin_Memory);

    cout << endl
         << "HOST-Info: DONE" << endl
         << endl;

    return EXIT_SUCCESS;
}
