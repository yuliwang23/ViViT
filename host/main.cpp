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
#include <map>
#include <limits>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>      // std::thread
#include <atomic>      // std::atomic
#include <dirent.h>
#include <sys/stat.h>

// openmp
#include <omp.h>

using namespace std;


#include <CL/cl.h>
#include <CL/cl_ext_xilinx.h>
#include "help_functions.h"
#include "host.h"
#include "utility.h"
#include "temporal_layer.h"

#define ALL_MESSAGES
#define HOST_DEBUG_SERIAL_STAGES 0
#define SYN_READ 0
#define HOST_DEBUG_READBACK 0
#define USE_KERNEL_LN 0
#define DEBUG_MSG 0
#define USE_HOST_LN 1
#define DO_TEMPORAL 1

#define BUF_COPY 0

constexpr int MLP_KERNEL_SEQ_LEN = SEQ_LEN;
constexpr int MLP_CHUNK_U32 = 8;
constexpr int MLP_OUTPUT_CHUNKS = (SEQ_LEN * EMBED_DIM) / MLP_CHUNK_U32;
constexpr int MLP_W1_CHUNKS = (MLP_DIM * EMBED_DIM) / 32;
constexpr int MLP_W2_CHUNKS = (EMBED_DIM * MLP_DIM) / 32;

constexpr size_t OpenCL_Subbuffer_Alignment = 4096;
constexpr size_t single_qkv_size = EMBED_DIM * EMBED_DIM * sizeof(int8_t);
constexpr size_t single_mlp_w_size = EMBED_DIM * MLP_DIM * sizeof(int8_t);
constexpr size_t QKV_Stride = 589824;
constexpr size_t MLP_W1_Stride = 2359296;
constexpr size_t MLP_W2_Stride = 2359296;
constexpr size_t MLP_Packed_Stride = 2359296;  
constexpr size_t Bias_Stride = 12288; 
constexpr size_t QKV_Stride_rows = 768;
constexpr size_t MLP_W1_Stride_rows = 768;
constexpr size_t MLP_W2_Stride_rows = 3072;
constexpr size_t MLP_Packed_Stride_rows = 73728;
constexpr size_t FLOAT_VEC_Stride = 4096 / sizeof(float);

constexpr int MEM_BANK_MHA = 6;
constexpr int MEM_BANK_PROJ = 6;
constexpr int MEM_BANK_FFN1 = 6;
constexpr int MEM_BANK_FFN2 = 6;
constexpr int MHA_CU = 6;
constexpr int PROJ_CU = 6;
constexpr int FFN1_CU = 6;
constexpr int FFN2_CU = 6;

constexpr unsigned int MEM_BANKS_MHA[MEM_BANK_MHA] = {0, 4, 8, 12, 16, 20}; // for params
constexpr unsigned int MEM_BANKS_PROJ[MEM_BANK_PROJ] = {0, 4, 8, 12, 16, 20}; // 參數放哪些hbm bank
constexpr unsigned int MEM_BANKS_FFN1[MEM_BANK_FFN1] = {0, 4, 8, 12, 16, 20}; // 每個參數buffer會開[MEM_BANK_PROJ]個
constexpr unsigned int MEM_BANKS_FFN2[MEM_BANK_FFN2] = {0, 4, 8, 12, 16, 20};
constexpr unsigned int MEM_PARAMS_MHA[MHA_CU] = {0, 1, 2, 3, 4, 5}; // use which mem. lane
constexpr unsigned int MEM_PARAMS_PROJ[PROJ_CU] = {0, 1, 2, 3, 4, 5}; // 該CU用第幾個參數buffer
constexpr unsigned int MEM_PARAMS_FFN1[FFN1_CU] = {0, 1, 2, 3, 4, 5};
constexpr unsigned int MEM_PARAMS_FFN2[FFN2_CU] = {0, 1, 2, 3, 4, 5};
constexpr unsigned int K_MHA_OUT_BUF[MHA_CU] = {0, 4, 8, 12, 16, 20}; // output to which hbm
constexpr unsigned int K_PROJ_OUT_BUF[PROJ_CU] = {0, 4, 8, 12, 16, 20}; // 該CU的output要輸出到哪個buffer
constexpr unsigned int K_FFN1_OUT_BUF[FFN1_CU] = {0, 4, 8, 12, 16, 20}; // (某些output buffer有兩個要注意)
constexpr unsigned int K_FFN2_OUT_BUF[FFN2_CU] = {0, 4, 8, 12, 16, 20};
constexpr unsigned int K_MHA_IN_CU[MHA_CU] = {0, 1, 2, 3, 4, 5}; // from which cu
constexpr unsigned int K_PROJ_IN_CU[PROJ_CU] = {0, 1, 2, 3, 4, 5}; // 該CU的屬於哪個pipeline (哪一組CU)
constexpr unsigned int K_FFN1_IN_CU[FFN1_CU] = {0, 1, 2, 3, 4, 5};
constexpr unsigned int K_FFN2_IN_CU[FFN2_CU] = {0, 1, 2, 3, 4, 5};
constexpr int MHA_ATTENTION_ARGS_NUM = 14;
constexpr int MHA_OUTPUT_PROJECTION_ARGS_NUM = 7;
constexpr int FFN1_ARGS_NUM = 8;
constexpr int FFN2_ARGS_NUM = 7;

int success_cnt = 0;
int total_cnt = 0;
constexpr int MAX_BATCH_VIDEOS = 1000;
double Host_Post_Time_ms = 0.0;
double Total_Kernel_Time_ms = 0.0;
int Total_Kernel_Events = 0;
int Total_Memory_Events = 0;
vector<int> each_frame_kernel_profile;
vector<int> each_frame_host_profile;
vector<cl_event> Profiling_Kernel_Events;
vector<cl_event> Profiling_Memory_Events;
vector<string> Profiling_Kernel_Names;

#define OCL_CHECK(error, call) \
	call; \
    if (error != CL_SUCCESS) { \
        fprintf(stderr, "%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        fprintf(stderr, "See the error code reference:\n"); \
		fprintf(stderr, "https://www.docdroid.net/Ee5RTkS/opencl-error-codes-1x-and-2x-pdf#page=6\n"); \
		fprintf(stderr, "https://registry.khronos.org/OpenCL/sdk/1.0/docs/man/xhtml/errors.html\n"); \
        exit(EXIT_FAILURE); \
    }  


int const pipeline_depth = 5;
float dummy_spatial_out[pipeline_depth][FRAME * SEQ_LEN][EMBED_DIM];
float dummy_label[pipeline_depth][400];
bool pipeline_ctl[pipeline_depth] = {true, true, true, true, true};  


struct HostContext
{
    int video_id;

    // callback 會收到 ReadBuffer 完成的資料
    float (*X_tmp)[EMBED_DIM];
    float *label;

    // LayerNorm
    float *LayerNorm_encoder_1_gamma;
    float *LayerNorm_encoder_1_beta;

    // Temporal embedding
    float *temp_cls_embd;
    float (*temp_pos_embd)[EMBED_DIM];

    // Encoder blocks
    EncoderBlockParams *temp_p;

    // Final LayerNorm
    float *temp_scale;
    float *temp_bias;

    // Output projection
    int8_t (*out_kernel)[400];
    float *out_scale;
    float *out_bias;
    // OpenCL event
    int buffer_idx;
};

class HostWorker {
public:
    HostWorker() : running(true), worker(&HostWorker::Loop, this) {}

    ~HostWorker() {
        Stop();
    }

    void Push(HostContext* ctx) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            jobs.push(ctx);
        }
        cv.notify_one();
    }

    void Stop() {
        running = false;
        cv.notify_one();
        if (worker.joinable())
            worker.join();
    }
    
    void WaitAll() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] {
            return jobs.empty() && !busy;
        });
    }

    // 優雅關閉（等全部做完再停）
    void StopGracefully() {
        WaitAll();          // 先等乾淨
        running = false;
        cv.notify_one();
        if (worker.joinable())
            worker.join();
    }
    


private:
    void Loop() {
        while (running) {

            HostContext* ctx = nullptr;

            {
                std::unique_lock<std::mutex> lock(mutex);

                cv.wait(lock, [&] {
                    return !jobs.empty() || !running;
                });

                if (!running && jobs.empty())
                    break;

                ctx = jobs.front();
                jobs.pop();
                busy = true;
            }

            // HostPostProcess(ctx);
            cout << "Start video_" << ctx->video_id << " post-host process\n";
            float (*X_tmp)[EMBED_DIM] = ctx->X_tmp;
            float *label = ctx->label;

            // LayerNorm
            float *LayerNorm_encoder_1_gamma = ctx->LayerNorm_encoder_1_gamma;
            float *LayerNorm_encoder_1_beta  = ctx->LayerNorm_encoder_1_beta;

            // Temporal embedding
            float *temp_cls_embd = ctx->temp_cls_embd;
            float (*temp_pos_embd)[EMBED_DIM] = ctx->temp_pos_embd;

            // Temporal input
            float temporal_in[TEMP_LEN][EMBED_DIM];
            // Encoder blocks
            EncoderBlockParams *temp_p = ctx->temp_p;

            // Intermediate buffers
            
            float state[TEMP_LEN][EMBED_DIM];
            float norm[TEMP_LEN][EMBED_DIM];
            float attn[TEMP_LEN][EMBED_DIM];
            float mlp[TEMP_LEN][EMBED_DIM];
            float tmp[TEMP_LEN][EMBED_DIM];
            int8_t mlp_in[TEMP_LEN][EMBED_DIM];
            float mlp_in_scale[EMBED_DIM];
            // Final LayerNorm
            float *temp_scale = ctx->temp_scale;
            float *temp_bias = ctx->temp_bias;

            // Output projection
            int8_t (*out_kernel)[400] = ctx->out_kernel;
            float *out_scale = ctx->out_scale;
            float *out_bias = ctx->out_bias;


            int8_t out_in[TEMP_LEN][EMBED_DIM];
            float out_in_scale[TEMP_LEN];
            float vivit_out[400];
            
            auto Host_Start = chrono::high_resolution_clock::now();

            constexpr double kEps = 1.0e-5;

            for (int i = 0; i < SEQ_LEN * FRAME; ++i) {
                double mean = 0.0;
                for (int d = 0; d < EMBED_DIM; ++d) {
                    mean += static_cast<double>(X_tmp[i][d]);
                }
                mean /= static_cast<double>(EMBED_DIM);

                double var = 0.0;
                for (int d = 0; d < EMBED_DIM; ++d) {
                    const double diff = static_cast<double>(X_tmp[i][d]) - mean;
                    var += diff * diff;
                }
                var /= static_cast<double>(EMBED_DIM);

                const double inv_std = 1.0 / std::sqrt(var + kEps);
                for (int d = 0; d < EMBED_DIM; ++d) {
                    const double norm = (static_cast<double>(X_tmp[i][d]) - mean) * inv_std;
                    X_tmp[i][d] = static_cast<float>(static_cast<double>(LayerNorm_encoder_1_gamma[d]) * norm +
                                                static_cast<double>(LayerNorm_encoder_1_beta[d]));
                }
            }

            memset(temporal_in, 0, sizeof(temporal_in));

            for (int j = 0; j < EMBED_DIM; j++){
                temporal_in[0][j] = temp_cls_embd[j];
            }
            for (int i = 0; i < TEMP_LEN-1; i++){
                for (int j = 0; j < EMBED_DIM; j++){
                    temporal_in[i+1][j] = X_tmp[i * SEQ_LEN + 0][j];
                }
            }

            memset(state, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
            memset(norm, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
            memset(attn, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
            memset(mlp, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
            memset(tmp, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
            memset(mlp_in, 0, sizeof(int8_t) * TEMP_LEN * EMBED_DIM);
            memset(mlp_in_scale, 0, sizeof(float) * TEMP_LEN);
            
            

            for (int i = 0; i < TEMP_LEN; i++){
                for (int j = 0; j < EMBED_DIM; j++){
                    state[i][j] = temporal_in[i][j] + temp_pos_embd[i][j];
                }
            }
            
            for (int block = 0; block < NUM_ENCODER_BLOCKS; block++){
                
                EncoderBlockParams p = temp_p[block];
            
                /*
                std::cout << "Running temporal encoder block " << block << " / " << NUM_ENCODER_BLOCKS << std::endl;
                
                if (!vivit_temp::load_Temporal_encoder_block("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer", block, p)){
                    std::cerr << "ERROR: failed to load temporal encoder block " << block << '\n';
                    return EXIT_FAILURE;
                }
                */
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
            
            for (int i = 0; i < TEMP_LEN; i++){
                for (int d = 0; d < EMBED_DIM; d++){
                    temporal_in[i][d] = state[i][d];
                }
            }

            for (int i = 0; i < TEMP_LEN; ++i) {
                double mean = 0.0;
                for (int d = 0; d < EMBED_DIM; ++d) {
                    mean += static_cast<double>(temporal_in[i][d]);
                }
                mean /= static_cast<double>(EMBED_DIM);

                double var = 0.0;
                for (int d = 0; d < EMBED_DIM; ++d) {
                    const double diff = static_cast<double>(temporal_in[i][d]) - mean;
                    var += diff * diff;
                }
                var /= static_cast<double>(EMBED_DIM);

                const double inv_std = 1.0 / std::sqrt(var + kEps);
                for (int d = 0; d < EMBED_DIM; ++d) {
                    const double norm = (static_cast<double>(temporal_in[i][d]) - mean) * inv_std;
                    temporal_in[i][d] = static_cast<float>(static_cast<double>(temp_scale[d]) * norm +
                                                static_cast<double>(temp_bias[d]));
                }
            }


            memset(out_in, 0, sizeof(out_in));
            memset(out_in_scale, 0, sizeof(out_in_scale));
            memset(vivit_out, 0, sizeof(vivit_out));

            
            vivit_temp::quantize_activation_sw(temporal_in, out_in, out_in_scale);
            
            for (int d = 0; d < EMBED_DIM; d++){
                for (int j = 0; j < 400; j++){
                    vivit_out[j] += out_in[0][d] * out_in_scale[0] * out_scale[j] * out_kernel[d][j];
                }
            }
            float vivit_result = std::numeric_limits<float>::lowest();
            int vivit_idx = -1;
            for (int i = 0; i < 400; i++){
                vivit_out[i] += out_bias[i];
                if (vivit_out[i] - vivit_result > 0){
                    vivit_result = vivit_out[i];
                    vivit_idx = i;
                }
            }

            auto Host_End = chrono::high_resolution_clock::now();
            each_frame_host_profile.push_back(chrono::duration<double, std::milli>(Host_End - Host_Start).count());
            Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_End - Host_Start).count();
            
            float label_result = std::numeric_limits<float>::lowest();
            int label_idx = -1;
            for (int i = 0; i < 400; i++){
                if (label[i] - label_result > 0){
                    label_result = label[i];
                    label_idx = i;
                }
            }

            if (label_idx == vivit_idx){
                success_cnt += 1;
            }else{
                double mae = 0.0f;
                for (int i = 0; i < 400; i++){
                    mae += fabs(label[i] - vivit_out[i]);
                }
                mae /= 400;
                cout << "mae error: " << mae << endl;
            }
            total_cnt += 1;
            cout << "ViViT Result: " << vivit_idx << endl;
            cout << "Schedule: " << total_cnt << " / " << MAX_BATCH_VIDEOS << endl;
            cout << "Success: " << success_cnt << endl;
            pipeline_ctl[ctx->buffer_idx] = true;
            delete ctx;               // ← 記得釋放！
            {
                std::lock_guard<std::mutex> lock(mutex);
                busy = false;         // ← 標記處理完畢
                cv.notify_all();      // 通知 WaitAll
        
            }
        }
        return ;
    }

    std::thread worker;
    std::queue<HostContext*> jobs;

    std::mutex mutex;
    std::condition_variable cv;
    bool busy = false;
    bool running;
};

HostWorker gWorker;



void CL_CALLBACK ReadDone(
    cl_event,
    cl_int status,
    void* user_data)
{
    if (status != CL_COMPLETE)
        return;

    gWorker.Push(static_cast<HostContext*>(user_data));
}

bool is_directory(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) return false;
    return S_ISDIR(statbuf.st_mode);
}

bool is_regular_file(const std::string& path) {
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) return false;
    return S_ISREG(statbuf.st_mode);
}

// 從檔名中提取 sample 後面的數字
int extract_sample_index(const std::string& filename) {
    size_t pos = filename.find("sample");
    if (pos == std::string::npos) return -1;
    
    pos += 6; // "sample" 長度
    std::string num_str;
    
    while (pos < filename.size() && isdigit(filename[pos])) {
        num_str += filename[pos];
        ++pos;
    }
    
    return num_str.empty() ? -1 : std::stoi(num_str);
}

void traverse_directory(const std::string& path,
                        std::map<int, std::string>& vivit_files,
                        std::map<int, std::string>& encoder_files) {
    
    DIR* dir = opendir(path.c_str());
    if (!dir) {
        std::cerr << "無法開啟資料夾: " << path << std::endl;
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        
        // 跳過 . 和 ..
        if (name == "." || name == "..") continue;
        
        std::string fullpath = path + "/" + name;

        if (is_directory(fullpath)) {
            traverse_directory(fullpath, vivit_files, encoder_files);
        }
        else if (is_regular_file(fullpath)) {
            int idx = extract_sample_index(name);
            if (idx == -1) continue;

            if (name.find("ViViT_Output") != std::string::npos) {
                vivit_files[idx] = fullpath;
            }
            else if (name.find("_SpatialTransformer__encoder_block_in") != std::string::npos) {
                encoder_files[idx] = fullpath;
            }
        }
    }
    closedir(dir);
}

static cl_kernel create_kernel_for_cu(cl_program program,
                                      const char *kernel_name,
                                      const char *cu_name,
                                      cl_int *err)
{
    const string qualified_name = string(kernel_name) + ":{" + cu_name + "}";
    return clCreateKernel(program, qualified_name.c_str(), err);
}
static cl_mem create_hbm_buffer(cl_context context,
                                cl_mem_flags flags,
                                size_t size,
                                void *host_ptr,
                                unsigned int hbm_bank,
                                cl_int *err)
{
    cl_mem_ext_ptr_t ext;
    memset(&ext, 0, sizeof(ext));
    ext.flags = XCL_MEM_TOPOLOGY | hbm_bank;
    ext.obj = host_ptr;
    ext.param = 0;
    return clCreateBuffer(context, flags | CL_MEM_EXT_PTR_XILINX, size, &ext, err);
}
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

*/



static size_t align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static size_t align_frame_stride(size_t frame_size, size_t origin_alignment, size_t element_alignment)
{
    size_t stride = align_up(frame_size, origin_alignment);
    while ((stride % element_alignment) != 0)
    {
        stride += origin_alignment;
    }
    return stride;
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
	

	cout << "------------------------------------" << endl;
	cout << "|          Read Model Params.      |" << endl;
	cout << "------------------------------------" << endl;
	
    cout << "check size" << endl;
    cout << "Bias_Stride: "
        << align_frame_stride(MLP_DIM * sizeof(float),
                            OpenCL_Subbuffer_Alignment,
                            MLP_DIM * sizeof(float))
        << endl;
    cout << "Bias_Stride_rows: "
        << align_frame_stride(MLP_DIM * sizeof(float),
                            OpenCL_Subbuffer_Alignment,
                            MLP_DIM * sizeof(float))
            / (MLP_DIM * sizeof(float))
        << endl;
    cout << "Alignment: "
        << OpenCL_Subbuffer_Alignment << " bytes" << endl;

	
    float label[400];
    // Attention
    float (*LayerNorm_gamma)[FLOAT_VEC_Stride];                
    float (*LayerNorm_beta)[FLOAT_VEC_Stride];                 

    float (*bq)[FLOAT_VEC_Stride];                             
    float (*scale_wq)[FLOAT_VEC_Stride];  
    float (*bk)[FLOAT_VEC_Stride];                             
    float (*scale_wk)[FLOAT_VEC_Stride];   
    float (*bv)[FLOAT_VEC_Stride];                             
    float (*scale_wv)[FLOAT_VEC_Stride];    
    float (*bout)[FLOAT_VEC_Stride];                           
    float (*scale_wout)[FLOAT_VEC_Stride];                    
    
    int8_t (*Wq)[QKV_Stride_rows][EMBED_DIM];
    int8_t (*Wq_Linear_Packed)[QKV_Stride_rows][EMBED_DIM];

    int8_t (*Wk)[QKV_Stride_rows][EMBED_DIM];
    int8_t (*Wk_Linear_Packed)[QKV_Stride_rows][EMBED_DIM];

    int8_t (*Wv)[QKV_Stride_rows][EMBED_DIM];
    int8_t (*Wv_Linear_Packed)[QKV_Stride_rows][EMBED_DIM];

    int8_t (*Wout)[QKV_Stride_rows][EMBED_DIM];
    int8_t (*Wout_Linear_Packed)[QKV_Stride_rows][EMBED_DIM];

    // MLP
    float (*LayerNorm1_gamma)[FLOAT_VEC_Stride];               
    float (*LayerNorm1_beta)[FLOAT_VEC_Stride];                

    int8_t (*MLP_W1)[MLP_W1_Stride_rows][MLP_DIM];
    int8_t (*MLP_W2)[MLP_W2_Stride_rows][EMBED_DIM];
    int8_t (*MLP_W1_Packed)[MLP_Packed_Stride_rows][32];       
    int8_t (*MLP_W2_Packed)[MLP_Packed_Stride_rows][32];

    float (*MLP_b1)[MLP_DIM];                         
    float (*MLP_scale_w1)[MLP_DIM];                
    float (*MLP_b2)[FLOAT_VEC_Stride];                         
    float (*MLP_scale_w2)[FLOAT_VEC_Stride];                   

    // LayerNorm
    LayerNorm_gamma =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));

    LayerNorm_beta =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    // QKV + Out weights
    bq =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    scale_wq =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    bk = 
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    scale_wk =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    bv =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    scale_wv =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    bout =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    scale_wout =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    
	// QKV proj.

    Wq = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wq_Linear_Packed = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wk = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wk_Linear_Packed = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wv = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wv_Linear_Packed = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wout = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    Wout_Linear_Packed = reinterpret_cast<int8_t (*)[QKV_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(QKV_Stride * NUM_ENCODER_BLOCKS));

    LayerNorm1_gamma =
    reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
        host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    LayerNorm1_beta =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));

    // MLP
    
    MLP_b1 = reinterpret_cast<float (*)[MLP_DIM]>(
        host_aligned_alloc(NUM_ENCODER_BLOCKS * MLP_DIM * sizeof(float)));
    MLP_scale_w1 = reinterpret_cast<float (*)[MLP_DIM]>(
        host_aligned_alloc(NUM_ENCODER_BLOCKS * MLP_DIM * sizeof(float)));

    MLP_b2 =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));

    MLP_scale_w2 =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
    

    MLP_W1 = reinterpret_cast<int8_t (*)[MLP_W1_Stride_rows][MLP_DIM]>(
    host_aligned_alloc(MLP_W1_Stride * NUM_ENCODER_BLOCKS));

    MLP_W2 = reinterpret_cast<int8_t (*)[MLP_W2_Stride_rows][EMBED_DIM]>(
        host_aligned_alloc(MLP_W2_Stride * NUM_ENCODER_BLOCKS));
    MLP_W1_Packed = reinterpret_cast<int8_t (*)[MLP_Packed_Stride_rows][32]>(
        host_aligned_alloc(NUM_ENCODER_BLOCKS * MLP_Packed_Stride));
    MLP_W2_Packed = reinterpret_cast<int8_t (*)[MLP_Packed_Stride_rows][32]>(
        host_aligned_alloc(NUM_ENCODER_BLOCKS * MLP_Packed_Stride));
		
		
	float (*tmp_spatial_out)[EMBED_DIM];
    tmp_spatial_out = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    

    if (!LayerNorm_gamma || !LayerNorm_beta || 
        !Wq || !bq || !scale_wq || !Wk || !bk || !scale_wk || !Wv || !bv || !scale_wv ||
        !Wq_Linear_Packed || !Wk_Linear_Packed || !Wv_Linear_Packed || !Wout_Linear_Packed ||
        !Wout || !bout || !scale_wout || !LayerNorm1_gamma || !LayerNorm1_beta ||
        !MLP_W1_Packed || !MLP_W2_Packed ||
        !MLP_W1 || !MLP_b1 || !MLP_scale_w1 || !MLP_W2 || !MLP_b2 || !MLP_scale_w2)
    {
        cout << endl
             << "HOST-Error: Out of Memory during spatial encoder allocation" << endl
             << endl;
        return EXIT_FAILURE;
    }
	
	cout << "Create TEMPORAL Encoder Params." << endl;
	/*
	float (*LayerNorm1_gamma)[FLOAT_VEC_Stride];               
    float (*LayerNorm1_beta)[FLOAT_VEC_Stride];    
	LayerNorm_gamma =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));

    LayerNorm_beta =
        reinterpret_cast<float (*)[FLOAT_VEC_Stride]>(
            host_aligned_alloc(NUM_ENCODER_BLOCKS * FLOAT_VEC_Stride * sizeof(float)));
	
	*/
	
	float LayerNorm_encoder_1_gamma[EMBED_DIM];                
    float LayerNorm_encoder_1_beta[EMBED_DIM];

	// LayerNorm
	if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/scale.txt", &LayerNorm_encoder_1_gamma[0], EMBED_DIM) ||
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/bias.txt", &LayerNorm_encoder_1_beta[0], EMBED_DIM))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Temporal cls vector" << endl
                << endl;
        return EXIT_FAILURE;
	}
	
	float temp_cls_embd[EMBED_DIM];
    float temp_pos_embd[TEMP_LEN][EMBED_DIM];
	
	if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/cls_TemporalTransformer.txt", temp_cls_embd, EMBED_DIM))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Temporal cls vector" << endl
                << endl;
        return EXIT_FAILURE;
    }
    if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer/posembed_input/pos_embedding.txt", &temp_pos_embd[0][0], TEMP_LEN * EMBED_DIM))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Temporal pos. embedd" << endl
                << endl;
        return EXIT_FAILURE;
    }
	
	float temporal_in[TEMP_LEN][EMBED_DIM];
	
	EncoderBlockParams temp_p[NUM_ENCODER_BLOCKS];
	for (int b = 0; b < NUM_ENCODER_BLOCKS; b++){
		if (!vivit_temp::load_Temporal_encoder_block("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer", b, temp_p[b])){
            std::cerr << "ERROR: failed to load temporal encoder block " << b << '\n';
            return EXIT_FAILURE;
		}
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
    float (*mlp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(mlp_storage.data());
    float (*tmp)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(tmp_storage.data());
    int8_t (*mlp_in)[EMBED_DIM] = reinterpret_cast<int8_t (*)[EMBED_DIM]>(mlp_in_storage.data());
    float mlp_in_scale[TEMP_LEN];
	
	float *temp_bias;
    float *temp_scale;
	
	temp_scale = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    temp_bias = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
	
	if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer/encoder_norm/scale.txt", &temp_scale[0], EMBED_DIM) ||
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer/encoder_norm/bias.txt", &temp_bias[0], EMBED_DIM))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Temporal cls vector" << endl
                << endl;
        return EXIT_FAILURE;
    }
	
    int8_t out_kernel[EMBED_DIM][400];
    float out_scale[400];
    float out_bias[400];
    if (!load_txt_values_as_int8("/home/undergraduate/rjchen25/ViViT/params/all_params/output_projection/kernel_qvalue.txt", &out_kernel[0][0], EMBED_DIM * 400) ||
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/output_projection/kernel_scale.txt", &out_scale[0], 400) || 
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/output_projection/bias.txt", &out_bias[0], 400))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Output Proj. layer" << endl
                << endl;
        return EXIT_FAILURE;
    }



    int8_t out_in[TEMP_LEN][EMBED_DIM];
    float out_in_scale[TEMP_LEN];
    float vivit_out[400];
	
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
                                    : first_existing_path({"/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0710_result/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt",
                                                           "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0712_1000_result_3/sample630_intermediate_SpatialTransformer__encoder_block_in.txt",
                                                           "../params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "ViViT/params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
                                                           "/home/undergraduate/rjchen25/ViViT/params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt"});
    const string video_dir = "";

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

    // ====================== 重疊初始化 ======================

    cl_event params_ready_event[NUM_ENCODER_BLOCKS];
    for (int b = 0; b < NUM_ENCODER_BLOCKS; b++)
        OCL_CHECK(errCode, params_ready_event[b] = clCreateUserEvent(Context, &errCode));
        
    std::atomic<bool> params_ready{false};

    std::thread param_loader([&]() {
        cout << "HOST-Info: [Thread] Start loading all parameters in background..." << endl;

        for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {

            string dir = join_path(
                spatial_params_root,
                "encoderblock_" + to_string(block));

            if (!load_txt_values_as_float(join_path(dir, "LayerNorm_0_scale.txt"),
                                        LayerNorm_gamma[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "LayerNorm_0_bias.txt"),
                                        LayerNorm_beta[block], EMBED_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MultiHeadDotProductAttention_0_query_kernel_qvalue.txt"),
                                        &Wq[block][0][0],
                                        EMBED_DIM * EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_query_bias.txt"),
                                        bq[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_query_kernel_scale.txt"),
                                        scale_wq[block], EMBED_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MultiHeadDotProductAttention_0_key_kernel_qvalue.txt"),
                                        &Wk[block][0][0],
                                        EMBED_DIM * EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_key_bias.txt"),
                                        bk[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_key_kernel_scale.txt"),
                                        scale_wk[block], EMBED_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MultiHeadDotProductAttention_0_value_kernel_qvalue.txt"),
                                        &Wv[block][0][0],
                                        EMBED_DIM * EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_value_bias.txt"),
                                        bv[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_value_kernel_scale.txt"),
                                        scale_wv[block], EMBED_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MultiHeadDotProductAttention_0_out_kernel_qvalue.txt"),
                                        &Wout[block][0][0],
                                        EMBED_DIM * EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_out_bias.txt"),
                                        bout[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MultiHeadDotProductAttention_0_out_kernel_scale.txt"),
                                        scale_wout[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "LayerNorm_1_scale.txt"),
                                        LayerNorm1_gamma[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "LayerNorm_1_bias.txt"),
                                        LayerNorm1_beta[block], EMBED_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MlpBlock_0_Dense_0_kernel_qvalue.txt"),
                                        &MLP_W1[block][0][0],
                                        EMBED_DIM * MLP_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MlpBlock_0_Dense_0_bias.txt"),
                                        MLP_b1[block], MLP_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MlpBlock_0_Dense_0_kernel_scale.txt"),
                                        MLP_scale_w1[block], MLP_DIM) ||

                !load_txt_values_as_int8(join_path(dir, "MlpBlock_0_Dense_1_kernel_qvalue.txt"),
                                        &MLP_W2[block][0][0],
                                        MLP_DIM * EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MlpBlock_0_Dense_1_bias.txt"),
                                        MLP_b2[block], EMBED_DIM) ||

                !load_txt_values_as_float(join_path(dir, "MlpBlock_0_Dense_1_kernel_scale.txt"),
                                        MLP_scale_w2[block], EMBED_DIM))
            {
                cout << endl
                    << "HOST-Error: Failed to load one or more parameter files for encoder block "
                    << block << endl
                    << endl;
                return;
            }

            pack_linear_layernorm_weight(Wq[block], Wq_Linear_Packed[block]);
            pack_linear_layernorm_weight(Wk[block], Wk_Linear_Packed[block]);
            pack_linear_layernorm_weight(Wv[block], Wv_Linear_Packed[block]);
            pack_linear_layernorm_weight(Wout[block], Wout_Linear_Packed[block]);
            pack_mlp_weights(MLP_W1[block], MLP_W2[block], MLP_W1_Packed[block], MLP_W2_Packed[block]);

            OCL_CHECK(errCode,
                errCode = clSetUserEventStatus(params_ready_event[block], CL_COMPLETE));
        }

        params_ready = true;
        cout << "HOST-Info: [Thread] All parameters loaded into memory." << endl;
    });


    cout << "Open other thread\n";

#if !SYN_READ
    // release thread
    if (param_loader.joinable()) {
        param_loader.join();
    }
#endif


    // ====================== 主執行緒做 OpenCL 初始化 ======================


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
    cl_kernel K_mha_attention[MHA_CU],
              K_mha_output_projection[PROJ_CU],
              K_FFN1[FFN1_CU],
              K_FFN2[FFN2_CU];

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Kernels: mha_attention and mha_output_projection ..." << endl;
#endif
    for (int cu = 0; cu < MHA_CU; cu++){
        const string attn_cu_name = "mha_attention_" + to_string(cu + 1);
        OCL_CHECK(errCode, K_mha_attention[cu] = create_kernel_for_cu(Program, "mha_attention", attn_cu_name.c_str(), &errCode));
    }
	for (int cu = 0; cu < PROJ_CU; cu++){
		const string oproj_cu_name = "mha_output_projection_" + to_string(cu + 1);
        OCL_CHECK(errCode, K_mha_output_projection[cu] = create_kernel_for_cu(Program, "mha_output_projection", oproj_cu_name.c_str(), &errCode));
	}


#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Kernels: FFN1 and FFN2 ..." << endl;
#endif
    for (int cu = 0; cu < FFN1_CU; cu++){
        const string ffn1_cu_name = "FFN1_" + to_string(cu + 1);
        OCL_CHECK(errCode, K_FFN1[cu] = create_kernel_for_cu(Program, "FFN1", ffn1_cu_name.c_str(), &errCode));
    }
	for (int cu = 0; cu < FFN2_CU; cu++){
		const string ffn2_cu_name = "FFN2_" + to_string(cu + 1);
		OCL_CHECK(errCode, K_FFN2[cu] = create_kernel_for_cu(Program, "FFN2", ffn2_cu_name.c_str(), &errCode));
	}

#if USE_KERNEL_LN
    cl_kernel K_layernorm;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
#endif
    K_layernorm = clCreateKernel(Program, "layernorm", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_layernorm" << endl
             << endl;
        return EXIT_FAILURE;
    }
#endif

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    //   o) Allocate and initialize spatial encoder input and weights
    //   o) Allocate Q/K/V, attention, MLP, and final output arrays
    //   o) Create Buffers in Global Memory to store FPGA data
    // ================================================================
    // allocate host memory for spatial encoder input, weights, and outputs
    
    // TODO: Wk_Linear_Packed.. transpose at first
    // model params

    // intermediate value
    //float *LayerNorm_scaler;           // FRAME * SEQ_LEN
    //int8_t (*X_norm)[EMBED_DIM];       // FRAME * SEQ_LEN * EMBED_DIM
    //float (*Q)[SEQ_LEN][HEAD_DIM];     // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    //float (*K_out)[SEQ_LEN][HEAD_DIM]; // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    //float (*V)[SEQ_LEN][HEAD_DIM];     // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM

    //float (*Attention_Out)[SEQ_LEN][HEAD_DIM];          // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    //int8_t (*Attention_Out_Quant)[SEQ_LEN][HEAD_DIM];   // FRAME * NUM_HEADS * SEQ_LEN * HEAD_DIM
    //float *Attention_Quant_Scales;                      // FRAME * SEQ_LEN
    //int8_t (*Attention_Concat)[EMBED_DIM];              // FRAME * SEQ_LEN * EMBED_DIM
    //float (*Attention_Projected)[EMBED_DIM];            // FRAME * SEQ_LEN * EMBED_DIM

    float (*X)[EMBED_DIM]; // FRAME * SEQ_LEN * EMBED_DIM
    float (*After_Attention_Residual_Float)[EMBED_DIM]; // FRAME * SEQ_LEN * EMBED_DIM
    uint32_t (*MLP_Output_Packed)[MLP_CHUNK_U32];       // SEQ_LEN * EMBED_DIM fp32 bits


#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // ------------------------------------------------------------------
    // Step 4.1: Allocate and initialize spatial encoder arrays
    // ------------------------------------------------------------------
    
    const size_t frame_size_X = SEQ_LEN * EMBED_DIM * sizeof(float);
    const size_t frame_size_X_norm = SEQ_LEN * EMBED_DIM * sizeof(int8_t);
    const size_t frame_size_scaler = SEQ_LEN * sizeof(float);
    const size_t frame_size_QKV = NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
    const size_t frame_size_Attention_Out = NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float);
    const size_t frame_size_Attention_Projected = SEQ_LEN * EMBED_DIM * sizeof(float);
    const size_t frame_size_After_Attention_Residual_Float = SEQ_LEN * EMBED_DIM * sizeof(float);
    const size_t frame_stride_X = align_frame_stride(frame_size_X, OpenCL_Subbuffer_Alignment, EMBED_DIM * sizeof(float));
    const size_t frame_stride_X_norm = align_frame_stride(frame_size_X_norm, OpenCL_Subbuffer_Alignment, EMBED_DIM * sizeof(int8_t));
    const size_t frame_stride_scaler = align_up(frame_size_scaler, OpenCL_Subbuffer_Alignment);
    const size_t frame_stride_QKV = align_frame_stride(frame_size_QKV, OpenCL_Subbuffer_Alignment, SEQ_LEN * HEAD_DIM * sizeof(float));
    const size_t frame_stride_Attention_Out = align_frame_stride(frame_size_Attention_Out, OpenCL_Subbuffer_Alignment, SEQ_LEN * HEAD_DIM * sizeof(float));
    const size_t frame_stride_Attention_Projected = align_frame_stride(frame_size_Attention_Projected, OpenCL_Subbuffer_Alignment, EMBED_DIM * sizeof(float));
    const size_t frame_stride_After_Attention_Residual_Float = align_frame_stride(frame_size_After_Attention_Residual_Float, OpenCL_Subbuffer_Alignment, EMBED_DIM * sizeof(float));
    const size_t frame_stride_X_rows = frame_stride_X / (EMBED_DIM * sizeof(float));


    

    // config memory
    X = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_stride_X));    
    After_Attention_Residual_Float = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_stride_After_Attention_Residual_Float));
    MLP_Output_Packed = reinterpret_cast<uint32_t (*)[MLP_CHUNK_U32]>(host_aligned_alloc(MLP_OUTPUT_CHUNKS * MLP_CHUNK_U32 * sizeof(uint32_t)));
    
    if (!X || !After_Attention_Residual_Float || !MLP_Output_Packed)
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

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
#endif

    // buffer relative to X
    // cl_mem GlobMem_BUF_X[MEM_BANK], 
    //     GlobMem_BUF_After_Attention_Residual_Float[MEM_BANK];

    cl_mem GlobMem_BUF_frame_X[FRAME],
        GlobMem_BUF_frame_Attention_Out[FRAME],
        GlobMem_BUF_frame_Attention_Max[FRAME],
        GlobMem_BUF_frame_Proj_Out[FRAME],
		GlobMem_BUF_frame_FFN_ITM[FRAME],
		GlobMem_BUF_frame_Gelu_Max[FRAME];

#if BUF_COPY
	cl_mem GlobMem_BUF_frame_X[FRAME],
        GlobMem_BUF_frame_Attention_Out[FRAME],
        GlobMem_BUF_frame_Attention_Max[FRAME],
        GlobMem_BUF_frame_Proj_Out[FRAME],
		GlobMem_BUF_frame_FFN_ITM[FRAME],
		GlobMem_BUF_frame_Gelu_Max[FRAME];
#endif

	/*
    // each kernel need one
	cl_mem GlobMem_BUF_MLP_FP32_Intermediate[MLP_CU], // SEQ_LEN * MLP_DIM
           GlobMem_BUF_HBM_GELU_Max[MLP_CU],          // SEQ_LEN
		   GlobMem_BUF_HBM_Attention_Out[MHA_CU],     // SEQ_LEN * EMBED_DIM
           GlobMem_BUF_HBM_Attention_Max[MHA_CU];     // SEQ_LEN
	*/


    cl_mem GlobMem_BUF_LayerNorm_Gamma[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_LayerNorm_Beta[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];

    cl_mem GlobMem_BUF_Wq[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_bq[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_scale_wq[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];

    cl_mem GlobMem_BUF_Wk[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_bk[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_scale_wk[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];

    cl_mem GlobMem_BUF_Wv[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_bv[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];
    cl_mem GlobMem_BUF_scale_wv[NUM_ENCODER_BLOCKS][MEM_BANK_MHA];

    cl_mem GlobMem_BUF_Wout[NUM_ENCODER_BLOCKS][MEM_BANK_PROJ];
    cl_mem GlobMem_BUF_bout[NUM_ENCODER_BLOCKS][MEM_BANK_PROJ];
    cl_mem GlobMem_BUF_scale_wout[NUM_ENCODER_BLOCKS][MEM_BANK_PROJ];

    // MLP parameters (per block)
    cl_mem GlobMem_BUF_LayerNorm1_Gamma[NUM_ENCODER_BLOCKS][MEM_BANK_FFN1];
    cl_mem GlobMem_BUF_LayerNorm1_Beta[NUM_ENCODER_BLOCKS][MEM_BANK_FFN1];

    cl_mem GlobMem_BUF_MLP_W1[NUM_ENCODER_BLOCKS][MEM_BANK_FFN1];
    cl_mem GlobMem_BUF_MLP_W2[NUM_ENCODER_BLOCKS][MEM_BANK_FFN2];

    cl_mem GlobMem_BUF_MLP_b1[NUM_ENCODER_BLOCKS][MEM_BANK_FFN1];
    cl_mem GlobMem_BUF_MLP_b2[NUM_ENCODER_BLOCKS][MEM_BANK_FFN2];

    cl_mem GlobMem_BUF_MLP_scale_w1[NUM_ENCODER_BLOCKS][MEM_BANK_FFN1];
    cl_mem GlobMem_BUF_MLP_scale_w2[NUM_ENCODER_BLOCKS][MEM_BANK_FFN2];

    // 等待參數載入完成
    // finish and release this work
    const size_t SEQ_LEN_FIX = 1024;
    const size_t SEQ_LEN_MLP_DIM_FIX = 606208;
    
    // ref: clCreateBuffer method
    // https://www.cnblogs.com/willhua/p/9463515.html
    // https://zhuanlan.zhihu.com/p/684133601
    // don't care loading data at later
    for (int f = 0; f < FRAME; f++) {
        //const int hbm_allocate_const = 8;
        const unsigned int Runtime_MHA_Bank  = K_FFN2_OUT_BUF[K_MHA_IN_CU[f / (FRAME / MHA_CU + 1)]];
        const unsigned int Runtime_PROJ_Bank = K_MHA_OUT_BUF[K_PROJ_IN_CU[f / (FRAME / PROJ_CU + 1)]];
        const unsigned int Runtime_FFN1_Bank = K_PROJ_OUT_BUF[K_FFN1_IN_CU[f / (FRAME / FFN1_CU + 1)]];
        const unsigned int Runtime_FFN2_Bank = K_FFN1_OUT_BUF[K_FFN2_IN_CU[f / (FRAME / FFN2_CU + 1)]];

        OCL_CHECK(errCode, GlobMem_BUF_frame_X[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            frame_size_X, X + frame_stride_X_rows * f,
                            Runtime_MHA_Bank, &errCode));
        //cout << "BUF_X_"<< f << " OK\n";
        OCL_CHECK(errCode, GlobMem_BUF_frame_Attention_Out[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE,
                            frame_size_X,
                            NULL,
                            Runtime_PROJ_Bank, &errCode));
        //cout << "BUF_Attention_Out_"<< f << " OK\n";
        OCL_CHECK(errCode, GlobMem_BUF_frame_Attention_Max[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE,
                            SEQ_LEN_FIX * sizeof(float),
                            NULL,
                            Runtime_PROJ_Bank, &errCode));
        //cout << "BUF_Attention_Max_"<< f << " OK\n";

        OCL_CHECK(errCode, GlobMem_BUF_frame_Proj_Out[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                            frame_size_After_Attention_Residual_Float,
                            After_Attention_Residual_Float + f * frame_stride_X_rows,
                            Runtime_FFN1_Bank, &errCode));
        //cout << "BUF_Proj_Out_"<< f << " OK\n";

        OCL_CHECK(errCode, GlobMem_BUF_frame_FFN_ITM[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE,
                            SEQ_LEN_MLP_DIM_FIX * sizeof(float),
                            NULL,
                            Runtime_FFN2_Bank, &errCode));
        //cout << "BUF_FFN_ITM_"<< f << " OK\n";

        OCL_CHECK(errCode, GlobMem_BUF_frame_Gelu_Max[f] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE,
                            SEQ_LEN_FIX * sizeof(float),
                            NULL,
                            Runtime_FFN2_Bank, &errCode));
        //cout << "BUF_Gelu_Max_"<< f << " OK\n";

    }
    

    /*
    for (int lane = 0; lane < MHA_CU; lane++){
        const unsigned int Runtime_HBM_Bank = HBM_BANKS_MHA[K_MHA_BUF[lane]];
        OCL_CHECK(errCode, GlobMem_BUF_HBM_Attention_Out[lane] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE, frame_size_X, NULL, Runtime_HBM_Bank, &errCode));
        OCL_CHECK(errCode, GlobMem_BUF_HBM_Attention_Max[lane] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE, frame_size_scaler, NULL, Runtime_HBM_Bank, &errCode));
    }
    cout << "test here\n";
    for (int lane = 0; lane < MLP_CU; lane++){
        const unsigned int Runtime_HBM_Bank = HBM_BANKS_MLP[K_MLP_BUF[lane]];
        OCL_CHECK(errCode, GlobMem_BUF_MLP_FP32_Intermediate[lane] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE, SEQ_LEN * MLP_DIM * sizeof(float), NULL, Runtime_HBM_Bank, &errCode));
        OCL_CHECK(errCode, GlobMem_BUF_HBM_GELU_Max[lane] =
            create_hbm_buffer(Context, CL_MEM_READ_WRITE, frame_size_scaler, NULL, Runtime_HBM_Bank, &errCode));
    }
    */
    
    // Create Params. Buffer
    // MHA Params
    // TODO: fix here
    
    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {

        for (int lane = 0; lane < MEM_BANK_MHA; lane++) {
            const unsigned int Runtime_HBM_Bank = MEM_BANKS_MHA[lane];

            OCL_CHECK(errCode, GlobMem_BUF_LayerNorm_Gamma[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            LayerNorm_gamma[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_LayerNorm_Beta[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            LayerNorm_beta[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_Wq[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * EMBED_DIM * sizeof(int8_t),
                            Wq_Linear_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_bq[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            bq[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_scale_wq[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            scale_wq[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_Wk[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * EMBED_DIM * sizeof(int8_t),
                            Wk_Linear_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_bk[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            bk[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_scale_wk[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            scale_wk[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_Wv[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * EMBED_DIM * sizeof(int8_t),
                            Wv_Linear_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_bv[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            bv[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_scale_wv[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            scale_wv[block], Runtime_HBM_Bank, &errCode));

        }

		for (int lane = 0; lane < MEM_BANK_PROJ; lane++){
			// proj kernel
			const unsigned int Runtime_HBM_Bank = MEM_BANKS_PROJ[lane];
            OCL_CHECK(errCode, GlobMem_BUF_Wout[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * EMBED_DIM * sizeof(int8_t),
                            Wout_Linear_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_bout[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            bout[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_scale_wout[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            scale_wout[block], Runtime_HBM_Bank, &errCode));
		}

        for (int lane = 0; lane < MEM_BANK_FFN1; lane++) {
            const unsigned int Runtime_HBM_Bank = MEM_BANKS_FFN1[lane];

            OCL_CHECK(errCode, GlobMem_BUF_LayerNorm1_Gamma[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            LayerNorm1_gamma[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_LayerNorm1_Beta[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            LayerNorm1_beta[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_MLP_W1[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            MLP_W1_CHUNKS * 32 * sizeof(int8_t),
                            MLP_W1_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_MLP_b1[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            MLP_DIM * sizeof(float),
                            MLP_b1[block], Runtime_HBM_Bank, &errCode));

			OCL_CHECK(errCode, GlobMem_BUF_MLP_scale_w1[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            MLP_DIM * sizeof(float),
                            MLP_scale_w1[block], Runtime_HBM_Bank, &errCode));
		}

		for (int lane = 0; lane < MEM_BANK_FFN2; lane++){
			const unsigned int Runtime_HBM_Bank = MEM_BANKS_FFN2[lane];
            //cout << "FFN2 Runtime_HBM_Bank: " << Runtime_HBM_Bank << endl;
			OCL_CHECK(errCode, GlobMem_BUF_MLP_W2[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            MLP_W2_CHUNKS * 32 * sizeof(int8_t),
                            MLP_W2_Packed[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_MLP_b2[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            MLP_b2[block], Runtime_HBM_Bank, &errCode));

            OCL_CHECK(errCode, GlobMem_BUF_MLP_scale_w2[block][lane] =
                create_hbm_buffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                            EMBED_DIM * sizeof(float),
                            MLP_scale_w2[block], Runtime_HBM_Bank, &errCode));
        }
    }
    

    // ============================================================================
    // Step 5: Run configured Spatial Transformer encoder blocks in sequence
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run " << NUM_ENCODER_BLOCKS << " Spatial Encoder Blocks" << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    
    
	// Per-block parameter buffer arrays for migration
    cl_mem MHA_BUF_Params[NUM_ENCODER_BLOCKS][11 * MEM_BANK_MHA];
    cl_mem PROJ_BUF_Params[NUM_ENCODER_BLOCKS][3 * MEM_BANK_PROJ];
    cl_mem FFN1_BUF_Params[NUM_ENCODER_BLOCKS][5 * MEM_BANK_FFN1];
    cl_mem FFN2_BUF_Params[NUM_ENCODER_BLOCKS][3 * MEM_BANK_FFN2];

    // Fill MHA parameter pointers for each block
    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {
    int idx = 0;

        for (int lane = 0; lane < MEM_BANK_MHA; lane++) {
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_LayerNorm_Gamma[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_LayerNorm_Beta[block][lane];

            MHA_BUF_Params[block][idx++] = GlobMem_BUF_Wq[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_bq[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_scale_wq[block][lane];

            MHA_BUF_Params[block][idx++] = GlobMem_BUF_Wk[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_bk[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_scale_wk[block][lane];

            MHA_BUF_Params[block][idx++] = GlobMem_BUF_Wv[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_bv[block][lane];
            MHA_BUF_Params[block][idx++] = GlobMem_BUF_scale_wv[block][lane];
        }
    }

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {
        int idx = 0;

        for (int lane = 0; lane < MEM_BANK_PROJ; lane++) {
            PROJ_BUF_Params[block][idx++] = GlobMem_BUF_Wout[block][lane];
            PROJ_BUF_Params[block][idx++] = GlobMem_BUF_bout[block][lane];
            PROJ_BUF_Params[block][idx++] = GlobMem_BUF_scale_wout[block][lane];
        }
    }

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {
    int idx = 0;

        for (int lane = 0; lane < MEM_BANK_FFN1; lane++) {
            FFN1_BUF_Params[block][idx++] = GlobMem_BUF_LayerNorm1_Gamma[block][lane];
            FFN1_BUF_Params[block][idx++] = GlobMem_BUF_LayerNorm1_Beta[block][lane];

            FFN1_BUF_Params[block][idx++] = GlobMem_BUF_MLP_W1[block][lane];
            FFN1_BUF_Params[block][idx++] = GlobMem_BUF_MLP_b1[block][lane];
            FFN1_BUF_Params[block][idx++] = GlobMem_BUF_MLP_scale_w1[block][lane];
        }
    }

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {
        int idx = 0;

        for (int lane = 0; lane < MEM_BANK_FFN2; lane++) {
            FFN2_BUF_Params[block][idx++] = GlobMem_BUF_MLP_W2[block][lane];
            FFN2_BUF_Params[block][idx++] = GlobMem_BUF_MLP_b2[block][lane];
            FFN2_BUF_Params[block][idx++] = GlobMem_BUF_MLP_scale_w2[block][lane];
        }
    }

    // ============================================================================
    // EVENT EXECUTION PIPELINE (PER FRAME, PER BLOCK)
    // ============================================================================
    //   [STAGE 1] K_mha_attention: LN0 + QKV + attention.
    //   [STAGE 2] K_mha_output_projection: output projection + residual.
    //   [STAGE 3] K_FFN1: LN1 + first MLP projection + GELU.
    //   [STAGE 4] K_FFN2: second MLP projection + residual into frame_X.
    // ============================================================================
    /*
    string spatial_encoder_dir = join_path(spatial_params_root, "encoderblock_" + to_string(0));
    cout << "HOST-Info: Loading parameters for encoder block " << 0 << " from " << spatial_encoder_dir << endl;

    // loading weight per block
    */

    
    

#if 1
    // batch start
    
    // find matched file
    
    

    

    std::string root_path = "/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0712_1000_result/";  // ← 修改成你的實際路徑

    std::map<int, std::string> vivit_files;
    std::map<int, std::string> encoder_files;
    std::vector<pair<int, std::string>> video_vector_files;
    std::vector<pair<int, std::string>> encoder_vector_files;

    traverse_directory(root_path, vivit_files, encoder_files);

    // 輸出配對結果
    int paired = 0;
    std::cout << "=== 成功配對的 sample ===\n\n";
    int check_tot = 0;
    for (const auto& p : vivit_files) {
        int idx = p.first;
        auto it = encoder_files.find(idx);
        check_tot++;
        if (it != encoder_files.end()) {
            paired++;
            video_vector_files.push_back({idx, it->second});
            encoder_vector_files.push_back({idx, p.second});
#if DEBUG_MSG
            std::cout << "sample" << idx << " 已配對：\n";
            std::cout << "   ViViT  : " << p.second << "\n";
            std::cout << "   Encoder: " << it->second << "\n\n";
#endif
        }
    }
    cout << "check total: " << check_tot << endl;
    //return 0;

    std::cout << "總共找到 " << paired << " 組成功配對。\n";

    float (*X_compact)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_size_X));

    if (!X_compact)
    {
        cout << endl
             << "HOST-Error: Out of Memory during input staging allocation" << endl
             << endl;
        return EXIT_FAILURE;
    }

    check_tot = video_vector_files.size();
    
    string first_video_path = video_vector_files[0].second;

    if (!load_txt_values_as_float(first_video_path, &X_compact[0][0], FRAME * SEQ_LEN * EMBED_DIM))
    {
        cout << endl
            << "HOST-Error: Failed to load encoder input X from " << first_video_path << endl
            << endl;
        free(X_compact);
        return EXIT_FAILURE;
    }

    

#endif
for (int video = 0; video < check_tot; video++)
{ // 
    cout << "start video_" << video << " inference\n";
    if (total_cnt >= MAX_BATCH_VIDEOS)
        break;

    /*
    int idx = video.first;
    auto it = encoder_files.find(idx);
    if (it == encoder_files.end())
        continue;
    string video_path = it->second;
    string label_path = video.second;
    */
    int idx = video;
    string now_video_path = video_vector_files[video].second;
    string nxt_video_path;
    if (video != check_tot-1)
        nxt_video_path = video_vector_files[video+1].second;
    string label_path = encoder_vector_files[video].second;

        
    memset(X, 0, FRAME * frame_stride_X);
    for (int f = 0; f < FRAME; f++)
    {
        memcpy(X + f * frame_stride_X_rows, X_compact + f * SEQ_LEN, frame_size_X);
    }
    memset(X_compact, 0, FRAME * frame_size_X);
    

    // read nxt one
	
	/*
    std::atomic<bool> parallel_read_video{false};
    std::thread video_loader([&]() {
        cout << "parallel reading " << label_path << endl;
        if (!load_txt_values_as_float(label_path, label, 400)){
            cout << endl
                        << "HOST-Error: Failed to load video label "
                        << endl;
                    return ;
        }
        
        if (video != check_tot-1){
            cout << "parallel reading " << nxt_video_path << endl;
            if (!load_txt_values_as_float(nxt_video_path, &X_compact[0][0], FRAME * SEQ_LEN * EMBED_DIM))
            {
                cout << endl
                    << "HOST-Error: Failed to load encoder input X from " << nxt_video_path << endl
                    << endl;
                return ;
            }
            
        }
        //cout << "thread check\n";
        parallel_read_video = true;
        //cout << "reading done!\n";
    });
	*/
	
    
// #if !SYN_READ
    // release thread
    // if (video_loader.joinable()) {
    //     video_loader.join();
    // }
// #endif
    
    cout << "HOST-Info: Loaded spatial encoder input X from " << now_video_path << endl;

    cout << endl;
	
    cl_event MHA_mem_event[NUM_ENCODER_BLOCKS], 
             PROJ_mem_event[NUM_ENCODER_BLOCKS],
             FFN1_mem_event[NUM_ENCODER_BLOCKS],
             FFN2_mem_event[NUM_ENCODER_BLOCKS];
    cl_event move_x_in;
    cl_event MHA_K_event[NUM_ENCODER_BLOCKS][FRAME];
    cl_event PROJ_K_event[NUM_ENCODER_BLOCKS][FRAME];
    cl_event FFN1_K_event[NUM_ENCODER_BLOCKS][FRAME];
    cl_event FFN2_K_event[NUM_ENCODER_BLOCKS][FRAME];
	cl_event Temporal_Finish;
    // cl_event last_mha_scratch_done[MHA_CU] = {};
    // cl_event last_mlp_scratch_done[MLP_CU] = {};
    
    //cl_event Copy_buf1_event[NUM_ENCODER_BLOCKS][FRAME]; // GlobMem_BUF_frame_After_Attention_Residual_Float_1 -> 2

	
    auto Kernel_Start = chrono::high_resolution_clock::now();
    // Migrate first X
    //debug_event_log("BEFORE enqueue move_x_in: migrate multihead input and parameter buffers");
    errCode = clEnqueueMigrateMemObjects(Command_Queue, FRAME, GlobMem_BUF_frame_X, 0, 0, NULL, &move_x_in);
    clWaitForEvents(1, &move_x_in);
    //debug_event_status("AFTER enqueue move_x_in", move_x_in);
    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++)
    {
        //cout << "block, check" << block << " " << " " << endl;
        for (int cu = 0; cu < MHA_CU; cu++){
            
            const unsigned int lane = MEM_PARAMS_MHA[cu];

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 1, 
                sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 2, 
                sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 3, 
                sizeof(cl_mem), &GlobMem_BUF_Wq[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 4, 
                sizeof(cl_mem), &GlobMem_BUF_Wk[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 5, 
                sizeof(cl_mem), &GlobMem_BUF_Wv[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 6, 
                sizeof(cl_mem), &GlobMem_BUF_bq[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 7, 
                sizeof(cl_mem), &GlobMem_BUF_bk[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 8, 
                sizeof(cl_mem), &GlobMem_BUF_bv[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 9, 
                sizeof(cl_mem), &GlobMem_BUF_scale_wq[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 10, 
                sizeof(cl_mem), &GlobMem_BUF_scale_wk[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[cu], 11, 
                sizeof(cl_mem), &GlobMem_BUF_scale_wv[block][lane]));
            /*

            */
        }
        for (int cu = 0; cu < PROJ_CU; cu++){
            const unsigned int lane = MEM_PARAMS_PROJ[cu];

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[cu], 2,
                sizeof(cl_mem), &GlobMem_BUF_Wout[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[cu], 4,
                sizeof(cl_mem), &GlobMem_BUF_bout[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[cu], 5,
                sizeof(cl_mem), &GlobMem_BUF_scale_wout[block][lane]));
        }

        for (int cu = 0; cu < FFN1_CU; cu++) {
            const unsigned int lane = MEM_PARAMS_FFN1[cu];

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[cu], 1, 
                sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Gamma[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[cu], 2, 
                sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Beta[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[cu], 3, 
                sizeof(cl_mem), &GlobMem_BUF_MLP_W1[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[cu], 4, 
                sizeof(cl_mem), &GlobMem_BUF_MLP_b1[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[cu], 5, 
                sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w1[block][lane]));

        }
        for (int cu = 0; cu < FFN2_CU; cu++){
			const unsigned int lane = MEM_PARAMS_FFN2[cu];

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[cu], 2,
                sizeof(cl_mem), &GlobMem_BUF_MLP_W2[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[cu], 4,
                sizeof(cl_mem), &GlobMem_BUF_MLP_b2[block][lane]));

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[cu], 5,
                sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w2[block][lane]));
        }
		// Enqueue Task
        // Migrate Params & Start
        
        OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, 11 * MEM_BANK_MHA, MHA_BUF_Params[block], 0, 1, &params_ready_event[block], &MHA_mem_event[block]));
        OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, 3 * MEM_BANK_PROJ, PROJ_BUF_Params[block], 0, 1, &params_ready_event[block], &PROJ_mem_event[block]));
        OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, 5 * MEM_BANK_FFN1, FFN1_BUF_Params[block], 0, 1, &params_ready_event[block], &FFN1_mem_event[block]));
        OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, 3 * MEM_BANK_FFN2, FFN2_BUF_Params[block], 0, 1, &params_ready_event[block], &FFN2_mem_event[block]));

        for (int frame = 0; frame < FRAME; frame++){
            
            cl_event mha_attention_wait_list[] = {move_x_in, MHA_mem_event[block]};
            cl_uint mha_attention_wait_count = 2;
            if (block != 0)
                mha_attention_wait_list[0] = FFN2_K_event[block-1][frame];
            //const unsigned int frame_allocate = 8;
            const unsigned int mha_cu = frame / (FRAME / MHA_CU + 1);
            const unsigned int proj_cu = frame / (FRAME / PROJ_CU + 1);
            const unsigned int ffn1_cu = frame / (FRAME / FFN1_CU + 1);
            const unsigned int ffn2_cu = frame / (FRAME / FFN2_CU + 1);

            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[mha_cu], 0, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[mha_cu], 12, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Out[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_attention[mha_cu], 13, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Max[frame]));

            OCL_CHECK(errCode, errCode = clEnqueueTask(Command_Queue, K_mha_attention[mha_cu], mha_attention_wait_count, mha_attention_wait_list, &MHA_K_event[block][frame]));

            cl_event mha_proj_wait_list[] = {MHA_K_event[block][frame], PROJ_mem_event[block]};
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[proj_cu], 0, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Out[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[proj_cu], 1, sizeof(cl_mem), &GlobMem_BUF_frame_Attention_Max[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[proj_cu], 3, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[proj_cu], 6, sizeof(cl_mem), &GlobMem_BUF_frame_Proj_Out[frame]));

            OCL_CHECK(errCode, errCode = clEnqueueTask(Command_Queue, K_mha_output_projection[proj_cu], 2, mha_proj_wait_list, &PROJ_K_event[block][frame]));

            cl_event ffn1_wait_list[] = {PROJ_K_event[block][frame], FFN1_mem_event[block]};
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[ffn1_cu], 0, sizeof(cl_mem), &GlobMem_BUF_frame_Proj_Out[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[ffn1_cu], 6, sizeof(cl_mem), &GlobMem_BUF_frame_FFN_ITM[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN1[ffn1_cu], 7, sizeof(cl_mem), &GlobMem_BUF_frame_Gelu_Max[frame]));

            OCL_CHECK(errCode, errCode = clEnqueueTask(Command_Queue, K_FFN1[ffn1_cu], 2, ffn1_wait_list, &FFN1_K_event[block][frame]));

            cl_event ffn2_wait_list[] = {FFN1_K_event[block][frame], FFN2_mem_event[block]};
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[ffn2_cu], 0, sizeof(cl_mem), &GlobMem_BUF_frame_FFN_ITM[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[ffn2_cu], 1, sizeof(cl_mem), &GlobMem_BUF_frame_Gelu_Max[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[ffn2_cu], 3, sizeof(cl_mem), &GlobMem_BUF_frame_Proj_Out[frame]));
            OCL_CHECK(errCode, errCode = clSetKernelArg(K_FFN2[ffn2_cu], 6, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]));

            //OCL_CHECK(errCode, errCode = clSetKernelArg(K_mha_output_projection[mha_cu], 6, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float_2[frame]));
            OCL_CHECK(errCode, errCode = clEnqueueTask(Command_Queue, K_FFN2[ffn2_cu], 2, ffn2_wait_list, &FFN2_K_event[block][frame]));

        }
		
#if HOST_DEBUG_READBACK
            cout << "[HOST DEBUG]: Read Back GlobMem_BUF_frame_After_Attention_Residual_Float_2 from Encoder_" << to_string(block) << endl;
            cl_event debug_read_attention_wait_list[FRAME];
            cl_event debug_read_attention;
            for (int i = 0; i < FRAME; i++)
                debug_read_attention_wait_list[i] = MHA_Output_Projection_K_event[block][i];

            
            OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, FRAME, GlobMem_BUF_frame_After_Attention_Residual_Float_2,
                                         CL_MIGRATE_MEM_OBJECT_HOST, FRAME, debug_read_attention_wait_list, &debug_read_attention));
            

            clWaitForEvents(1, &debug_read_attention);
            string Output_File_Name_Debug_1 = "spatial_encoder_" + to_string(block) + "_attention_output.txt";
    
            cout << "HOST_Info: Store output results in: " << Output_File_Name_Debug_1 << endl;

            fstream Output_File_Debug_1;
            Output_File_Debug_1.open(Output_File_Name_Debug_1, ios::out);
            if (!Output_File_Debug_1.is_open())
            {
                cout << endl
                    << "HOST-Error: Failed to open the " << Output_File_Name_Debug_1 << " file for write" << endl
                    << endl;
                return EXIT_FAILURE;
            }
            for (int f = 0; f < FRAME; f++)
            {
                for (int i = 0; i < SEQ_LEN; i++)
                {
                    for (int d = 0; d < EMBED_DIM; d++)
                    {
                        Output_File_Debug_1 << After_Attention_Residual_Float[f * frame_stride_X_rows + i][d] << " ";
                    }
                    Output_File_Debug_1 << endl;
                }
            }
            Output_File_Debug_1.close();
#endif
        
        // pipeline read params.

        if (block >= 11)
        {
            //clFinish(Command_Queue);
            //debug_event_log("Finish all spatial encoder block\n");
            break;
        }
        

                
#if HOST_DEBUG_READBACK
            cout << "[HOST DEBUG]: Read Back GlobMem_BUF_frame_X from Encoder_" << to_string(block) << endl;
            cl_event debug_read_mlp_wait_list[FRAME];
            cl_event debug_read_mlp;
            for (int i = 0; i < FRAME; i++)
                debug_read_mlp_wait_list[i] = FFN2_K_event[block][i];
            OCL_CHECK(errCode, errCode = clEnqueueMigrateMemObjects(Command_Queue, FRAME, GlobMem_BUF_frame_X,
                                         CL_MIGRATE_MEM_OBJECT_HOST, FRAME, debug_read_mlp_wait_list, &debug_read_mlp));
            
            clWaitForEvents(1, &debug_read_mlp);
            string Output_File_Name_Debug_2 = "spatial_encoder_" + to_string(block) + "_mlp_output.txt";
    
            cout << "HOST_Info: Store output results in: " << Output_File_Name_Debug_2 << endl;

            fstream Output_File_Debug_2;
            Output_File_Debug_2.open(Output_File_Name_Debug_2, ios::out);
            if (!Output_File_Debug_2.is_open())
            {
                cout << endl
                    << "HOST-Error: Failed to open the " << Output_File_Name_Debug_2 << " file for write" << endl
                    << endl;
                return EXIT_FAILURE;
            }
            for (int f = 0; f < FRAME; f++)
            {
                for (int i = 0; i < SEQ_LEN; i++)
                {
                    for (int d = 0; d < EMBED_DIM; d++)
                    {
                        Output_File_Debug_2 << X[f * frame_stride_X_rows + i][d] << " ";
                    }
                    Output_File_Debug_2 << endl;
                }
            }
            Output_File_Debug_2.close();
#endif

        //clFinish(Command_Queue);
    }
    
    // for (int block = 0; block < NUM_ENCODER_BLOCKS; block++){
    //             if (MHA_mem_event[block] != nullptr)
    //     {
    //         Profiling_Memory_Events.push_back(MHA_mem_event[block]);
    //         Total_Memory_Events++;
    //     }
    //     if (MLP_mem_event[block] != nullptr)
    //     {
    //         Profiling_Memory_Events.push_back(MLP_mem_event[block]);
    //         Total_Memory_Events++;
    //     }
    //     Total_Memory_Events += 2;


    //     for (int f = 0; f < FRAME; f++){
    //         Profiling_Kernel_Events.push_back(MHA_Attention_K_event[block][f]);
    //         Profiling_Kernel_Events.push_back(MHA_Output_Projection_K_event[block][f]);
    //         Profiling_Kernel_Events.push_back(FFN1_K_event[block][f]);
    //         Profiling_Kernel_Events.push_back(FFN2_K_event[block][f]);
    //         Total_Kernel_Events += 4;
    //     }

    // }

#if SYN_READ
    // release thread
	
	/*
    if (param_loader.joinable()) {
        param_loader.join();
    }
	*/
	
	
#endif
	if (video != check_tot-1){
		cout << "parallel reading " << nxt_video_path << endl;
		if (!load_txt_values_as_float(nxt_video_path, &X_compact[0][0], FRAME * SEQ_LEN * EMBED_DIM))
		{
			cout << endl
				<< "HOST-Error: Failed to load encoder input X from " << nxt_video_path << endl
				<< endl;
			return -1;
		}
	}
	
	if (!load_txt_values_as_float(label_path, label, 400)){
		cout << endl
			<< "HOST-Error: Failed to load video label "
			<< endl;
		return -1;
    }
    

    //cout << endl
    //     << "HOST_Info: Waiting for application to be completed ..." << endl;
    //debug_event_log("BEFORE final clFinish before output read");
    clFinish(Command_Queue);
    //debug_event_log("AFTER final clFinish before output read");

    cl_event final_output_read_event = nullptr;
    //debug_event_log("BEFORE enqueue final_output_read_event: migrate final spatial encoder output to host");
    cl_mem final_output_buffers[FRAME];
    for (int f = 0; f < FRAME; f++)
    {
        final_output_buffers[f] = GlobMem_BUF_frame_X[f];
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, FRAME, final_output_buffers,
                                         CL_MIGRATE_MEM_OBJECT_HOST, 0, NULL, &final_output_read_event);
    //debug_event_status("AFTER enqueue final_output_read_event", final_output_read_event);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to copy final spatial encoder output to host, err=" << errCode << endl
             << endl;
        return EXIT_FAILURE;
    }

    //debug_event_status("BEFORE wait final_output_read_event", final_output_read_event);
    clWaitForEvents(1, &final_output_read_event);
    
    cout << "finish video_" << video << " spatial encoder\n";
    auto Kernel_End = chrono::high_resolution_clock::now();
    each_frame_kernel_profile.push_back((double)chrono::duration<double, std::milli>(Kernel_End - Kernel_Start).count());
    Total_Kernel_Time_ms += chrono::duration<double, std::milli>(Kernel_End - Kernel_Start).count();

    //debug_event_status("AFTER wait final_output_read_event", final_output_read_event);

// ============================================================================
// Step 6: Processing Output Results
//         o) Store spatial encoder output results to a file
// ============================================================================

#ifdef ALL_MESSAGES
/*
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
*/
#endif

    // ------------------------------------------------------
    // Step 6.1: Store output Result to the spatial encoder output file
    // ------------------------------------------------------

#if HOST_DEBUG_READBACK
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
                Output_File << X[f * frame_stride_X_rows + i][d] << " ";

            }
            Output_File << endl;
        }
    }
    Output_File.close();
#endif

    // ============================================================================
    // Step Layernorm(AFTER Spatial Encoder): 
    // ============================================================================
    
	//cout << "----------- Doing Spatial Encoder_norm ------------\n";


#if USE_HOST_LN

    

    for (int f = 0; f < FRAME; f++){
        for (int i = 0; i < SEQ_LEN; i++){
            for (int j = 0; j < EMBED_DIM; j++){
                tmp_spatial_out[f * SEQ_LEN + i][j] = X[f * frame_stride_X_rows + i][j];
            }
        }
    }

    int idle_idx = -1;
    bool find_flag = false;
    while (idle_idx == -1 && find_flag == false){
        for (int i = 0; i < pipeline_depth; i++){
            if (pipeline_ctl[i]){
                pipeline_ctl[i] = false;
                idle_idx = i;
                find_flag = true;
                break;
            }
        }
    }
    cout << "using " << idle_idx << "buffer\n";
    for (int f = 0; f < FRAME; f++){
        for (int i = 0; i < SEQ_LEN; i++){
            for (int j = 0; j < EMBED_DIM; j++){
                //tmp_spatial_out[f * SEQ_LEN + i][j] = X[f * frame_stride_X_rows + i][j];
                dummy_spatial_out[idle_idx][f * SEQ_LEN + i][j] = tmp_spatial_out[f * SEQ_LEN + i][j];
            }
        }
    }
    for (int i = 0; i < 400; i++){
        dummy_label[idle_idx][i] = label[i];
    }

    // setting 
    // now here
    auto *ctx = new HostContext;
    // Callback data
    ctx->X_tmp = dummy_spatial_out[idle_idx];
    ctx->label = dummy_label[idle_idx];
    ctx->buffer_idx = idle_idx;

    // LayerNorm
    ctx->LayerNorm_encoder_1_gamma = LayerNorm_encoder_1_gamma;
    ctx->LayerNorm_encoder_1_beta = LayerNorm_encoder_1_beta;
    // Temporal embedding
    ctx->temp_cls_embd = temp_cls_embd;
    ctx->temp_pos_embd = temp_pos_embd;


    // Encoder parameters
    ctx->temp_p = temp_p;
    // Final LayerNorm
    ctx->temp_scale = temp_scale;
    ctx->temp_bias = temp_bias;
    // Output projection
    ctx->out_kernel = out_kernel;
    ctx->out_scale = out_scale;
    ctx->out_bias = out_bias;
    // Basic info
    ctx->video_id = video;
    // OpenCL event
    //ctx->read_event = nullptr;
    cout << "Push video_" << video << " to post-host process\n";
    OCL_CHECK(errCode, errCode = clSetEventCallback(final_output_read_event, CL_COMPLETE, ReadDone, ctx));

    
    /*
    constexpr double kEps = 1.0e-5;

    for (int i = 0; i < SEQ_LEN * FRAME; ++i) {
        double mean = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            mean += static_cast<double>(tmp_spatial_out[i][d]);
        }
        mean /= static_cast<double>(EMBED_DIM);

        double var = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double diff = static_cast<double>(tmp_spatial_out[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(EMBED_DIM);

        const double inv_std = 1.0 / std::sqrt(var + kEps);
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double norm = (static_cast<double>(tmp_spatial_out[i][d]) - mean) * inv_std;
            tmp_spatial_out[i][d] = static_cast<float>(static_cast<double>(LayerNorm_encoder_1_gamma[d]) * norm +
                                        static_cast<double>(LayerNorm_encoder_1_beta[d]));
        }
    }
    */



#if HOST_DEBUG_READBACK
    char After_spatial_transformer[] = "After_spatial_transformer.txt";
    
    cout << "HOST_Info: Store output results in: " << After_spatial_transformer << endl;

    fstream After_spatial_File;
    After_spatial_File.open(After_spatial_transformer, ios::out);
    if (!After_spatial_File.is_open())
    {
        cout << endl
             << "HOST-Error: Failed to open the " << After_spatial_transformer << " file for write" << endl
             << endl;
        return EXIT_FAILURE;
    }
    for (int f = 0; f < FRAME; f++)
    {
        for (int d = 0; d < EMBED_DIM; d++)
        {
            After_spatial_File << tmp_spatial_out[f * SEQ_LEN + 0][d] << " ";
        }
        After_spatial_File << endl;  
    }
	After_spatial_File.close();
#endif // test1

    //cout << "--------- Encoder norm END ------------\n";

#endif
    // ============================================================================
    // Step Layernorm(AFTER Spatial Encoder) END
    // ============================================================================
	
    // ============================================================================
    // Step Temporal: Temporal Encoder
    // ============================================================================
#if DO_TEMPORAL & USE_HOST_LN
	/*
	memset(temporal_in, 0, sizeof(temporal_in));

    for (int j = 0; j < EMBED_DIM; j++){
		temporal_in[0][j] = temp_cls_embd[j];
	}
    for (int i = 0; i < TEMP_LEN-1; i++){
        for (int j = 0; j < EMBED_DIM; j++){
            temporal_in[i+1][j] = tmp_spatial_out[i * SEQ_LEN + 0][j];
        }
    }
    
	memset(state, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
	memset(norm, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
	memset(attn, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
	memset(mlp, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
	memset(tmp, 0, sizeof(float) * TEMP_LEN * EMBED_DIM);
	memset(mlp_in, 0, sizeof(int8_t) * TEMP_LEN * EMBED_DIM);
	memset(mlp_in_scale, 0, sizeof(float) * TEMP_LEN);
    
    

    for (int i = 0; i < TEMP_LEN; i++){
        for (int j = 0; j < EMBED_DIM; j++){
            state[i][j] = temporal_in[i][j] + temp_pos_embd[i][j];
        }
    }
	
	*/
    
#if HOST_DEBUG_READBACK
    const auto start = std::chrono::steady_clock::now();
#endif
    /*
    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++){
		
		EncoderBlockParams p = temp_p[block];

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
	*/
	
#if HOST_DEBUG_READBACK
    const auto stop = std::chrono::steady_clock::now();

    std::string temp_output_path = "temporal_encoder_output.txt";
    std::ofstream fout(temp_output_path);
    if (!fout.is_open()) {
        std::cerr << "ERROR: cannot write " << temp_output_path << '\n';
        return EXIT_FAILURE;
    }
    fout << std::setprecision(9);
    for (int i = 0; i < TEMP_LEN; i++){
        for (int d = 0; d < EMBED_DIM; ++d) {
            fout << state[i][d] << ((d + 1 == EMBED_DIM) ? '\n' : ' ');
        }
    }
	
	fout.close();

    double elapsed = std::chrono::duration<double>(stop - start).count();
    std::cout << "Wrote temporal encoder output to " << temp_output_path << '\n'
            << "Processed " << NUM_ENCODER_BLOCKS
            << " encoder blocks in " << elapsed << " seconds\n";
#endif

    /*
    for (int i = 0; i < TEMP_LEN; i++){
        for (int d = 0; d < EMBED_DIM; d++){
            temporal_in[i][d] = state[i][d];
        }
    }

    for (int i = 0; i < TEMP_LEN; ++i) {
        double mean = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            mean += static_cast<double>(temporal_in[i][d]);
        }
        mean /= static_cast<double>(EMBED_DIM);

        double var = 0.0;
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double diff = static_cast<double>(temporal_in[i][d]) - mean;
            var += diff * diff;
        }
        var /= static_cast<double>(EMBED_DIM);

        const double inv_std = 1.0 / std::sqrt(var + kEps);
        for (int d = 0; d < EMBED_DIM; ++d) {
            const double norm = (static_cast<double>(temporal_in[i][d]) - mean) * inv_std;
            temporal_in[i][d] = static_cast<float>(static_cast<double>(temp_scale[d]) * norm +
                                        static_cast<double>(temp_bias[d]));
        }
    }

    //cout << "--------- Temporal Encoder norm END ------------\n";

    memset(out_in, 0, sizeof(out_in));
    memset(out_in_scale, 0, sizeof(out_in_scale));
    memset(vivit_out, 0, sizeof(vivit_out));

    
    vivit_temp::quantize_activation_sw(temporal_in, out_in, out_in_scale);
    
    for (int d = 0; d < EMBED_DIM; d++){
        for (int j = 0; j < 400; j++){
            vivit_out[j] += out_in[0][d] * out_in_scale[0] * out_scale[j] * out_kernel[d][j];
        }
    }
    float vivit_result = std::numeric_limits<float>::lowest();
    int vivit_idx = -1;
    for (int i = 0; i < 400; i++){
        vivit_out[i] += out_bias[i];
        if (vivit_out[i] - vivit_result > 0){
            vivit_result = vivit_out[i];
            vivit_idx = i;
        }
    }

    auto Host_End = chrono::high_resolution_clock::now();
    each_frame_host_profile.push_back(chrono::duration<double, std::milli>(Host_End - Host_Start).count());
    Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_End - Host_Start).count();

    */
	
#if HOST_DEBUG_READBACK

    char Pre_logit_output[] = "Pre_logit_output.txt";
    
    cout << "HOST_Info: Store output results in: " << Pre_logit_output << endl;

    fstream pre_logit_file; 
    pre_logit_file.open(Pre_logit_output, ios::out);
    if (!pre_logit_file.is_open())
    {
        cout << endl
             << "HOST-Error: Failed to open the " << Pre_logit_output << " file for write" << endl
             << endl;
        return EXIT_FAILURE;
    }
    for (int f = 0; f < 400; f++)
    {
        pre_logit_file << vivit_out[f] << " ";
    }
	pre_logit_file.close();
#endif
    // ===================== 等待所有 post-process 完成 =====================
    
    // =====================================================================
    
    // release thread
    // if (video_loader.joinable()) {
    //     video_loader.join();
    // }
    /*
    float label_result = std::numeric_limits<float>::lowest();
    int label_idx = -1;
    for (int i = 0; i < 400; i++){
        if (label[i] - label_result > 0){
            label_result = label[i];
            label_idx = i;
        }
    }

    if (label_idx == vivit_idx){
        success_cnt += 1;
    }else{
		double mae = 0.0f;
		for (int i = 0; i < 400; i++){
			mae += fabs(label[i] - vivit_out[i]);
		}
		mae /= 400;
		cout << "mae error: " << mae << endl;
	}
    total_cnt += 1;
    cout << "ViViT Result: " << vivit_idx << endl;
	cout << "Schedule: " << total_cnt << " / " << MAX_BATCH_VIDEOS << endl;
	cout << "Success: " << success_cnt << endl;
    */


    for (int b = 0; b < NUM_ENCODER_BLOCKS; b++){
        //cout << "free block: " << b << endl; 
        clReleaseEvent(MHA_mem_event[b]);
        clReleaseEvent(PROJ_mem_event[b]);
        clReleaseEvent(FFN1_mem_event[b]);
        clReleaseEvent(FFN2_mem_event[b]);
        for (int f = 0; f < FRAME; f++){
            //cout << "free frame: " << f << endl;
            clReleaseEvent(MHA_K_event[b][f]);
            clReleaseEvent(PROJ_K_event[b][f]);
            clReleaseEvent(FFN1_K_event[b][f]);
            clReleaseEvent(FFN2_K_event[b][f]);
        }
    }
    clReleaseEvent(move_x_in);

}
// batch end
#endif
    cout << "HOST-Info: Waiting for all host post-processing to finish..." << endl;
    gWorker.WaitAll();          // 或 gWorker.StopGracefully();
    cout << "HOST-Info: All host post-processing done." << endl;
    float success_rate = ((float)success_cnt) / total_cnt;
    cout << "Success rate: " << success_rate << endl;
    // ============================================================================
    // Temporal Transformer Encoder END
    // ============================================================================
        

    
    cout << "HOST-Info: Host post-processing time (concat + residual + MLP) = "
         << Host_Post_Time_ms << " ms" << endl;
    cout << "HOST-Info: Kernel post-processing time = "
         << Total_Kernel_Time_ms << " ms" << endl;


    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    int Nb_Of_Kernels = static_cast<int>(Profiling_Kernel_Events.size());
    int Nb_Of_Memory_Tranfers = static_cast<int>(Profiling_Memory_Events.size());
    /*
    if (Nb_Of_Kernels > 0 && Nb_Of_Memory_Tranfers > 0)
    {
        run_custom_profiling(
            Nb_Of_Kernels,
            Nb_Of_Memory_Tranfers,
            Profiling_Kernel_Events.data(),
            Profiling_Memory_Events.data(),
            Profiling_Kernel_Names.data());
    }
    */


    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

    
    
	/*
    debug_event_status("BEFORE release final_output_read_event", final_output_read_event);
    clReleaseEvent(final_output_read_event);
    debug_event_log("AFTER release final_output_read_event");
	*/
	
	
    for (int f = 0; f < FRAME; f++)
    {
        clReleaseMemObject(GlobMem_BUF_frame_X[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Attention_Out[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Attention_Max[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Proj_Out[f]);
        clReleaseMemObject(GlobMem_BUF_frame_FFN_ITM[f]);
        clReleaseMemObject(GlobMem_BUF_frame_Gelu_Max[f]);
    }
    // clReleaseMemObject(GlobMem_BUF_X);
    // clReleaseMemObject(GlobMem_BUF_After_Attention_Residual_Float);

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++) {
        // MHA parameters
        for (int l = 0; l < MEM_BANK_MHA; l++) {
            if (GlobMem_BUF_LayerNorm_Gamma[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_LayerNorm_Gamma[block][l]);
            if (GlobMem_BUF_LayerNorm_Beta[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_LayerNorm_Beta[block][l]);
            if (GlobMem_BUF_Wq[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_Wq[block][l]);
            if (GlobMem_BUF_bq[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_bq[block][l]);
            if (GlobMem_BUF_scale_wq[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_scale_wq[block][l]);
            if (GlobMem_BUF_Wk[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_Wk[block][l]);
            if (GlobMem_BUF_bk[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_bk[block][l]);
            if (GlobMem_BUF_scale_wk[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_scale_wk[block][l]);
            if (GlobMem_BUF_Wv[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_Wv[block][l]);
            if (GlobMem_BUF_bv[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_bv[block][l]);
            if (GlobMem_BUF_scale_wv[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_scale_wv[block][l]);
        }

        // Output projection parameters (sized MEM_BANK_PROJ)
        for (int l = 0; l < MEM_BANK_PROJ; l++) {
            if (GlobMem_BUF_Wout[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_Wout[block][l]);
            if (GlobMem_BUF_bout[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_bout[block][l]);
            if (GlobMem_BUF_scale_wout[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_scale_wout[block][l]);
        }

        // MLP / FFN1 parameters (sized MEM_BANK_FFN1)
        for (int l = 0; l < MEM_BANK_FFN1; l++) {
            if (GlobMem_BUF_LayerNorm1_Gamma[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_LayerNorm1_Gamma[block][l]);
            if (GlobMem_BUF_LayerNorm1_Beta[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_LayerNorm1_Beta[block][l]);
            if (GlobMem_BUF_MLP_W1[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_W1[block][l]);
            if (GlobMem_BUF_MLP_b1[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_b1[block][l]);
            if (GlobMem_BUF_MLP_scale_w1[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_scale_w1[block][l]);
        }

        // FFN2 parameters (sized MEM_BANK_FFN2)
        for (int l = 0; l < MEM_BANK_FFN2; l++) {
            if (GlobMem_BUF_MLP_W2[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_W2[block][l]);
            if (GlobMem_BUF_MLP_b2[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_b2[block][l]);
            if (GlobMem_BUF_MLP_scale_w2[block][l] != nullptr) clReleaseMemObject(GlobMem_BUF_MLP_scale_w2[block][l]);
        }
    }
        

    for (int f = 0; f < MHA_CU; f++){
        clReleaseKernel(K_mha_attention[f]);
    }
    for (int f = 0; f < PROJ_CU; f++){
        clReleaseKernel(K_mha_output_projection[f]);
    }
    for (int f = 0; f < FFN1_CU; f++){
        clReleaseKernel(K_FFN1[f]);
    }
    for (int f = 0; f < FFN2_CU; f++){
        clReleaseKernel(K_FFN2[f]);
    }

    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    free(Platform_IDs);
    free(Device_IDs);

    // Release host memory (SpatialEncoderParams)

    
    free(X);
    free(After_Attention_Residual_Float);
    free(MLP_Output_Packed);
   

    free(xclbin_Memory);

    cout << endl
         << "HOST-Info: DONE" << endl
         << endl;

    return EXIT_SUCCESS;
}
