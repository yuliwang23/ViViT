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
#include "temporal_layer.h"

#define ALL_MESSAGES
#define HOST_DEBUG_SERIAL_STAGES 1
#define HOST_DEBUG_LN_SELF_TEST 0
#define HOST_DEBUG_READBACK 0
#define USE_KERNEL_LN 0
#define USE_HOST_LN 0
#define DO_TEMPORAL 0

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
                                    : first_existing_path({"/home/undergraduate/rjchen25/test/intermediates/vivit_intermediate_0619_result/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt","../params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt",
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
    cl_kernel K_multihead_attention, K_mlp, K_layernorm;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: multihead_attention ..." << endl;
#endif
    K_multihead_attention = clCreateKernel(Program, "multihead_attention", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_multihead_attention" << endl
             << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: mlp ..." << endl;
#endif
    K_mlp = clCreateKernel(Program, "mlp", &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create K_mlp" << endl
             << endl;
        return EXIT_FAILURE;
    }

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

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    //   o) Allocate and initialize spatial encoder input and weights
    //   o) Allocate Q/K/V, attention, MLP, and final output arrays
    //   o) Create Buffers in Global Memory to store FPGA data
    // ================================================================
    // allocate host memory for spatial encoder input, weights, and outputs
    float (*X)[EMBED_DIM]; // FRAME * SEQ_LEN * EMBED_DIM
    // TODO: Wk_Linear_Packed.. transpose at first
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
    int8_t (*Wout)[EMBED_DIM];               // EMBED_DIM * EMBED_DIM (NUM_HEADS * HEAD_DIM * EMBED_DIM)
    int8_t (*Wout_Linear_Packed)[EMBED_DIM]; // EMBED_DIM * EMBED_DIM, output-major for fused attention
    float *bout;                             // HEAD_NUM * HEAD_DIM
    float *scale_wout;                       // HEAD_NUM * HEAD_DIM
    float *LayerNorm1_gamma;                 // EMBED_DIM
    float *LayerNorm1_beta;                  // EMBED_DIM
    int8_t (*MLP_W1)[MLP_DIM];               // EMBED_DIM * MLP_DIM
    float *MLP_b1;                           // MLP_DIM
    float *MLP_scale_w1;                     // MLP_DIM
    int8_t (*MLP_W2)[EMBED_DIM];             // MLP_DIM * EMBED_DIM
    float *MLP_b2;                           // EMBED_DIM
    float *MLP_scale_w2;                     // EMBED_DIM

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
    const size_t OpenCL_Subbuffer_Alignment = 4096;
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
    Wout_Linear_Packed = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(EMBED_DIM * EMBED_DIM * sizeof(int8_t)));
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


    After_Attention_Residual_Float = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_stride_After_Attention_Residual_Float));
    MLP_Output_Packed = reinterpret_cast<uint32_t (*)[MLP_CHUNK_U32]>(host_aligned_alloc(MLP_OUTPUT_CHUNKS * MLP_CHUNK_U32 * sizeof(uint32_t)));
    MLP_W1_Packed = reinterpret_cast<int8_t (*)[32]>(host_aligned_alloc(MLP_W1_CHUNKS * 32 * sizeof(int8_t)));
    MLP_W2_Packed = reinterpret_cast<int8_t (*)[32]>(host_aligned_alloc(MLP_W2_CHUNKS * 32 * sizeof(int8_t)));

    if (!X || !LayerNorm_gamma || !LayerNorm_beta || 
        !Wq || !bq || !scale_wq || !Wk || !bk || !scale_wk || !Wv || !bv || !scale_wv ||
        !Wq_Linear_Packed || !Wk_Linear_Packed || !Wv_Linear_Packed || !Wout_Linear_Packed ||
        !Wout || !bout || !scale_wout || !LayerNorm1_gamma || !LayerNorm1_beta ||
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
    float (*X_compact)[EMBED_DIM] = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_size_X));
    if (!X_compact)
    {
        cout << endl
             << "HOST-Error: Out of Memory during input staging allocation" << endl
             << endl;
        return EXIT_FAILURE;
    }
    if (!load_txt_values_as_float(x_input_file, &X_compact[0][0], FRAME * SEQ_LEN * EMBED_DIM))
    {
        cout << endl
             << "HOST-Error: Failed to load encoder input X from " << x_input_file << endl
             << endl;
        free(X_compact);
        return EXIT_FAILURE;
    }
    memset(X, 0, FRAME * frame_stride_X);
    for (int f = 0; f < FRAME; f++)
    {
        memcpy(X + f * frame_stride_X_rows, X_compact + f * SEQ_LEN, frame_size_X);
    }
    free(X_compact);
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
    cl_mem GlobMem_BUF_X, 
        GlobMem_BUF_After_Attention_Residual_Float;

    cl_mem GlobMem_BUF_frame_X[FRAME],
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

    GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * frame_stride_X, X, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_X" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_After_Attention_Residual_Float = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * frame_stride_After_Attention_Residual_Float, After_Attention_Residual_Float, &errCode);
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
    for (int f = 0; f < FRAME; f++)
    {
        cl_buffer_region region;

        // ---------------- X ----------------
        region.origin = f * frame_stride_X;
        region.size = frame_size_X;
        GlobMem_BUF_frame_X[f] = clCreateSubBuffer(
            GlobMem_BUF_X, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_X creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }


        // ---------------- Attention Residual Float ----------------
        region.origin = f * frame_stride_After_Attention_Residual_Float;
        region.size = frame_size_After_Attention_Residual_Float;
        GlobMem_BUF_frame_After_Attention_Residual_Float[f] = clCreateSubBuffer(
            GlobMem_BUF_After_Attention_Residual_Float, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_After_Attention_Residual_Float creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }

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
    GlobMem_BUF_Wout = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wout_Linear_Packed, &errCode);
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

    GlobMem_BUF_MLP_Output = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, static_cast<size_t>(MLP_OUTPUT_CHUNKS) * MLP_CHUNK_U32 * sizeof(uint32_t), MLP_Output_Packed, &errCode);
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

    const int Nb_Of_Mem_Events = 4;
    const int Nb_Of_Exe_Events = 3 * FRAME;
    cl_event move_x_in = 0;
    cl_event Mem_op_event[Nb_Of_Mem_Events];
    cl_event K_exe_event[Nb_Of_Exe_Events * FRAME];
    // ============================================================================
    // EVENT EXECUTION PIPELINE (PER FRAME, PER BLOCK)
    // ============================================================================
    //   [STAGE 1]
    //   1. K_multihead_attention: LN0 + QKV + attention + output projection + residual.
    //        waits: Mem_op_event[0], Mem_op_event[1]
    //        produces: K_exe_event[0]
    //   [STAGE 2]
    //   2. K_mlp: LN1 + MLP + residual.
    //        waits: K_exe_event[0], Mem_op_event[2], Mem_op_event[3]
    //        produces: K_exe_event[1]
    //   [STAGE 3]
    //   3. Copy MLP output back into frame_X for the next encoder block.
    //        waits: K_exe_event[1]
    //        produces: Mem_op_event[4]
    // ============================================================================
    string spatial_encoder_dir = join_path(spatial_params_root, "encoderblock_" + to_string(0));
    cout << "HOST-Info: Loading parameters for encoder block " << 0 << " from " << spatial_encoder_dir << endl;

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
                << "HOST-Error: Failed to load one or more parameter files for encoder block " << 0 << endl
                << endl;
        return EXIT_FAILURE;
    }

    pack_linear_layernorm_weight(Wq, Wq_Linear_Packed);
    pack_linear_layernorm_weight(Wk, Wk_Linear_Packed);
    pack_linear_layernorm_weight(Wv, Wv_Linear_Packed);
    pack_linear_layernorm_weight(Wout, Wout_Linear_Packed);
    pack_mlp_weights(MLP_W1, MLP_W2, MLP_W1_Packed, MLP_W2_Packed);

    auto Host_Post_Start = chrono::high_resolution_clock::now();

    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++)
    {
        memset(Mem_op_event, 0, sizeof(Mem_op_event));
        memset(K_exe_event, 0, sizeof(K_exe_event));
        

        cl_mem Multihead_Input_Buffers[] = {
            GlobMem_BUF_LayerNorm_Gamma,
            GlobMem_BUF_LayerNorm_Beta,
            GlobMem_BUF_Wq,
            GlobMem_BUF_Wk,
            GlobMem_BUF_Wv,
            GlobMem_BUF_Wout,
            GlobMem_BUF_bq,
            GlobMem_BUF_bk,
            GlobMem_BUF_bv,
            GlobMem_BUF_bout,
            GlobMem_BUF_scale_wq,
            GlobMem_BUF_scale_wk,
            GlobMem_BUF_scale_wv,
            GlobMem_BUF_scale_wout};
        cl_mem MLP_Param_Buffers[] = {
            GlobMem_BUF_LayerNorm1_Gamma,
            GlobMem_BUF_LayerNorm1_Beta,
            GlobMem_BUF_MLP_W1,
            GlobMem_BUF_MLP_W2,
            GlobMem_BUF_MLP_b1,
            GlobMem_BUF_MLP_b2,
            GlobMem_BUF_MLP_scale_w1,
            GlobMem_BUF_MLP_scale_w2};

        debug_event_log("BEGIN block " + to_string(block) + ", frame " + to_string(0) + " integrated memory migrations");
        if (block == 0)
        {
            debug_event_log("BEFORE enqueue move_x_in: migrate multihead input and parameter buffers");
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_X, 0, 0, NULL, &move_x_in);
            clWaitForEvents(1, &move_x_in);
            debug_event_status("AFTER enqueue move_x_in", move_x_in);
        }
        
        debug_event_log("BEFORE enqueue Mem_op_event[0]: migrate multihead parameter buffers");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 14, Multihead_Input_Buffers, 0, 0, NULL, &Mem_op_event[0]);
        debug_event_status("AFTER enqueue Mem_op_event[0]", Mem_op_event[0]);

        debug_event_log("BEFORE enqueue Mem_op_event[1]: allocate attention residual output buffer");
        errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_After_Attention_Residual_Float,
                                                CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[1]);
        debug_event_status("AFTER enqueue Mem_op_event[1]", Mem_op_event[1]);

        debug_event_log("BEFORE enqueue Mem_op_event[2]: migrate MLP parameter buffers");
        errCode |= clEnqueueMigrateMemObjects(Command_Queue, 8, MLP_Param_Buffers, 0, 0, NULL, &Mem_op_event[2]);
        debug_event_status("AFTER enqueue Mem_op_event[2]", Mem_op_event[2]);

        debug_event_log("BEFORE enqueue Mem_op_event[3]: allocate MLP output buffer");
        errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_MLP_Output,
                                                CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[3]);
        debug_event_status("AFTER enqueue Mem_op_event[3]", Mem_op_event[3]);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                    << "Host-Error: Failed to migrate integrated encoder buffers for block " << block << endl
                    << endl;
            return EXIT_FAILURE;
        }
        
        clFinish(Command_Queue);
        
#if 1
        for (int frame = 0; frame < FRAME; frame++)
        {


            cout << "----- running block " << block << ", frame " << frame << " with integrated kernels -----\n";

            

            errCode = false;
            errCode |= clSetKernelArg(K_multihead_attention, 0, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]);
            errCode |= clSetKernelArg(K_multihead_attention, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma);
            errCode |= clSetKernelArg(K_multihead_attention, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta);
            errCode |= clSetKernelArg(K_multihead_attention, 3, sizeof(cl_mem), &GlobMem_BUF_Wq);
            errCode |= clSetKernelArg(K_multihead_attention, 4, sizeof(cl_mem), &GlobMem_BUF_Wk);
            errCode |= clSetKernelArg(K_multihead_attention, 5, sizeof(cl_mem), &GlobMem_BUF_Wv);
            errCode |= clSetKernelArg(K_multihead_attention, 6, sizeof(cl_mem), &GlobMem_BUF_Wout);
            errCode |= clSetKernelArg(K_multihead_attention, 7, sizeof(cl_mem), &GlobMem_BUF_bq);
            errCode |= clSetKernelArg(K_multihead_attention, 8, sizeof(cl_mem), &GlobMem_BUF_bk);
            errCode |= clSetKernelArg(K_multihead_attention, 9, sizeof(cl_mem), &GlobMem_BUF_bv);
            errCode |= clSetKernelArg(K_multihead_attention, 10, sizeof(cl_mem), &GlobMem_BUF_bout);
            errCode |= clSetKernelArg(K_multihead_attention, 11, sizeof(cl_mem), &GlobMem_BUF_scale_wq);
            errCode |= clSetKernelArg(K_multihead_attention, 12, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
            errCode |= clSetKernelArg(K_multihead_attention, 13, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
            errCode |= clSetKernelArg(K_multihead_attention, 14, sizeof(cl_mem), &GlobMem_BUF_scale_wout);
            errCode |= clSetKernelArg(K_multihead_attention, 15, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float[frame]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set multihead_attention kernel arguments for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            errCode = false;
            errCode |= clSetKernelArg(K_mlp, 0, sizeof(cl_mem), &GlobMem_BUF_frame_After_Attention_Residual_Float[frame]);
            errCode |= clSetKernelArg(K_mlp, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Gamma);
            errCode |= clSetKernelArg(K_mlp, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm1_Beta);
            errCode |= clSetKernelArg(K_mlp, 3, sizeof(cl_mem), &GlobMem_BUF_MLP_W1);
            errCode |= clSetKernelArg(K_mlp, 4, sizeof(cl_mem), &GlobMem_BUF_MLP_W2);
            errCode |= clSetKernelArg(K_mlp, 5, sizeof(cl_mem), &GlobMem_BUF_MLP_b1);
            errCode |= clSetKernelArg(K_mlp, 6, sizeof(cl_mem), &GlobMem_BUF_MLP_b2);
            errCode |= clSetKernelArg(K_mlp, 7, sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w1);
            errCode |= clSetKernelArg(K_mlp, 8, sizeof(cl_mem), &GlobMem_BUF_MLP_scale_w2);
            errCode |= clSetKernelArg(K_mlp, 9, sizeof(cl_mem), &GlobMem_BUF_MLP_Output);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-ERROR: Failed to set mlp kernel arguments for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            

            cl_event multihead_wait_list[] = {Mem_op_event[0], Mem_op_event[1]};
            debug_event_wait_list("BEFORE enqueue K_exe_event[0] multihead_attention", 2, multihead_wait_list);
            errCode = clEnqueueTask(Command_Queue, K_multihead_attention, 2, multihead_wait_list, &K_exe_event[frame * 3 + 0]);
            debug_event_status("AFTER enqueue K_exe_event[0] multihead_attention", K_exe_event[frame * 3 + 0]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit multihead_attention for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
#if HOST_DEBUG_SERIAL_STAGES
            debug_event_status("BEFORE serial wait K_exe_event[0] multihead_attention", K_exe_event[frame * 3 + 0]);
            clWaitForEvents(1, &K_exe_event[frame * 3 + 0]);
            debug_event_status("AFTER serial wait K_exe_event[0] multihead_attention", K_exe_event[frame * 3 + 0]);
#endif

            Host_Post_Start = chrono::high_resolution_clock::now();
            cl_event mlp_wait_list[] = {K_exe_event[frame * 3 + 0], Mem_op_event[2], Mem_op_event[3]};
            debug_event_wait_list("BEFORE enqueue K_exe_event[1] mlp", 3, mlp_wait_list);
            errCode = clEnqueueTask(Command_Queue, K_mlp, 3, mlp_wait_list, &K_exe_event[frame * 3 + 1]);
            debug_event_status("AFTER enqueue K_exe_event[1] mlp", K_exe_event[frame * 3 + 1]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "HOST-Error: Failed to submit mlp for frame: " << frame << ", block: " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }
#if HOST_DEBUG_SERIAL_STAGES
            debug_event_status("BEFORE serial wait K_exe_event[1] mlp", K_exe_event[frame * 3 + 1]);
            clWaitForEvents(1, &K_exe_event[frame * 3 + 1]);
            debug_event_status("AFTER serial wait K_exe_event[1] mlp", K_exe_event[frame * 3 + 1]);
#endif
            
            size_t copy_size = SEQ_LEN * EMBED_DIM * sizeof(float);
            debug_event_log("BEFORE enqueue K_exe_event[2]: copy MLP output to next block input buffer");
            errCode = clEnqueueCopyBuffer(Command_Queue,
                                          GlobMem_BUF_MLP_Output,
                                          GlobMem_BUF_frame_X[frame],
                                          0,
                                          0,
                                          static_cast<size_t>(SEQ_LEN) * EMBED_DIM * sizeof(float),
                                          1,
                                          &K_exe_event[frame * 3 + 1],
                                          &K_exe_event[frame * 3 + 2]);
            debug_event_status("AFTER enqueue K_exe_event[2]", K_exe_event[frame * 3 + 2]);
            if (errCode != CL_SUCCESS)
            {
                cout << endl
                     << "Host-Error: Failed to copy MLP output back to frame input for block " << block << endl
                     << endl;
                return EXIT_FAILURE;
            }

            debug_event_status("BEFORE wait K_exe_event[2] block output copy", K_exe_event[frame * 3 + 2]);
            clWaitForEvents(1, &K_exe_event[frame * 3 + 2]);
            debug_event_status("AFTER wait K_exe_event[2] block output copy", K_exe_event[frame * 3 + 2]);
            cout << "------------integrated spatial encoder block compute finish---------------------\n";

            
        }

        auto Host_Post_End = chrono::high_resolution_clock::now();
        Host_Post_Time_ms += chrono::duration<double, std::milli>(Host_Post_End - Host_Post_Start).count();

        for (int i = 0; i < Nb_Of_Mem_Events; i++)
        {
            if (Mem_op_event[i] != nullptr)
            {
                Profiling_Memory_Events.push_back(Mem_op_event[i]);
                Total_Memory_Events++;
            }
        }

        for (int f = 0; f < FRAME; f++){
            Profiling_Kernel_Events.push_back(K_exe_event[f * 3 + 0]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_multihead_attention");
            Profiling_Kernel_Events.push_back(K_exe_event[f * 3 + 1]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_mlp");
            Profiling_Kernel_Events.push_back(K_exe_event[f * 3 + 2]);
            Profiling_Kernel_Names.push_back("encoderblock_" + to_string(block) + "_copy_buffer");
        }
        
        Total_Kernel_Events += Nb_Of_Exe_Events;

        if (block >= 11)
        {
            debug_event_log("BEFORE clFinish after encoder block " + to_string(block));
            clFinish(Command_Queue);
            debug_event_log("AFTER clFinish after encoder block " + to_string(block));
            break;
        }
        spatial_encoder_dir = join_path(spatial_params_root, "encoderblock_" + to_string(block+1));
        cout << "HOST-Info: Loading parameters for encoder block " << block+1 << " from " << spatial_encoder_dir << endl;

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
        pack_linear_layernorm_weight(Wout, Wout_Linear_Packed);
        pack_mlp_weights(MLP_W1, MLP_W2, MLP_W1_Packed, MLP_W2_Packed);


        debug_event_log("BEFORE clFinish after encoder block " + to_string(block));
        clFinish(Command_Queue);
        debug_event_log("AFTER clFinish after encoder block " + to_string(block));
#endif

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
                Output_File << X[f * frame_stride_X_rows + i][d] << " ";

            }
            Output_File << endl;
        }
    }
    Output_File.close();


    // ============================================================================
    // Step Layernorm(AFTER Spatial Encoder): 
    // ============================================================================
    cout << "----------- Doing Spatial Encoder_norm ------------\n";
    
#if USE_KERNEL_LN
    int8_t (*X_norm)[EMBED_DIM];
    float *LayerNorm_scaler;
    X_norm = reinterpret_cast<int8_t (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * frame_stride_X_norm));
    LayerNorm_scaler = reinterpret_cast<float *>(host_aligned_alloc(FRAME * frame_stride_scaler));
    
    cl_mem GlobMem_BUF_X_norm, GlobMem_BUF_LayerNorm_Scaler;
    cl_mem GlobMem_BUF_frame_X_norm[FRAME], GlobMem_BUF_frame_LayerNorm_Scaler[FRAME];

    GlobMem_BUF_X_norm = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * frame_stride_X_norm, X_norm, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_X_norm" << endl
             << endl;
        return EXIT_FAILURE;
    }
    GlobMem_BUF_LayerNorm_Scaler = clCreateBuffer(Context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, FRAME * frame_stride_scaler, LayerNorm_scaler, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl
             << "Host-Error: Failed to allocate GlobMem_BUF_LayerNorm_Scaler" << endl
             << endl;
        return EXIT_FAILURE;
    }

    for (int f = 0; f < FRAME; f++)
    {
        cl_buffer_region region;

        // ---------------- X_norm ----------------
        region.origin = f * frame_stride_X_norm;
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
        region.origin = f * frame_stride_scaler;
        region.size = frame_size_scaler;
        GlobMem_BUF_frame_LayerNorm_Scaler[f] = clCreateSubBuffer(
            GlobMem_BUF_LayerNorm_Scaler, CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);

        if (errCode != CL_SUCCESS)
        {
            cout << "Host-Error: SubBuffer GlobMem_BUF_frame_LayerNorm_Scaler creation failed at frame " << f << endl;
            return EXIT_FAILURE;
        }
    }

    cl_event ln_mem_op[3] = {};
    cl_event ln_op = 0;
    if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/scale.txt", LayerNorm_gamma, EMBED_DIM) ||
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/bias.txt", LayerNorm_beta, EMBED_DIM)){
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for spatial encoder norm params." << endl
                << endl;
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < 1; frame++){
        memset(ln_mem_op, 0, sizeof(ln_mem_op));
        ln_op = 0;


        cl_mem Ln_input_buffers[] = {
            GlobMem_BUF_LayerNorm_Gamma,
            GlobMem_BUF_LayerNorm_Beta,
            GlobMem_BUF_frame_X[frame]
        };
        cl_mem Ln_output_buffers[] = {
            GlobMem_BUF_frame_X_norm[frame], 
            GlobMem_BUF_frame_LayerNorm_Scaler[frame]
        };

        errCode |= clSetKernelArg(K_layernorm, 0, sizeof(cl_mem), &GlobMem_BUF_frame_X[frame]);
        errCode |= clSetKernelArg(K_layernorm, 1, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Gamma);
        errCode |= clSetKernelArg(K_layernorm, 2, sizeof(cl_mem), &GlobMem_BUF_LayerNorm_Beta);
        errCode |= clSetKernelArg(K_layernorm, 3, sizeof(cl_mem), &GlobMem_BUF_frame_X_norm[frame]);
        errCode |= clSetKernelArg(K_layernorm, 4, sizeof(cl_mem), &GlobMem_BUF_frame_LayerNorm_Scaler[frame]);

        if (frame == 0){
            debug_event_log("BEFORE enqueue ln_mem_op: migrate ln(encoder norm) input buffers");
            errCode = clEnqueueMigrateMemObjects(Command_Queue, 3, Ln_input_buffers, 0, 0, NULL, &ln_mem_op[0]);
            debug_event_status("AFTER enqueue ln_mem_op[0]", ln_mem_op[0]);
        }
    
        debug_event_log("BEFORE enqueue ln_mem_op: migrate ln(encoder norm) output buffers");
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 2, Ln_output_buffers, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 1, &ln_mem_op[0], &ln_mem_op[1]);
        debug_event_status("AFTER enqueue ln_mem_op[1]", ln_mem_op[1]);

        if (errCode != CL_SUCCESS)
        {
            cout << endl
                    << "Host-Error: Failed to migrate ln input / output buffers " << endl
                    << endl;
            return EXIT_FAILURE;
        }
        
        cl_event layernorm_wait_list[] = {ln_mem_op[0], ln_mem_op[1]};

        clWaitForEvents(2, layernorm_wait_list);
        
        debug_event_wait_list("BEFORE enqueue ln_op encoder layernorm", 2, layernorm_wait_list);
        errCode = clEnqueueTask(Command_Queue, K_layernorm, 2, layernorm_wait_list, &ln_op);
        debug_event_status("AFTER enqueue ln_op encoder layernorm", ln_op);
        if (errCode != CL_SUCCESS)
        {
            cout << endl
                    << "HOST-Error: Failed to submit K_layernorm for frame " << frame << endl
                    << endl;
            return EXIT_FAILURE;
        }


        debug_event_status("BEFORE serial wait ln_op multihead_attention", ln_op);
        clWaitForEvents(1, &ln_op);
        debug_event_status("AFTER serial wait ln_op multihead_attention", ln_op);        
    }

    debug_event_log("BEFORE final clFinish before encoder norm read");
    clFinish(Command_Queue);
    debug_event_log("AFTER final clFinish before encoder norm read");

#endif

#if USE_HOST_LN
    float *temp_bias;
    float *temp_scale;
    float (*tmp_spatial_out)[EMBED_DIM];

    tmp_spatial_out = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    temp_scale = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
    temp_bias = reinterpret_cast<float *>(host_aligned_alloc(EMBED_DIM * sizeof(float)));
   
    
    for (int f = 0; f < FRAME; f++){
        for (int i = 0; i < SEQ_LEN; i++){
            for (int j = 0; j < EMBED_DIM; j++){
                tmp_spatial_out[f * SEQ_LEN + i][j] = X[f * frame_stride_X_rows + i][j];
            }
        }
    }


    if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/scale.txt", &temp_scale[0], EMBED_DIM) ||
        !load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/SpatialTransformer/encoder_norm/bias.txt", &temp_bias[0], EMBED_DIM))
    {
        cout << endl
                << "HOST-Error: Failed to load one or more parameter files for Temporal cls vector" << endl
                << endl;
        return EXIT_FAILURE;
    }
    

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
            tmp_spatial_out[i][d] = static_cast<float>(static_cast<double>(temp_scale[d]) * norm +
                                        static_cast<double>(temp_bias[d]));
        }
    }





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

    cout << "--------- Encoder norm END ------------\n";

#endif
    // ============================================================================
    // Step Layernorm(AFTER Spatial Encoder) END
    // ============================================================================

    // ============================================================================
    // Step Temporal: Temporal Encoder
    // ============================================================================
#if DO_TEMPORAL
    float temporal_in[TEMP_LEN][EMBED_DIM];
    float temp_pos_embd[TEMP_LEN][EMBED_DIM];
    for (int i = 0; i < TEMP_LEN-1; i++){
        for (int j = 0; j < EMBED_DIM; j++){
            temporal_in[i+1][j] = tmp_spatial_out[i * SEQ_LEN + 0][j];
        }
    }
    if (!load_txt_values_as_float("/home/undergraduate/rjchen25/ViViT/params/all_params/cls_TemporalTransformer.txt", &temporal_in[0][0], EMBED_DIM))
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

    for (int i = 0; i < TEMP_LEN; i++){
        for (int j = 0; j < EMBED_DIM; j++){
            state[i][j] = temporal_in[i][j] + temp_pos_embd[i][j];
        }
    }
    EncoderBlockParams p;

    const auto start = std::chrono::steady_clock::now();
    for (int block = 0; block < NUM_ENCODER_BLOCKS; block++){
        std::cout << "Running temporal encoder block " << block << " / " << NUM_ENCODER_BLOCKS << std::endl;
        if (!vivit_temp::load_Temporal_encoder_block("/home/undergraduate/rjchen25/ViViT/params/all_params/TemporalTransformer", block, p)){
            std::cerr << "ERROR: failed to load temporal encoder block " << block << '\n';
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

    double elapsed = std::chrono::duration<double>(stop - start).count();
    std::cout << "Wrote temporal encoder output to " << temp_output_path << '\n'
            << "Processed " << NUM_ENCODER_BLOCKS
            << " encoder blocks in " << elapsed << " seconds\n";

#endif
    // ============================================================================
    // Temporal Transformer Encoder END
    // ============================================================================
        

    
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
    clReleaseEvent(move_x_in);
    debug_event_status("BEFORE release final_output_read_event", final_output_read_event);
    clReleaseEvent(final_output_read_event);
    debug_event_log("AFTER release final_output_read_event");

    for (int f = 0; f < FRAME; f++)
    {
        clReleaseMemObject(GlobMem_BUF_frame_X[f]);
        clReleaseMemObject(GlobMem_BUF_frame_After_Attention_Residual_Float[f]);
    }
    clReleaseMemObject(GlobMem_BUF_X);
    clReleaseMemObject(GlobMem_BUF_After_Attention_Residual_Float);

    clReleaseMemObject(GlobMem_BUF_LayerNorm_Gamma);
    clReleaseMemObject(GlobMem_BUF_LayerNorm_Beta);
    clReleaseMemObject(GlobMem_BUF_LayerNorm1_Gamma);
    clReleaseMemObject(GlobMem_BUF_LayerNorm1_Beta);

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


    clReleaseKernel(K_multihead_attention);
    clReleaseKernel(K_mlp);

    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    free(Platform_IDs);
    free(Device_IDs);
    free(X);
    free(LayerNorm_gamma);
    free(LayerNorm_beta);

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
    free(Wout_Linear_Packed);
    free(bout);
    free(scale_wout);
    free(LayerNorm1_gamma);
    free(LayerNorm1_beta);

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