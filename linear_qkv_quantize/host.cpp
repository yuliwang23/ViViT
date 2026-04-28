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
** HOST Code for linear_qkv_multihead (quantized version)
**
** 主要變更（相對於原版 host.cpp）:
**   1. 所有輸入資料改為從文字檔讀入（不再 random 產生）
**   2. 資料型別配合 quantized kernel：
**        X / Wq / Wk / Wv   → int8_t
**        bq / bk / bv        → float
**        scale_input          → float  [SEQ_LEN=197]
**        scale_wq/wk/wv       → float  [EMBED_DIM=768]
**        Q / K / V (output)   → float
**   3. 新增 scale_input / scale_wq / scale_wk / scale_wv 參數
**   4. Kernel argument 數量：14 個（原版 10 個）
**   5. HBM bank 指定：透過 cl_mem_ext_ptr_t + XCL_MEM_TOPOLOGY
**        gmem0 → HBM bank 0  : X, scale_input, bq, bk, bv, scale_wq/wk/wv
**        gmem1 → HBM bank 2  : Wq, Q
**        gmem2 → HBM bank 4  : Wk, K
**        gmem3 → HBM bank 6  : Wv, V
**      ※ bank 編號需與 connectivity.cfg 的 sp= 設定一致，
**        只需修改下方四個 #define 即可。
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
#include <sstream>
#include <cstdlib>

using namespace std;

#include <CL/cl.h>
#include <CL/cl_ext_xilinx.h>   // XCL_MEM_TOPOLOGY

#include "help_functions.h"
#include "linear_qkv_multihead_quan.h"   // SEQ_LEN=197, EMBED_DIM=768, NUM_HEADS=12, HEAD_DIM=64

#define ALL_MESSAGES

// ============================================================
// HBM Bank 編號設定（XCL_MEM_TOPOLOGY）
//   index 對應到 xclbin memory topology 裡的 HBM bank 編號。
//   Alveo U280 每個 HBM pseudo-channel 各佔一個 topology index。
//   請依照你的 connectivity.cfg 中 sp= 的實際 bank 編號調整。
// ============================================================
#define HBM_BANK0  (0  | XCL_MEM_TOPOLOGY)   // gmem0: X, scale_input, b*, scale_w*
#define HBM_BANK2  (2  | XCL_MEM_TOPOLOGY)   // gmem1: Wq, Q
#define HBM_BANK4  (4  | XCL_MEM_TOPOLOGY)   // gmem2: Wk, K
#define HBM_BANK6  (6  | XCL_MEM_TOPOLOGY)   // gmem3: Wv, V

// ============================================================
// Helper: 從文字檔讀 int8 矩陣
//   格式：每行 cols 個整數（以空格分隔），共 rows 行
// ============================================================
static bool load_int8_matrix(const char* filename, int8_t* buf, int rows, int cols)
{
    ifstream f(filename);
    if (!f.is_open()) {
        cerr << "HOST-Error: Cannot open " << filename << endl;
        return false;
    }
    for (int i = 0; i < rows; i++) {
        string line;
        getline(f, line);
        stringstream ss(line);
        for (int j = 0; j < cols; j++) {
            int tmp;
            ss >> tmp;
            buf[i * cols + j] = static_cast<int8_t>(tmp);
        }
    }
    f.close();
    return true;
}

// ============================================================
// Helper: 從文字檔讀 float 向量
//   格式：空格或換行分隔，共 n 個 float 值
// ============================================================
static bool load_float_vector(const char* filename, float* buf, int n)
{
    ifstream f(filename);
    if (!f.is_open()) {
        cerr << "HOST-Error: Cannot open " << filename << endl;
        return false;
    }
    for (int i = 0; i < n; i++) {
        f >> buf[i];
    }
    f.close();
    return true;
}

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
    cout << endl;

    // ============================================================================
    // Step 1: Check Command Line Arguments
    // ============================================================================
    //    o) argv[1] Platform Vendor
    //    o) argv[2] Device Name
    //    o) argv[3] XCLBIN file
    // ============================================================================
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    if (argc != 4)
    {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File>" << endl << endl;
        return EXIT_FAILURE;
    }

    const char* Target_Platform_Vendor   = argv[1];
    const char* Target_Device_Name       = argv[2];
    const char* xclbinFilename           = argv[3];
    cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
    cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;

    // ============================================================================
    // Step 2: Detect Target Platform and Target Device in a system.
    //         Create Context and Command Queue.
    // ============================================================================
    cout << endl;
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
    cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    cl_uint          ui;

    cl_platform_id  *Platform_IDs;
    cl_uint          Nb_Of_Platforms;
    cl_platform_id   Target_Platform_ID;
    bool             Platform_Detected;
    char            *platform_info;

    cl_device_id    *Device_IDs;
    cl_uint          Nb_Of_Devices;
    cl_device_id     Target_Device_ID;
    bool             Device_Detected;
    char            *device_info;

    cl_context       Context;
    cl_command_queue Command_Queue;

    cl_int           errCode;
    size_t           size;

    // ------------------------------------------------------------------------------------
    // Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
    // ------------------------------------------------------------------------------------

    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
        cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
        return EXIT_FAILURE;
    }

    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
    #endif

    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
    if (!Platform_IDs) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl;
        return EXIT_FAILURE;
    }

    // Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++) {

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get the size of the Platofrm parameter " << "CL_PLATFORM_VENDOR" << " value " << endl << endl;
            return EXIT_FAILURE;
        }

        platform_info = new char[size];
        if (!platform_info) {
            cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info , NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl << endl;
            return EXIT_FAILURE;
        }

        // Check if the current platform matches Target_Platform_Vendor
        if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
            Platform_Detected        = true;
            Target_Platform_ID       = Platform_IDs[ui];
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl << endl;
            #endif
        }
    }

    if (Platform_Detected == false) {
        cout << endl << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------------------
    // Step 2.2:  Get All Devices for selected platform Target_Platform_ID
    //            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
    // ------------------------------------------------------------------------------------

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get the number of available Devices" << endl << endl;
        return EXIT_FAILURE;
    }
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of available devices  : " << Nb_Of_Devices << endl;
    #endif

    Device_IDs = new cl_device_id[Nb_Of_Devices];
    if (!Device_IDs) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Device_IDs" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get available Devices" << endl << endl;
        return EXIT_FAILURE;
    }

    // Search for CL_DEVICE_NAME = Target_Device_Name
    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++) {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl << endl;
            return EXIT_FAILURE;
        }

        device_info = new char[size];
        if (!device_info) {
            cout << endl << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl << endl;
            return EXIT_FAILURE;
        }

        // Check if the current device matches Target_Device_Name
        if (strcmp(device_info, Target_Device_Name) == 0) {
            Device_Detected        = true;
            Target_Device_ID       = Device_IDs[ui];
        }
    }

    if (Device_Detected == false) {
        cout << endl << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl << endl;
        return EXIT_FAILURE;
    } else {
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl << endl;
        #endif
    }

    // ------------------------------------------------------------------------------------
    // Step 2.3: Create Context
    // ------------------------------------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Context ... " << endl;
    #endif
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------------------
    // Step 2.4: Create Command Queue (commands are executed in-order)
    // ------------------------------------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Command Queue ... " << endl;
    #endif
    Command_Queue = clCreateCommandQueue(Context, Target_Device_ID,
                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
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

    program_length = loadFile2Memory(xclbinFilename, (char **) &xclbin_Memory);
    if (program_length < 0) {
        cout << endl << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------
    // Step 3.2: Create a program using a Binary File
    // ------------------------------------------------------------
    size_t     Program_Length_in_Bytes;
    cl_program Program;
    cl_int     Binary_Status;

    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Program with Binary ..." << endl;
    #endif
    Program_Length_in_Bytes = program_length;
    Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
                                        (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
        return EXIT_FAILURE;
    }

    // ----------------------------------------------------------------------
    // Step 3.3: Build (compiles and links) a program executable from binary
    // ----------------------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Building the Program ..." << endl;
    #endif

    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------
    // Step 3.4: Create Kernel
    // -------------------------------------------------------------
    cl_kernel K_linear_qkv_multihead;

    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: linear_qkv_multihead ..." << endl;
    #endif
    K_linear_qkv_multihead = clCreateKernel(Program, "linear_qkv_multihead", &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create K_linear_qkv_multihead" << endl << endl;
        return EXIT_FAILURE;
    }

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    // Quantized kernel 輸入/輸出型別：
    //   X            : int8_t [SEQ_LEN=197][EMBED_DIM=768]    ← X.txt
    //   Wq/Wk/Wv     : int8_t [EMBED_DIM=768][EMBED_DIM=768]  ← Wq/Wk/Wv.txt
    //   bq/bk/bv     : float  [EMBED_DIM=768]                 ← bq/bk/bv.txt
    //   scale_input  : float  [SEQ_LEN=197]                   ← scale_input.txt
    //   scale_wq/wk/wv: float [EMBED_DIM=768]                 ← scale_w*.txt
    //   Q/K/V (out)  : float  [NUM_HEADS=12][SEQ_LEN=197][HEAD_DIM=64]
    // ================================================================
    #ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    void *ptr = nullptr;

    // ---- 輸入指標 ----
    int8_t *X;
    int8_t *Wq, *Wk, *Wv;
    float  *bq,  *bk,  *bv;
    float  *scale_input;
    float  *scale_wq, *scale_wk, *scale_wv;

    // ---- 輸出指標 ----
    float  *Q_out, *K_out, *V_out;

    // ------------------------------------------------------------------
    // Step 4.1: Allocate & Load Input from Files
    // ------------------------------------------------------------------

    // X: [SEQ_LEN][EMBED_DIM]  int8  ← X.txt（每行 EMBED_DIM 個整數，共 SEQ_LEN=197 行）
    cout << "HOST-Info: Allocating and loading X from X.txt ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for X array" << endl << endl;
        return EXIT_FAILURE;
    }
    X = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("X.txt", X, SEQ_LEN, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " elements)" << endl;

    // Wq: [EMBED_DIM][EMBED_DIM]  int8  ← Wq.txt
    cout << "HOST-Info: Allocating and loading Wq from Wq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Wq array" << endl << endl;
        return EXIT_FAILURE;
    }
    Wq = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wq.txt", Wq, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // Wk: [EMBED_DIM][EMBED_DIM]  int8  ← Wk.txt
    cout << "HOST-Info: Allocating and loading Wk from Wk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Wk array" << endl << endl;
        return EXIT_FAILURE;
    }
    Wk = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wk.txt", Wk, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // Wv: [EMBED_DIM][EMBED_DIM]  int8  ← Wv.txt
    cout << "HOST-Info: Allocating and loading Wv from Wv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Wv array" << endl << endl;
        return EXIT_FAILURE;
    }
    Wv = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wv.txt", Wv, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // bq: [EMBED_DIM]  float  ← bq.txt
    cout << "HOST-Info: Allocating and loading bq from bq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for bq array" << endl << endl;
        return EXIT_FAILURE;
    }
    bq = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bq.txt", bq, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // bk: [EMBED_DIM]  float  ← bk.txt
    cout << "HOST-Info: Allocating and loading bk from bk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for bk array" << endl << endl;
        return EXIT_FAILURE;
    }
    bk = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bk.txt", bk, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // bv: [EMBED_DIM]  float  ← bv.txt
    cout << "HOST-Info: Allocating and loading bv from bv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for bv array" << endl << endl;
        return EXIT_FAILURE;
    }
    bv = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bv.txt", bv, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // scale_input: [SEQ_LEN]  float  ← scale_input.txt（per-token scale，來自前一層 LayerNorm）
    cout << "HOST-Info: Allocating and loading scale_input from scale_input.txt ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_input array" << endl << endl;
        return EXIT_FAILURE;
    }
    scale_input = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_input.txt", scale_input, SEQ_LEN)) return EXIT_FAILURE;
    cout << "Done (" << SEQ_LEN << " elements)" << endl;

    // scale_wq: [EMBED_DIM]  float  ← scale_wq.txt（per-output-channel weight scale）
    cout << "HOST-Info: Allocating and loading scale_wq from scale_wq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_wq array" << endl << endl;
        return EXIT_FAILURE;
    }
    scale_wq = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wq.txt", scale_wq, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // scale_wk: [EMBED_DIM]  float  ← scale_wk.txt
    cout << "HOST-Info: Allocating and loading scale_wk from scale_wk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_wk array" << endl << endl;
        return EXIT_FAILURE;
    }
    scale_wk = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wk.txt", scale_wk, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // scale_wv: [EMBED_DIM]  float  ← scale_wv.txt
    cout << "HOST-Info: Allocating and loading scale_wv from scale_wv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_wv array" << endl << endl;
        return EXIT_FAILURE;
    }
    scale_wv = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wv.txt", scale_wv, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // Q_out: [NUM_HEADS][SEQ_LEN][HEAD_DIM]  float
    cout << "HOST-Info: Allocating memory for Q_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for Q_out array" << endl << endl;
        return EXIT_FAILURE;
    }
    Q_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    // K_out: [NUM_HEADS][SEQ_LEN][HEAD_DIM]  float
    cout << "HOST-Info: Allocating memory for K_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for K_out array" << endl << endl;
        return EXIT_FAILURE;
    }
    K_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    // V_out: [NUM_HEADS][SEQ_LEN][HEAD_DIM]  float
    cout << "HOST-Info: Allocating memory for V_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory during memory allocation for V_out array" << endl << endl;
        return EXIT_FAILURE;
    }
    V_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    cout << endl;

    // ------------------------------------------------------------------
    // Step 4.2: Create Buffers in Global Memory to store data
    //           使用 cl_mem_ext_ptr_t + XCL_MEM_TOPOLOGY 指定 HBM bank
    // ------------------------------------------------------------------
    //   GlobMem_BUF_X           → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_scale_input → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_bq          → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_bk          → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_bv          → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_scale_wq    → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_scale_wk    → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_scale_wv    → HBM_BANK0  (gmem0, READ)
    //   GlobMem_BUF_Wq          → HBM_BANK2  (gmem1, READ)
    //   GlobMem_BUF_Q           → HBM_BANK2  (gmem1, WRITE)
    //   GlobMem_BUF_Wk          → HBM_BANK4  (gmem2, READ)
    //   GlobMem_BUF_K           → HBM_BANK4  (gmem2, WRITE)
    //   GlobMem_BUF_Wv          → HBM_BANK6  (gmem3, READ)
    //   GlobMem_BUF_V           → HBM_BANK6  (gmem3, WRITE)
    // ------------------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory (HBM) to store Input and Output Data ..." << endl;
    #endif

    // cl_mem_ext_ptr_t 欄位：flags = HBM bank index, obj = host ptr, param = 0
    cl_mem_ext_ptr_t ext_X, ext_scale_input;
    cl_mem_ext_ptr_t ext_bq, ext_bk, ext_bv;
    cl_mem_ext_ptr_t ext_scale_wq, ext_scale_wk, ext_scale_wv;
    cl_mem_ext_ptr_t ext_Wq, ext_Q;
    cl_mem_ext_ptr_t ext_Wk, ext_K;
    cl_mem_ext_ptr_t ext_Wv, ext_V;

    // gmem0 → HBM_BANK0
    ext_X.flags           = HBM_BANK0;  ext_X.obj           = X;           ext_X.param           = 0;
    ext_scale_input.flags = HBM_BANK0;  ext_scale_input.obj = scale_input; ext_scale_input.param = 0;
    ext_bq.flags          = HBM_BANK0;  ext_bq.obj          = bq;          ext_bq.param          = 0;
    ext_bk.flags          = HBM_BANK0;  ext_bk.obj          = bk;          ext_bk.param          = 0;
    ext_bv.flags          = HBM_BANK0;  ext_bv.obj          = bv;          ext_bv.param          = 0;
    ext_scale_wq.flags    = HBM_BANK0;  ext_scale_wq.obj    = scale_wq;    ext_scale_wq.param    = 0;
    ext_scale_wk.flags    = HBM_BANK0;  ext_scale_wk.obj    = scale_wk;    ext_scale_wk.param    = 0;
    ext_scale_wv.flags    = HBM_BANK0;  ext_scale_wv.obj    = scale_wv;    ext_scale_wv.param    = 0;

    // gmem1 → HBM_BANK2
    ext_Wq.flags = HBM_BANK2;  ext_Wq.obj = Wq;    ext_Wq.param = 0;
    ext_Q.flags  = HBM_BANK2;  ext_Q.obj  = Q_out; ext_Q.param  = 0;

    // gmem2 → HBM_BANK4
    ext_Wk.flags = HBM_BANK4;  ext_Wk.obj = Wk;    ext_Wk.param = 0;
    ext_K.flags  = HBM_BANK4;  ext_K.obj  = K_out; ext_K.param  = 0;

    // gmem3 → HBM_BANK6
    ext_Wv.flags = HBM_BANK6;  ext_Wv.obj = Wv;    ext_Wv.param = 0;
    ext_V.flags  = HBM_BANK6;  ext_V.obj  = V_out; ext_V.param  = 0;

    cl_mem GlobMem_BUF_X, GlobMem_BUF_scale_input;
    cl_mem GlobMem_BUF_bq, GlobMem_BUF_bk, GlobMem_BUF_bv;
    cl_mem GlobMem_BUF_scale_wq, GlobMem_BUF_scale_wk, GlobMem_BUF_scale_wv;
    cl_mem GlobMem_BUF_Wq, GlobMem_BUF_Wk, GlobMem_BUF_Wv;
    cl_mem GlobMem_BUF_Q,  GlobMem_BUF_K,  GlobMem_BUF_V;

    // --- gmem0 buffers (READ, HBM_BANK0) ---
    GlobMem_BUF_X = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        SEQ_LEN * EMBED_DIM * sizeof(int8_t), &ext_X, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_X" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_input = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        SEQ_LEN * sizeof(float), &ext_scale_input, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_scale_input" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_bq, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bq" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_bk, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bk" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_bv, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bv" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_scale_wq, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_scale_wq" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_scale_wk, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_scale_wk" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), &ext_scale_wv, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_scale_wv" << endl << endl;
        return EXIT_FAILURE;
    }

    // --- gmem1 buffers (HBM_BANK2) ---
    GlobMem_BUF_Wq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), &ext_Wq, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wq" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Q = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), &ext_Q, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Q" << endl << endl;
        return EXIT_FAILURE;
    }

    // --- gmem2 buffers (HBM_BANK4) ---
    GlobMem_BUF_Wk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), &ext_Wk, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wk" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_K = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), &ext_K, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_K" << endl << endl;
        return EXIT_FAILURE;
    }

    // --- gmem3 buffers (HBM_BANK6) ---
    GlobMem_BUF_Wv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), &ext_Wv, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wv" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_V = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), &ext_V, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_V" << endl << endl;
        return EXIT_FAILURE;
    }

    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application
    // ============================================================================
    //         o) Set Kernel Arguments
    // 				------------------------------------------------------
    // 				 Kernel	  		          Arg#   Description
    // 				------------------------------------------------------
    //  			 linear_qkv_multihead      0     GlobMem_BUF_X
    //				 linear_qkv_multihead      1     GlobMem_BUF_Wq
    //				 linear_qkv_multihead      2     GlobMem_BUF_bq
    //				 linear_qkv_multihead      3     GlobMem_BUF_Wk
    //				 linear_qkv_multihead      4     GlobMem_BUF_bk
    //				 linear_qkv_multihead      5     GlobMem_BUF_Wv
    //				 linear_qkv_multihead      6     GlobMem_BUF_bv
    //				 linear_qkv_multihead      7     GlobMem_BUF_scale_input
    //				 linear_qkv_multihead      8     GlobMem_BUF_scale_wq
    //				 linear_qkv_multihead      9     GlobMem_BUF_scale_wk
    //				 linear_qkv_multihead     10     GlobMem_BUF_scale_wv
    //				 linear_qkv_multihead     11     GlobMem_BUF_Q  (output)
    //				 linear_qkv_multihead     12     GlobMem_BUF_K  (output)
    //				 linear_qkv_multihead     13     GlobMem_BUF_V  (output)
    // 				------------------------------------------------------
    //         o) Copy Input Data from Host to Global Memory
    //         o) Submit Kernels for Execution
    //         o) Copy Results from Global Memory to Host
    // ============================================================================

    // 11 個輸入 migrate + 3 個輸出 migrate（content_undefined）= 14
    int Nb_Of_Mem_Events = 14;
    int Nb_Of_Exe_Events = 1;

    cl_event Mem_op_event[14];
    cl_event K_exe_event[1];

    #ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    // ----------------------------------------
    // Step 5.1: Set Kernel Arguments
    // ----------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Setting Kernel arguments ..." << endl;
    #endif
    errCode  = false;
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  0, sizeof(cl_mem), &GlobMem_BUF_X);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  1, sizeof(cl_mem), &GlobMem_BUF_Wq);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  2, sizeof(cl_mem), &GlobMem_BUF_bq);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  3, sizeof(cl_mem), &GlobMem_BUF_Wk);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  4, sizeof(cl_mem), &GlobMem_BUF_bk);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  5, sizeof(cl_mem), &GlobMem_BUF_Wv);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  6, sizeof(cl_mem), &GlobMem_BUF_bv);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  7, sizeof(cl_mem), &GlobMem_BUF_scale_input);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  8, sizeof(cl_mem), &GlobMem_BUF_scale_wq);
    errCode |= clSetKernelArg(K_linear_qkv_multihead,  9, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
    errCode |= clSetKernelArg(K_linear_qkv_multihead, 10, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
    errCode |= clSetKernelArg(K_linear_qkv_multihead, 11, sizeof(cl_mem), &GlobMem_BUF_Q);
    errCode |= clSetKernelArg(K_linear_qkv_multihead, 12, sizeof(cl_mem), &GlobMem_BUF_K);
    errCode |= clSetKernelArg(K_linear_qkv_multihead, 13, sizeof(cl_mem), &GlobMem_BUF_V);

    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------
    // Step 5.2: Copy Input data from Host to Global Memory
    // ------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
    #endif

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_X,           0, 0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write X to GlobMem_BUF_X"                       << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wq,          0, 0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wq to GlobMem_BUF_Wq"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bq,          0, 0, NULL, &Mem_op_event[2]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bq to GlobMem_BUF_bq"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wk,          0, 0, NULL, &Mem_op_event[3]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wk to GlobMem_BUF_Wk"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bk,          0, 0, NULL, &Mem_op_event[4]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bk to GlobMem_BUF_bk"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wv,          0, 0, NULL, &Mem_op_event[5]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wv to GlobMem_BUF_Wv"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bv,          0, 0, NULL, &Mem_op_event[6]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bv to GlobMem_BUF_bv"                     << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_input, 0, 0, NULL, &Mem_op_event[7]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_input to GlobMem_BUF_scale_input"   << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wq,    0, 0, NULL, &Mem_op_event[8]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wq to GlobMem_BUF_scale_wq"         << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wk,    0, 0, NULL, &Mem_op_event[9]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wk to GlobMem_BUF_scale_wk"         << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wv,    0, 0, NULL, &Mem_op_event[10]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wv to GlobMem_BUF_scale_wv"         << endl << endl; return EXIT_FAILURE; }

    // Migrate output buffers without content（只把 device side 準備好，不搬 host 資料）
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Q,
                  CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[11]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_Q without migrating content"   << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_K,
                  CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[12]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_K without migrating content"   << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_V,
                  CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[13]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_V without migrating content"   << endl << endl; return EXIT_FAILURE; }

    // --------------------------------------------------------
    errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to Submit BarrierWithWaitList" << endl << endl;
        return EXIT_FAILURE;
    }

    // ----------------------------------------
    // Step 5.3: Submit Kernel for Execution
    // ----------------------------------------
    cout << "HOST-Info: Submitting Kernel K_linear_qkv_multihead ..." << endl;

    errCode = clEnqueueTask(Command_Queue, K_linear_qkv_multihead, 0, NULL, &K_exe_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to submit K_linear_qkv_multihead" << endl << endl;
        return EXIT_FAILURE;
    }

    // ---------------------------------------------------------
    // Step 5.4: Submit Copy Results from Global Memory to Host
    // ---------------------------------------------------------
    #ifdef ALL_MESSAGES
    cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
    #endif

    cl_event Mem_read_event[3];

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Q,
                  CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_Q to Q" << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_K,
                  CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[1]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_K to K" << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_V,
                  CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[2]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_V to V" << endl << endl; return EXIT_FAILURE; }

    cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

    // ============================================================================
    // Step 6: Processing Output Results
    //         o) Store output results to files
    //         o) Check correctness of the output results (optional)
    // ============================================================================
    #ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    // ------------------------------------------------------
    // Step 6.1: Store output Results to files
    //   輸出格式：[NUM_HEADS][SEQ_LEN] 行，每行 HEAD_DIM 個 float（空格分隔）
    // ------------------------------------------------------
    const char* Q_File_Name = "output_Q.txt";
    const char* K_File_Name = "output_K.txt";
    const char* V_File_Name = "output_V.txt";

    auto write_output_float = [&](const char* fname, float* buf) -> bool {
        fstream f(fname, ios::out);
        if (!f.is_open()) {
            cout << endl << "HOST-Error: Failed to open the " << fname << " file for write" << endl << endl;
            return false;
        }
        // buf 記憶體佈局：[NUM_HEADS][SEQ_LEN][HEAD_DIM] flat（kernel 輸出即此順序）
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int i = 0; i < SEQ_LEN; i++) {
                for (int d = 0; d < HEAD_DIM; d++) {
                    f << buf[h * SEQ_LEN * HEAD_DIM + i * HEAD_DIM + d] << " ";
                }
                f << endl;
            }
        }
        f.close();
        return true;
    };

    cout << "HOST_Info: Store Q output results in: " << Q_File_Name << endl;
    if (!write_output_float(Q_File_Name, Q_out)) return EXIT_FAILURE;

    cout << "HOST_Info: Store K output results in: " << K_File_Name << endl;
    if (!write_output_float(K_File_Name, K_out)) return EXIT_FAILURE;

    cout << "HOST_Info: Store V output results in: " << V_File_Name << endl;
    if (!write_output_float(V_File_Name, V_out)) return EXIT_FAILURE;

    // ------------------------------------------------------
    // Step 6.2: Check correctness of the output results
    // Note: You can implement golden reference check here if needed
    // ------------------------------------------------------
    cout << "HOST-Info: Results stored successfully" << endl;

    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;

    int Nb_Of_Kernels = Nb_Of_Exe_Events;
    int Nb_Of_Memory_Transfers = Nb_Of_Mem_Events;

    string list_of_kernel_names[1] = {"linear_qkv_multihead"};
    run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

    for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
    for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
    for (int i = 0; i < 3; i++)               clReleaseEvent(Mem_read_event[i]);

    clReleaseMemObject(GlobMem_BUF_X);
    clReleaseMemObject(GlobMem_BUF_scale_input);
    clReleaseMemObject(GlobMem_BUF_bq);
    clReleaseMemObject(GlobMem_BUF_bk);
    clReleaseMemObject(GlobMem_BUF_bv);
    clReleaseMemObject(GlobMem_BUF_scale_wq);
    clReleaseMemObject(GlobMem_BUF_scale_wk);
    clReleaseMemObject(GlobMem_BUF_scale_wv);
    clReleaseMemObject(GlobMem_BUF_Wq);
    clReleaseMemObject(GlobMem_BUF_Wk);
    clReleaseMemObject(GlobMem_BUF_Wv);
    clReleaseMemObject(GlobMem_BUF_Q);
    clReleaseMemObject(GlobMem_BUF_K);
    clReleaseMemObject(GlobMem_BUF_V);

    clReleaseKernel(K_linear_qkv_multihead);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;

    free(X);
    free(Wq);  free(Wk);  free(Wv);
    free(bq);  free(bk);  free(bv);
    free(scale_input);
    free(scale_wq);  free(scale_wk);  free(scale_wv);
    free(Q_out);  free(K_out);  free(V_out);

    cout << endl << "HOST-Info: DONE" << endl << endl;

    return EXIT_SUCCESS;
}