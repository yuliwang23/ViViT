/**********
Copyright (c) 2018, Xilinx, Inc.
...（版權聲明同原版）
**********/

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
// 移除：#include <CL/cl_ext_xilinx.h>

#include "help_functions.h"
#include "linear_qkv_multihead_quan.h"

#define ALL_MESSAGES

// 移除：HBM_BANK0 / HBM_BANK2 / HBM_BANK4 / HBM_BANK6 定義

// ============================================================
// Helper: 從文字檔讀 int8 矩陣
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
//                          M A I N    F U N C T I O N                                //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
    cout << endl;

    // ============================================================================
    // Step 1: Check Command Line Arguments
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                         " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    if (argc != 4) {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File>" << endl << endl;
        return EXIT_FAILURE;
    }

    const char* Target_Platform_Vendor = argv[1];
    const char* Target_Device_Name     = argv[2];
    const char* xclbinFilename         = argv[3];
    cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
    cout << "HOST-Info: Device_Name       : " << Target_Device_Name     << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename         << endl;

    // ============================================================================
    // Step 2: Detect Platform / Device, Create Context and Command Queue
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

    // Step 2.1: Get All Platforms
    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
        cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
        return EXIT_FAILURE;
    }
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
#endif

    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl;
        return EXIT_FAILURE;
    }

    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++) {
        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR size" << endl << endl; return EXIT_FAILURE; }

        platform_info = new char[size];
        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR info" << endl << endl; return EXIT_FAILURE; }

        if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
            Platform_Detected  = true;
            Target_Platform_ID = Platform_IDs[ui];
#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform : " << Target_Platform_Vendor << endl << endl;
#endif
        }
        delete[] platform_info;
    }
    if (!Platform_Detected) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Platform_Vendor << " platform" << endl << endl;
        return EXIT_FAILURE;
    }

    // Step 2.2: Get All Devices
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get number of devices" << endl << endl; return EXIT_FAILURE; }
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Number of available devices : " << Nb_Of_Devices << endl;
#endif

    Device_IDs = new cl_device_id[Nb_Of_Devices];
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get available devices" << endl << endl; return EXIT_FAILURE; }

    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++) {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME size" << endl << endl; return EXIT_FAILURE; }

        device_info = new char[size];
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME info" << endl << endl; return EXIT_FAILURE; }

        if (strcmp(device_info, Target_Device_Name) == 0) {
            Device_Detected  = true;
            Target_Device_ID = Device_IDs[ui];
        }
        delete[] device_info;
    }
    if (!Device_Detected) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Device_Name << " device" << endl << endl;
        return EXIT_FAILURE;
    }
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Selected device : " << Target_Device_Name << endl << endl;
#endif

    // Step 2.3: Create Context
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Context ..." << endl;
#endif
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create a Context" << endl << endl; return EXIT_FAILURE; }

    // Step 2.4: Create Command Queue
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Command Queue ..." << endl;
#endif
    Command_Queue = clCreateCommandQueue(Context, Target_Device_ID,
                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 3: Create Program and Kernel
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

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

    size_t     Program_Length_in_Bytes;
    cl_program Program;
    cl_int     Binary_Status;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Program with Binary ..." << endl;
#endif
    Program_Length_in_Bytes = program_length;
    Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
                                        (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl; return EXIT_FAILURE; }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Building the Program ..." << endl;
#endif
    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl; return EXIT_FAILURE; }

    cl_kernel K_linear_qkv_multihead;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: linear_qkv_multihead ..." << endl;
#endif
    K_linear_qkv_multihead = clCreateKernel(Program, "linear_qkv_multihead", &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create K_linear_qkv_multihead" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 4: Prepare Data to Run Kernel
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                          " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    void   *ptr = nullptr;
    int8_t *X;
    int8_t *Wq, *Wk, *Wv;
    float  *bq,  *bk,  *bv;
    float  *scale_input;
    float  *scale_wq, *scale_wk, *scale_wv;
    float  *Q_out, *K_out, *V_out;

    // --- X ---
    cout << "HOST-Info: Allocating and loading X from X.txt ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int8_t))) { cout << endl << "HOST-Error: OOM for X" << endl << endl; return EXIT_FAILURE; }
    X = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("X.txt", X, SEQ_LEN, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " elements)" << endl;

    // --- Wq ---
    cout << "HOST-Info: Allocating and loading Wq from Wq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) { cout << endl << "HOST-Error: OOM for Wq" << endl << endl; return EXIT_FAILURE; }
    Wq = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wq.txt", Wq, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // --- Wk ---
    cout << "HOST-Info: Allocating and loading Wk from Wk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) { cout << endl << "HOST-Error: OOM for Wk" << endl << endl; return EXIT_FAILURE; }
    Wk = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wk.txt", Wk, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // --- Wv ---
    cout << "HOST-Info: Allocating and loading Wv from Wv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) { cout << endl << "HOST-Error: OOM for Wv" << endl << endl; return EXIT_FAILURE; }
    Wv = reinterpret_cast<int8_t*>(ptr);
    if (!load_int8_matrix("Wv.txt", Wv, EMBED_DIM, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

    // --- bq ---
    cout << "HOST-Info: Allocating and loading bq from bq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for bq" << endl << endl; return EXIT_FAILURE; }
    bq = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bq.txt", bq, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- bk ---
    cout << "HOST-Info: Allocating and loading bk from bk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for bk" << endl << endl; return EXIT_FAILURE; }
    bk = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bk.txt", bk, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- bv ---
    cout << "HOST-Info: Allocating and loading bv from bv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for bv" << endl << endl; return EXIT_FAILURE; }
    bv = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("bv.txt", bv, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- scale_input ---
    cout << "HOST-Info: Allocating and loading scale_input from scale_input.txt ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * sizeof(float))) { cout << endl << "HOST-Error: OOM for scale_input" << endl << endl; return EXIT_FAILURE; }
    scale_input = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_input.txt", scale_input, SEQ_LEN)) return EXIT_FAILURE;
    cout << "Done (" << SEQ_LEN << " elements)" << endl;

    // --- scale_wq ---
    cout << "HOST-Info: Allocating and loading scale_wq from scale_wq.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for scale_wq" << endl << endl; return EXIT_FAILURE; }
    scale_wq = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wq.txt", scale_wq, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- scale_wk ---
    cout << "HOST-Info: Allocating and loading scale_wk from scale_wk.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for scale_wk" << endl << endl; return EXIT_FAILURE; }
    scale_wk = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wk.txt", scale_wk, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- scale_wv ---
    cout << "HOST-Info: Allocating and loading scale_wv from scale_wv.txt ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for scale_wv" << endl << endl; return EXIT_FAILURE; }
    scale_wv = reinterpret_cast<float*>(ptr);
    if (!load_float_vector("scale_wv.txt", scale_wv, EMBED_DIM)) return EXIT_FAILURE;
    cout << "Done (" << EMBED_DIM << " elements)" << endl;

    // --- Outputs ---
    cout << "HOST-Info: Allocating memory for Q_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for Q_out" << endl << endl; return EXIT_FAILURE; }
    Q_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    cout << "HOST-Info: Allocating memory for K_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for K_out" << endl << endl; return EXIT_FAILURE; }
    K_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    cout << "HOST-Info: Allocating memory for V_out ... ";
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) { cout << endl << "HOST-Error: OOM for V_out" << endl << endl; return EXIT_FAILURE; }
    V_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    cout << endl;

    // ------------------------------------------------------------------
    // Step 4.2: Create Buffers in Global Memory
    //   移除所有 cl_mem_ext_ptr_t 與 CL_MEM_EXT_PTR_XILINX，
    //   改用標準 CL_MEM_READ_ONLY / CL_MEM_WRITE_ONLY，
    //   由 runtime 自動決定記憶體位置。
    // ------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory ..." << endl;
#endif

    cl_mem GlobMem_BUF_X, GlobMem_BUF_scale_input;
    cl_mem GlobMem_BUF_bq, GlobMem_BUF_bk, GlobMem_BUF_bv;
    cl_mem GlobMem_BUF_scale_wq, GlobMem_BUF_scale_wk, GlobMem_BUF_scale_wv;
    cl_mem GlobMem_BUF_Wq, GlobMem_BUF_Wk, GlobMem_BUF_Wv;
    cl_mem GlobMem_BUF_Q,  GlobMem_BUF_K,  GlobMem_BUF_V;

    GlobMem_BUF_X = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        SEQ_LEN * EMBED_DIM * sizeof(int8_t), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_X" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_scale_input = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        SEQ_LEN * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scale_input" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_bq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_bq" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_bk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_bk" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_bv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_bv" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_scale_wq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scale_wq" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_scale_wk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scale_wk" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_scale_wv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scale_wv" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_Wq = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Wq" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_Wk = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Wk" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_Wv = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Wv" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_Q = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Q" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_K = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_K" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_V = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_V" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application
    // ============================================================================
    // 輸入寫入 event 11 個，輸出 migrate 拿掉（無 USE_HOST_PTR 不需要），
    // 改為只用 WriteBuffer，event 數量調整為 11
    int Nb_Of_Mem_Events = 11;
    int Nb_Of_Exe_Events = 1;

    cl_event Mem_op_event[11];
    cl_event K_exe_event[1];

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    // Step 5.1: Set Kernel Arguments
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
    if (errCode != CL_SUCCESS) { cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl; return EXIT_FAILURE; }

    // Step 5.2: Copy Input data from Host to Global Memory (WriteBuffer)
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
#endif
    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_X,           CL_FALSE, 0, SEQ_LEN * EMBED_DIM * sizeof(int8_t),    X,           0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write X"           << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wq,          CL_FALSE, 0, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wq,           0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wq"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bq,          CL_FALSE, 0, EMBED_DIM * sizeof(float),              bq,           0, NULL, &Mem_op_event[2]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bq"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wk,          CL_FALSE, 0, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wk,           0, NULL, &Mem_op_event[3]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wk"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bk,          CL_FALSE, 0, EMBED_DIM * sizeof(float),              bk,           0, NULL, &Mem_op_event[4]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bk"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wv,          CL_FALSE, 0, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wv,           0, NULL, &Mem_op_event[5]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write Wv"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bv,          CL_FALSE, 0, EMBED_DIM * sizeof(float),              bv,           0, NULL, &Mem_op_event[6]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write bv"          << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_scale_input, CL_FALSE, 0, SEQ_LEN * sizeof(float),               scale_input,  0, NULL, &Mem_op_event[7]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_input" << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_scale_wq,    CL_FALSE, 0, EMBED_DIM * sizeof(float),              scale_wq,     0, NULL, &Mem_op_event[8]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wq"    << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_scale_wk,    CL_FALSE, 0, EMBED_DIM * sizeof(float),              scale_wk,     0, NULL, &Mem_op_event[9]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wk"    << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_scale_wv,    CL_FALSE, 0, EMBED_DIM * sizeof(float),              scale_wv,     0, NULL, &Mem_op_event[10]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to write scale_wv"    << endl << endl; return EXIT_FAILURE; }

    // 輸出 buffer 不需要 migrate，WriteBuffer 後加 Barrier 即可
    errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to Submit Barrier" << endl << endl; return EXIT_FAILURE; }

    // Step 5.3: Submit Kernel
    cout << "HOST-Info: Submitting Kernel K_linear_qkv_multihead ..." << endl;
    errCode = clEnqueueTask(Command_Queue, K_linear_qkv_multihead, 0, NULL, &K_exe_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to submit K_linear_qkv_multihead" << endl << endl; return EXIT_FAILURE; }

    // Step 5.4: Read back results
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Submitting Copy Results from Global Memory to Host ..." << endl;
#endif
    cl_event Mem_read_event[3];

    errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_Q,
                  CL_FALSE, 0, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), Q_out,
                  1, &K_exe_event[0], &Mem_read_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to read Q" << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_K,
                  CL_FALSE, 0, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), K_out,
                  1, &K_exe_event[0], &Mem_read_event[1]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to read K" << endl << endl; return EXIT_FAILURE; }

    errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_V,
                  CL_FALSE, 0, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), V_out,
                  1, &K_exe_event[0], &Mem_read_event[2]);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to read V" << endl << endl; return EXIT_FAILURE; }

    cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

    // ============================================================================
    // Step 6: Store Output Results
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    const char* Q_File_Name = "output_Q.txt";
    const char* K_File_Name = "output_K.txt";
    const char* V_File_Name = "output_V.txt";

    auto write_output_float = [&](const char* fname, float* buf) -> bool {
        fstream f(fname, ios::out);
        if (!f.is_open()) {
            cout << endl << "HOST-Error: Failed to open " << fname << " for write" << endl << endl;
            return false;
        }
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

    cout << "HOST-Info: Results stored successfully" << endl;

    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;

    string list_of_kernel_names[1] = {"linear_qkv_multihead"};
    run_custom_profiling(Nb_Of_Exe_Events, Nb_Of_Mem_Events, K_exe_event, Mem_op_event, list_of_kernel_names);

    // ============================================================================
    // Step 8: Release Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID);

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