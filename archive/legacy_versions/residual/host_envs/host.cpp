/**********
Copyright (c) 2018, Xilinx, Inc.
...（版權聲明同原版）
**********/

/*******************************************************************************
** HOST Code for layernorm
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

using namespace std;

#include <CL/cl.h>
// 移除：#include <CL/cl_ext_xilinx.h>

#include "help_functions.h"
#include "utility.h"
#include "host.h"
#include "layer.h"
#include "residual.h"

#define ALL_MESSAGES

// 移除：HBM_BANK0 / HBM_BANK2 / HBM_BANK4 定義

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
    if (!Platform_IDs) { cout << endl << "HOST-Error: Out of Memory for Platform_IDs" << endl << endl; return EXIT_FAILURE; }

    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl; return EXIT_FAILURE; }

    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++) {
        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR size" << endl << endl; return EXIT_FAILURE; }

        platform_info = new char[size];
        if (!platform_info) { cout << endl << "HOST-Error: Out of Memory for platform_info" << endl << endl; return EXIT_FAILURE; }

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
    if (!Device_IDs) { cout << endl << "HOST-Error: Out of Memory for Device_IDs" << endl << endl; return EXIT_FAILURE; }

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get available devices" << endl << endl; return EXIT_FAILURE; }

    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++) {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME size" << endl << endl; return EXIT_FAILURE; }

        device_info = new char[size];
        if (!device_info) { cout << endl << "HOST-Error: Out of Memory for device_info" << endl << endl; return EXIT_FAILURE; }

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

    cl_kernel K_residual;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
#endif
    K_residual = clCreateKernel(Program, "residual", &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create K_residual" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 4: Prepare Data to Run Kernel
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                          " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif
 
    float (*A_in)[EMBED_DIM];
    float (*B_in)[EMBED_DIM];
    float (*C_hout)[EMBED_DIM];
    float (*C_sout)[EMBED_DIM];
 
    A_in  = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    B_in  = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    C_hout = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    C_sout = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
 
    if (!A_in || !B_in || !C_hout || !C_sout)
    {
        cout << endl << "HOST-Error: Out of Memory during residual block allocation" << endl << endl;
        return EXIT_FAILURE;
    }
 
    // [FIX] 移除未使用的 void *ptr = nullptr（原始碼宣告但從未使用，保留會造成 compiler warning）
 
    // [TODO] Random init A_in, B_in
    srand(42);
    for (int i = 0; i < FRAME * SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++) {
            A_in[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;  // 均勻分布 [-1, 1]
            B_in[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
 
    // [TODO] Zero init C_hout, C_sout
    memset(C_hout, 0, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float));
    memset(C_sout, 0, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float));
 
    cout << "Done init. A_in, B_in" << endl;
 
    // Compute golden output: residual = A + B
    cout << "Compute golden output\n";
    for (int i = 0; i < FRAME * SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            C_sout[i][j] = A_in[i][j] + B_in[i][j];
 
    cout << endl;
 
    // ------------------------------------------------------------------
    // Step 4.2: Create Buffers in Global Memory
    // ------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory ..." << endl;
#endif
 
    cl_mem GlobMem_BUF_A, GlobMem_BUF_B, GlobMem_BUF_Ch;
 
    GlobMem_BUF_A = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), A_in, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_A" << endl << endl;
        return EXIT_FAILURE;
    }
 
    GlobMem_BUF_B = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), B_in, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_B" << endl << endl;
        return EXIT_FAILURE;
    }
 
    GlobMem_BUF_Ch = clCreateBuffer(Context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), C_hout, &errCode);
    if (errCode != CL_SUCCESS)
    {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Ch" << endl << endl;
        return EXIT_FAILURE;
    }
 
    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application
    // ============================================================================
    // [FIX] 實際事件：A migrate, B migrate, output migrate back，共 3 個 Mem event；1 個 Exe event
    int Nb_Of_Mem_Events = 3;
    int Nb_Of_Exe_Events = 1;
 
    cl_event Mem_op_event[3];
    cl_event K_exe_event[1];
    memset(Mem_op_event, 0, sizeof(Mem_op_event));
    memset(K_exe_event,  0, sizeof(K_exe_event));
 
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
 
    // [FIX] errCode 應為 cl_int，不可用 bool 做 OR 累計（bool OR 會將非零 errCode 截斷成 1，
    //        遮蔽真正的 OpenCL 錯誤碼）
    cl_int setArgErr = CL_SUCCESS;
    setArgErr |= clSetKernelArg(K_residual, 0, sizeof(cl_mem), &GlobMem_BUF_A);
    setArgErr |= clSetKernelArg(K_residual, 1, sizeof(cl_mem), &GlobMem_BUF_B);
    setArgErr |= clSetKernelArg(K_residual, 2, sizeof(cl_mem), &GlobMem_BUF_Ch);
    if (setArgErr != CL_SUCCESS)
    {
        cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
        return EXIT_FAILURE;
    }
 
    // Step 5.2: Migrate input buffers A, B to device
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
#endif
 
    // [FIX] clEnqueueMigrateMemObjects 第二個參數是 num_mem_objects（要傳入 1 個 buffer）
    //        第三個參數必須是 const cl_mem*（指標），不能直接傳 cl_mem 值
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_A, 0, 0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate A to device" << endl; return EXIT_FAILURE; }
 
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_B, 0, 0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate B to device" << endl; return EXIT_FAILURE; }
 
    // Step 5.3: Launch residual kernel（等待 A、B migrate 完成後才執行）
    cl_event residual_wait_list[] = { Mem_op_event[0], Mem_op_event[1] };
 
    // [FIX] clEnqueueTask 第一個參數是 kernel（cl_kernel），不能用陣列下標 K_residual[0]
    //        若 K_residual 本身就是 cl_kernel，直接傳入即可
    errCode = clEnqueueTask(Command_Queue, K_residual, 2, residual_wait_list, &K_exe_event[0]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to submit residual task" << endl; return EXIT_FAILURE; }
 
    // Step 5.4: Migrate output buffer back to host（等待 kernel 完成後才搬移）
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Submitting Copy Results from Global Memory to Host ..." << endl;
#endif
 
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Ch,
                                         CL_MIGRATE_MEM_OBJECT_HOST,
                                         1, &K_exe_event[0], &Mem_op_event[2]);
    if (errCode != CL_SUCCESS)
    {
        cout << endl << "Host-Error: Failed to migrate residual output back to host" << endl << endl;
        return EXIT_FAILURE;
    }
 
    cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);
 
    // ============================================================================
    // Step 6: Store Output Results and Verify
    // ============================================================================
#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif
 
    // [FIX] 原始碼寫入的是 out_data / scaler_out（layernorm 殘留變數），
    //        修正為 C_hout（hw 輸出）與 C_sout（golden 輸出），型別皆為 float
    const char Out_File_Name[]    = "output_hw.txt";
    const char Golden_File_Name[] = "output_golden.txt";
 
    // --- 寫出 hw 輸出 ---
    cout << "HOST_Info: Store residual HW output in: " << Out_File_Name << endl;
    {
        fstream Out_File(Out_File_Name, ios::out);
        if (!Out_File.is_open())
        {
            cout << endl << "HOST-Error: Failed to open " << Out_File_Name << " for write" << endl << endl;
            return EXIT_FAILURE;
        }
        Out_File << fixed << setprecision(9);
        for (int i = 0; i < FRAME * SEQ_LEN; i++)
            for (int d = 0; d < EMBED_DIM; d++)
                Out_File << C_hout[i][d] << "\n";
    }
 
    // --- 寫出 golden 輸出 ---
    cout << "HOST_Info: Store residual golden output in: " << Golden_File_Name << endl;
    {
        fstream Golden_File(Golden_File_Name, ios::out);
        if (!Golden_File.is_open())
        {
            cout << endl << "HOST-Error: Failed to open " << Golden_File_Name << " for write" << endl << endl;
            return EXIT_FAILURE;
        }
        Golden_File << fixed << setprecision(9);
        for (int i = 0; i < FRAME * SEQ_LEN; i++)
            for (int d = 0; d < EMBED_DIM; d++)
                Golden_File << C_sout[i][d] << "\n";
    }
 
    // [TODO] Compare HW output vs golden: 計算最大誤差、平均誤差，並列出差異最大的前 10 筆
    {
        float max_diff  = 0.0f;
        float sum_diff  = 0.0f;
        int   total     = FRAME * SEQ_LEN * EMBED_DIM;
 
        // 紀錄前 10 大誤差的 (差值, flat index)
        const int TOP_K = 10;
        vector<pair<float, int>> top_diffs;
        top_diffs.reserve(total);
 
        for (int i = 0; i < FRAME * SEQ_LEN; i++)
        {
            for (int d = 0; d < EMBED_DIM; d++)
            {
                float diff = fabs(C_hout[i][d] - C_sout[i][d]);
                sum_diff += diff;
                if (diff > max_diff) max_diff = diff;
                top_diffs.push_back({ diff, i * EMBED_DIM + d });
            }
        }
 
        // 取差值最大的前 TOP_K 筆（partial sort，複雜度 O(N log K)）
        partial_sort(top_diffs.begin(),
                     top_diffs.begin() + min(TOP_K, (int)top_diffs.size()),
                     top_diffs.end(),
                     [](const pair<float,int>& a, const pair<float,int>& b){ return a.first > b.first; });
 
        float avg_diff = (total > 0) ? (sum_diff / total) : 0.0f;
 
        cout << fixed << setprecision(9);
        cout << "\n=== Verification Result ===" << endl;
        cout << "  Max  diff : " << max_diff << endl;
        cout << "  Avg  diff : " << avg_diff << endl;
        cout << "  Top " << TOP_K << " biggest diffs (diff, frame_seq_idx, embed_idx):" << endl;
 
        int show = min(TOP_K, (int)top_diffs.size());
        for (int k = 0; k < show; k++)
        {
            int flat_idx  = top_diffs[k].second;
            int seq_idx   = flat_idx / EMBED_DIM;
            int embed_idx = flat_idx % EMBED_DIM;
            cout << "    [" << k << "] diff=" << top_diffs[k].first
                 << "  seq=" << seq_idx << "  embed=" << embed_idx << endl;
        }
        cout << "===========================" << endl;
    }
 
    cout << "HOST-Info: Results stored successfully" << endl;
 
    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
 
    // [FIX] kernel 名稱改為 "residual"（原本寫的是 "layernorm"）
    string list_of_kernel_names[1] = { "residual" };
    run_custom_profiling(Nb_Of_Exe_Events, Nb_Of_Mem_Events, K_exe_event, Mem_op_event, list_of_kernel_names);
 
    // ============================================================================
    // Step 8: Release Resources
    // ============================================================================
    // [FIX] 原始碼釋放的是 layernorm 相關的 cl_mem 與變數（GlobMem_BUF_x, gamma, beta, out, scaler）
    //        修正為實際在本測試中分配的 A, B, Ch buffer 與對應 host 指標
 
    clReleaseDevice(Target_Device_ID);
 
    for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
    for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
 
    // [FIX] 移除對 Mem_read_event 的釋放（此 scope 中從未宣告 Mem_read_event）
 
    clReleaseMemObject(GlobMem_BUF_A);
    clReleaseMemObject(GlobMem_BUF_B);
    clReleaseMemObject(GlobMem_BUF_Ch);
 
    clReleaseKernel(K_residual);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);
 
    delete[] Platform_IDs;
    delete[] Device_IDs;
 
    // [FIX] 原始碼用 free() 釋放，但分配時使用自訂的 host_aligned_alloc
    //        應改用對應的釋放函式（假設為 free 或平台對應的 aligned free）
    free(A_in);
    free(B_in);
    free(C_hout);
    free(C_sout);
 
    cout << endl << "HOST-Info: DONE" << endl << endl;
    return EXIT_SUCCESS;
}