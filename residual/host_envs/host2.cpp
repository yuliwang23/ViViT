/**********
Copyright (c) 2018, Xilinx, Inc.
...（版權聲明同原版）
**********/

/*******************************************************************************
** HOST Code for residual
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
#include <bits/stdc++.h>

using namespace std;

#include <CL/cl.h>

#include "help_functions.h"
#include "utility.h"
#include "host.h"
#include "layer.h"

#define ALL_MESSAGES

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
    cout << "HOST-Info: Creating a Kernel: residual ..." << endl;
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

    A_in   = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    B_in   = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    C_hout = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));
    C_sout = reinterpret_cast<float (*)[EMBED_DIM]>(host_aligned_alloc(FRAME * SEQ_LEN * EMBED_DIM * sizeof(float)));

    if (!A_in || !B_in || !C_hout || !C_sout) {
        cout << endl << "HOST-Error: Out of Memory during residual block allocation" << endl << endl;
        return EXIT_FAILURE;
    }

    srand(42);
    for (int i = 0; i < FRAME * SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++) {
            A_in[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
            B_in[i][j] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

    memset(C_hout, 0, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float));
    memset(C_sout, 0, FRAME * SEQ_LEN * EMBED_DIM * sizeof(float));

    cout << "Done init. A_in, B_in" << endl;

    cout << "Compute golden output\n";
    for (int i = 0; i < FRAME * SEQ_LEN; i++)
        for (int j = 0; j < EMBED_DIM; j++)
            C_sout[i][j] = A_in[i][j] + B_in[i][j];

    cout << endl;

    // -------------------------------------------------------------------------
    // Step 4.2: Create parent buffers in Global Memory (full FRAME * SEQ_LEN)
    // -------------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating parent buffers in Global Memory ..." << endl;
#endif

    cl_mem GlobMem_BUF_A, GlobMem_BUF_B, GlobMem_BUF_Ch;

    GlobMem_BUF_A = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), A_in, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_A" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_B = clCreateBuffer(Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), B_in, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_B" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_Ch = clCreateBuffer(Context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        FRAME * SEQ_LEN * EMBED_DIM * sizeof(float), C_hout, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Ch" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application (per-FRAME sub-buffer)
    // ============================================================================
    //   每個 FRAME 產生 3 個 mem event：migrate A、migrate B、migrate Ch back
    //   每個 FRAME 產生 1 個 exe event：kernel 執行
    const int Nb_Of_Mem_Events = FRAME * 3;
    const int Nb_Of_Exe_Events = FRAME;

    vector<cl_event> Mem_op_event(Nb_Of_Mem_Events, nullptr);
    vector<cl_event> K_exe_event(Nb_Of_Exe_Events, nullptr);

    const size_t frame_size_bytes = SEQ_LEN * EMBED_DIM * sizeof(float);

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run Application (sub-buffer per FRAME)               " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    for (int f = 0; f < FRAME; f++)
    {
#ifdef ALL_MESSAGES
        cout << "HOST-Info: Processing FRAME " << f << " ..." << endl;
#endif
        // ---------------------------------------------------------------------
        // 5.1: 建立本 FRAME 的 sub-buffer（origin = f * frame_size_bytes）
        // ---------------------------------------------------------------------
        cl_buffer_region region;
        region.origin = static_cast<size_t>(f) * frame_size_bytes;
        region.size   = frame_size_bytes;

        cl_mem SubBuf_A = clCreateSubBuffer(GlobMem_BUF_A,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);
        if (errCode != CL_SUCCESS) {
            cout << endl << "Host-Error: Failed to create SubBuf_A for frame " << f << endl << endl;
            return EXIT_FAILURE;
        }

        cl_mem SubBuf_B = clCreateSubBuffer(GlobMem_BUF_B,
            CL_MEM_READ_ONLY,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);
        if (errCode != CL_SUCCESS) {
            cout << endl << "Host-Error: Failed to create SubBuf_B for frame " << f << endl << endl;
            return EXIT_FAILURE;
        }

        cl_mem SubBuf_Ch = clCreateSubBuffer(GlobMem_BUF_Ch,
            CL_MEM_READ_WRITE,
            CL_BUFFER_CREATE_TYPE_REGION, &region, &errCode);
        if (errCode != CL_SUCCESS) {
            cout << endl << "Host-Error: Failed to create SubBuf_Ch for frame " << f << endl << endl;
            return EXIT_FAILURE;
        }

        // ---------------------------------------------------------------------
        // 5.2: Migrate A、B sub-buffer 到 device
        // ---------------------------------------------------------------------
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &SubBuf_A, 0,
            0, NULL, &Mem_op_event[f * 3 + 0]);
        if (errCode != CL_SUCCESS) {
            cout << "Host-Error: Failed to migrate SubBuf_A for frame " << f << endl;
            return EXIT_FAILURE;
        }

        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &SubBuf_B, 0,
            0, NULL, &Mem_op_event[f * 3 + 1]);
        if (errCode != CL_SUCCESS) {
            cout << "Host-Error: Failed to migrate SubBuf_B for frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------------------------------------------------------------
        // 5.3: 設定 kernel 參數（每 FRAME 用不同的 sub-buffer）
        // ---------------------------------------------------------------------
        cl_int setArgErr = CL_SUCCESS;
        setArgErr |= clSetKernelArg(K_residual, 0, sizeof(cl_mem), &SubBuf_A);
        setArgErr |= clSetKernelArg(K_residual, 1, sizeof(cl_mem), &SubBuf_B);
        setArgErr |= clSetKernelArg(K_residual, 2, sizeof(cl_mem), &SubBuf_Ch);
        if (setArgErr != CL_SUCCESS) {
            cout << "Host-Error: Failed to set Kernel args for frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------------------------------------------------------------
        // 5.4: 等 A、B migrate 完成後執行 kernel
        // ---------------------------------------------------------------------
        cl_event wait_migrate[2] = { Mem_op_event[f * 3 + 0], Mem_op_event[f * 3 + 1] };
        errCode = clEnqueueTask(Command_Queue, K_residual, 2, wait_migrate, &K_exe_event[f]);
        if (errCode != CL_SUCCESS) {
            cout << "Host-Error: Failed to submit residual task for frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------------------------------------------------------------
        // 5.5: kernel 完成後將 Ch sub-buffer migrate 回 host
        // ---------------------------------------------------------------------
        errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &SubBuf_Ch,
            CL_MIGRATE_MEM_OBJECT_HOST,
            1, &K_exe_event[f], &Mem_op_event[f * 3 + 2]);
        if (errCode != CL_SUCCESS) {
            cout << "Host-Error: Failed to migrate SubBuf_Ch back for frame " << f << endl;
            return EXIT_FAILURE;
        }

        // ---------------------------------------------------------------------
        // sub-buffer 已入隊，可立即釋放（parent buffer 與資料均不受影響）
        // ---------------------------------------------------------------------
        clReleaseMemObject(SubBuf_A);
        clReleaseMemObject(SubBuf_B);
        clReleaseMemObject(SubBuf_Ch);
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

    const char Out_File_Name[]    = "output_hw.txt";
    const char Golden_File_Name[] = "output_golden.txt";

    cout << "HOST_Info: Store residual HW output in: " << Out_File_Name << endl;
    {
        fstream Out_File(Out_File_Name, ios::out);
        if (!Out_File.is_open()) {
            cout << endl << "HOST-Error: Failed to open " << Out_File_Name << " for write" << endl << endl;
            return EXIT_FAILURE;
        }
        Out_File << fixed << setprecision(9);
        for (int i = 0; i < FRAME * SEQ_LEN; i++)
            for (int d = 0; d < EMBED_DIM; d++)
                Out_File << C_hout[i][d] << "\n";
    }

    cout << "HOST_Info: Store residual golden output in: " << Golden_File_Name << endl;
    {
        fstream Golden_File(Golden_File_Name, ios::out);
        if (!Golden_File.is_open()) {
            cout << endl << "HOST-Error: Failed to open " << Golden_File_Name << " for write" << endl << endl;
            return EXIT_FAILURE;
        }
        Golden_File << fixed << setprecision(9);
        for (int i = 0; i < FRAME * SEQ_LEN; i++)
            for (int d = 0; d < EMBED_DIM; d++)
                Golden_File << C_sout[i][d] << "\n";
    }

    {
        float max_diff = 0.0f;
        float sum_diff = 0.0f;
        int   total    = FRAME * SEQ_LEN * EMBED_DIM;

        const int TOP_K = 10;
        vector<pair<float, int>> top_diffs;
        top_diffs.reserve(total);

        for (int i = 0; i < FRAME * SEQ_LEN; i++) {
            for (int d = 0; d < EMBED_DIM; d++) {
                float diff = fabs(C_hout[i][d] - C_sout[i][d]);
                sum_diff += diff;
                if (diff > max_diff) max_diff = diff;
                top_diffs.push_back({ diff, i * EMBED_DIM + d });
            }
        }

        partial_sort(top_diffs.begin(),
                     top_diffs.begin() + min(TOP_K, (int)top_diffs.size()),
                     top_diffs.end(),
                     [](const pair<float,int>& a, const pair<float,int>& b){ return a.first > b.first; });

        float avg_diff = (total > 0) ? (sum_diff / total) : 0.0f;

        cout << fixed << setprecision(9);
        cout << "\n=== Verification Result ===" << endl;
        cout << "  Max  diff : " << max_diff << endl;
        cout << "  Avg  diff : " << avg_diff << endl;
        cout << "  Top " << TOP_K << " biggest diffs (diff, seq_idx, embed_idx):" << endl;

        int show = min(TOP_K, (int)top_diffs.size());
        for (int k = 0; k < show; k++) {
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

    // kernel 名稱陣列長度對應 Nb_Of_Exe_Events（FRAME 個）
    vector<string> list_of_kernel_names(FRAME, "residual");
    run_custom_profiling(Nb_Of_Exe_Events, Nb_Of_Mem_Events,
                         K_exe_event.data(), Mem_op_event.data(),
                         list_of_kernel_names.data());

    // ============================================================================
    // Step 8: Release Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID);

    for (int i = 0; i < Nb_Of_Mem_Events; i++) if (Mem_op_event[i]) clReleaseEvent(Mem_op_event[i]);
    for (int i = 0; i < Nb_Of_Exe_Events; i++) if (K_exe_event[i])  clReleaseEvent(K_exe_event[i]);

    clReleaseMemObject(GlobMem_BUF_A);
    clReleaseMemObject(GlobMem_BUF_B);
    clReleaseMemObject(GlobMem_BUF_Ch);

    clReleaseKernel(K_residual);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;

    free(A_in);
    free(B_in);
    free(C_hout);
    free(C_sout);

    cout << endl << "HOST-Info: DONE" << endl << endl;
    return EXIT_SUCCESS;
}