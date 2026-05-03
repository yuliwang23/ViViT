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
#include "layernorm.h"

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

    cl_kernel K_layernorm;
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
#endif
    K_layernorm = clCreateKernel(Program, "layernorm", &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to create K_layernorm" << endl << endl; return EXIT_FAILURE; }

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
    float  (*x_in)[EMBED_DIM];
    float  *gamma_in;
    float  *beta_in;
    int8_t (*out_data)[EMBED_DIM];
    float  *scaler_out;

    // --- x ---
    cout << "HOST-Info: Allocating and loading data for x ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for x array" << endl; return EXIT_FAILURE;
    }
    x_in = reinterpret_cast<float(*)[EMBED_DIM]>(ptr);
    {
        ifstream infile("input_x.txt");
        if (!infile.is_open()) { cout << endl << "HOST-Error: Could not open input_x.txt" << endl; return EXIT_FAILURE; }
        for (int i = 0; i < SEQ_LEN; i++)
            for (int j = 0; j < EMBED_DIM; j++)
                if (!(infile >> x_in[i][j])) {
                    cout << endl << "HOST-Error: Data mismatch in input_x.txt at [" << i << "][" << j << "]" << endl;
                    return EXIT_FAILURE;
                }
    }
    cout << "Done (Loaded from input_x.txt)" << endl;

    // --- gamma ---
    cout << "HOST-Info: Allocating and loading data for gamma ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for gamma array" << endl; return EXIT_FAILURE;
    }
    gamma_in = reinterpret_cast<float*>(ptr);
    {
        ifstream infile("input_gamma.txt");
        if (!infile.is_open()) { cout << endl << "HOST-Error: Could not open input_gamma.txt" << endl; return EXIT_FAILURE; }
        for (int j = 0; j < EMBED_DIM; j++)
            if (!(infile >> gamma_in[j])) {
                cout << endl << "HOST-Error: Data mismatch in input_gamma.txt at [" << j << "]" << endl;
                return EXIT_FAILURE;
            }
    }
    cout << "Done (Loaded from input_gamma.txt)" << endl;

    // --- beta ---
    cout << "HOST-Info: Allocating and loading data for beta ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for beta array" << endl; return EXIT_FAILURE;
    }
    beta_in = reinterpret_cast<float*>(ptr);
    {
        ifstream infile("input_beta.txt");
        if (!infile.is_open()) { cout << endl << "HOST-Error: Could not open input_beta.txt" << endl; return EXIT_FAILURE; }
        for (int j = 0; j < EMBED_DIM; j++)
            if (!(infile >> beta_in[j])) {
                cout << endl << "HOST-Error: Data mismatch in input_beta.txt at [" << j << "]" << endl;
                return EXIT_FAILURE;
            }
    }
    cout << "Done (Loaded from input_beta.txt)" << endl;

    // --- out ---
    cout << "HOST-Info: Allocating memory for out ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory for out array" << endl; return EXIT_FAILURE;
    }
    out_data = reinterpret_cast<int8_t(*)[EMBED_DIM]>(ptr);
    cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " elements)" << endl;

    // --- scaler ---
    cout << "HOST-Info: Allocating memory for scaler ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for scaler array" << endl; return EXIT_FAILURE;
    }
    scaler_out = reinterpret_cast<float*>(ptr);
    cout << "Done (" << SEQ_LEN << " elements)" << endl;

    cout << endl;

    // ------------------------------------------------------------------
    // Step 4.2: Create Buffers in Global Memory
    //   移除所有 cl_mem_ext_ptr_t 與 CL_MEM_EXT_PTR_XILINX，
    //   改用標準 CL_MEM_READ_ONLY / CL_MEM_WRITE_ONLY。
    // ------------------------------------------------------------------
#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory ..." << endl;
#endif

    cl_mem GlobMem_BUF_x, GlobMem_BUF_gamma, GlobMem_BUF_beta;
    cl_mem GlobMem_BUF_out, GlobMem_BUF_scaler;

    GlobMem_BUF_x = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        SEQ_LEN * EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_x" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_gamma = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_gamma" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_beta = clCreateBuffer(Context,
        CL_MEM_READ_ONLY,
        EMBED_DIM * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_beta" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_out = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY,
        SEQ_LEN * EMBED_DIM * sizeof(int8_t), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_out" << endl << endl; return EXIT_FAILURE; }

    GlobMem_BUF_scaler = clCreateBuffer(Context,
        CL_MEM_WRITE_ONLY,
        SEQ_LEN * sizeof(float), NULL, &errCode);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scaler" << endl << endl; return EXIT_FAILURE; }

    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application
    // ============================================================================
    // 輸入 WriteBuffer 3 個（x, gamma, beta），輸出 migrate 移除，共 3 個 event
    int Nb_Of_Mem_Events = 3;
    int Nb_Of_Exe_Events = 1;

    cl_event Mem_op_event[3];
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
    errCode |= clSetKernelArg(K_layernorm, 0, sizeof(cl_mem), &GlobMem_BUF_x);
    errCode |= clSetKernelArg(K_layernorm, 1, sizeof(cl_mem), &GlobMem_BUF_gamma);
    errCode |= clSetKernelArg(K_layernorm, 2, sizeof(cl_mem), &GlobMem_BUF_beta);
    errCode |= clSetKernelArg(K_layernorm, 3, sizeof(cl_mem), &GlobMem_BUF_out);
    errCode |= clSetKernelArg(K_layernorm, 4, sizeof(cl_mem), &GlobMem_BUF_scaler);
    if (errCode != CL_SUCCESS) { cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl; return EXIT_FAILURE; }

    // Step 5.2: Copy Input data from Host to Global Memory
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
#endif
    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_x,
                  CL_FALSE, 0, SEQ_LEN * EMBED_DIM * sizeof(float), x_in,
                  0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to write x" << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_gamma,
                  CL_FALSE, 0, EMBED_DIM * sizeof(float), gamma_in,
                  0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to write gamma" << endl; return EXIT_FAILURE; }

    errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_beta,
                  CL_FALSE, 0, EMBED_DIM * sizeof(float), beta_in,
                  0, NULL, &Mem_op_event[2]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to write beta" << endl; return EXIT_FAILURE; }

    // 輸出 buffer 不需要 migrate，加 Barrier 確保所有 WriteBuffer 完成後再執行 kernel
    errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to Submit Barrier" << endl << endl; return EXIT_FAILURE; }

    // Step 5.3: Submit Kernel
    cout << "HOST-Info: Submitting Kernel K_layernorm ..." << endl;
    errCode = clEnqueueTask(Command_Queue, K_layernorm, 0, NULL, &K_exe_event[0]);
    if (errCode != CL_SUCCESS) { cout << endl << "HOST-Error: Failed to submit K_layernorm" << endl << endl; return EXIT_FAILURE; }

    // Step 5.4: Read back results
#ifdef ALL_MESSAGES
    cout << "HOST_Info: Submitting Copy Results from Global Memory to Host ..." << endl;
#endif
    cl_event Mem_read_event[2];

    errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_out,
                  CL_FALSE, 0, SEQ_LEN * EMBED_DIM * sizeof(int8_t), out_data,
                  1, &K_exe_event[0], &Mem_read_event[0]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to read back out" << endl; return EXIT_FAILURE; }

    errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_scaler,
                  CL_FALSE, 0, SEQ_LEN * sizeof(float), scaler_out,
                  1, &K_exe_event[0], &Mem_read_event[1]);
    if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to read back scaler" << endl; return EXIT_FAILURE; }

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

    char Out_File_Name[]    = "output_out.txt";
    char Scaler_File_Name[] = "output_scaler.txt";

    cout << "HOST_Info: Store layernorm out results in: " << Out_File_Name << endl;
    fstream Out_File;
    Out_File.open(Out_File_Name, ios::out);
    if (!Out_File.is_open()) {
        cout << endl << "HOST-Error: Failed to open " << Out_File_Name << " for write" << endl << endl;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < SEQ_LEN; i++)
        for (int d = 0; d < EMBED_DIM; d++)
            Out_File << static_cast<int>(out_data[i][d]) << "\n";
    Out_File.close();

    cout << "HOST_Info: Store layernorm scaler results in: " << Scaler_File_Name << endl;
    fstream Scaler_File;
    Scaler_File.open(Scaler_File_Name, ios::out);
    if (!Scaler_File.is_open()) {
        cout << endl << "HOST-Error: Failed to open " << Scaler_File_Name << " for write" << endl << endl;
        return EXIT_FAILURE;
    }
    Scaler_File << fixed << setprecision(9);
    for (int i = 0; i < SEQ_LEN; i++)
        Scaler_File << scaler_out[i] << "\n";
    Scaler_File.close();

    cout << "HOST-Info: Results stored successfully" << endl;

    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;

    string list_of_kernel_names[1] = {"layernorm"};
    run_custom_profiling(Nb_Of_Exe_Events, Nb_Of_Mem_Events, K_exe_event, Mem_op_event, list_of_kernel_names);

    // ============================================================================
    // Step 8: Release Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID);

    for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
    for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
    for (int i = 0; i < 2; i++)               clReleaseEvent(Mem_read_event[i]);

    clReleaseMemObject(GlobMem_BUF_x);
    clReleaseMemObject(GlobMem_BUF_gamma);
    clReleaseMemObject(GlobMem_BUF_beta);
    clReleaseMemObject(GlobMem_BUF_out);
    clReleaseMemObject(GlobMem_BUF_scaler);

    clReleaseKernel(K_layernorm);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;

    free(x_in);
    free(gamma_in);
    free(beta_in);
    free(out_data);
    free(scaler_out);

    cout << endl << "HOST-Info: DONE" << endl << endl;
    return EXIT_SUCCESS;
}