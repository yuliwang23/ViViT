/*******************************************************************************
** HOST Code for multihead_attention (Full Transformer Attention Layer)
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

#include "help_functions.h"
#include "multihead_attention.h" // Updated Header

#define ALL_MESSAGES

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                            M A I N    F U N C T I O N                              //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
    cout << endl;

    // ============================================================================
    // Step 1: Check Command Line Arguments
    // ============================================================================
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                       " << endl;
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
    // Step 2: Detect Target Platform and Target Device. Create Context & Queue.
    // ============================================================================
    cout << endl;
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device           " << endl;
    cout << "HOST-Info:          Create Context and Command Queue                   " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    cl_uint             ui;
    cl_platform_id      *Platform_IDs;
    cl_uint             Nb_Of_Platforms;
    cl_platform_id      Target_Platform_ID;
    bool                Platform_Detected;
    char                *platform_info;

    cl_device_id        *Device_IDs;
    cl_uint             Nb_Of_Devices;
    cl_device_id        Target_Device_ID;
    bool                Device_Detected;
    char                *device_info;

    cl_context          Context;
    cl_command_queue    Command_Queue;

    cl_int              errCode;
    size_t              size;

    // ------------------------------------------------------------------------------------
    // Step 2.1: Get All PLATFORMS
    // ------------------------------------------------------------------------------------
    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) return EXIT_FAILURE;

    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
    errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);

    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++) {
        clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        platform_info = new char[size];
        clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info , NULL);

        if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
            Platform_Detected        = true;
            Target_Platform_ID       = Platform_IDs[ui];
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform             : " << Target_Platform_Vendor << endl;
            #endif
        }
        delete[] platform_info;
    }
    if (!Platform_Detected) return EXIT_FAILURE;

    // ------------------------------------------------------------------------------------
    // Step 2.2: Get All Devices
    // ------------------------------------------------------------------------------------
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    Device_IDs = new cl_device_id[Nb_Of_Devices];
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);

    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++) {
        clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        device_info = new char[size];
        clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);

        if (strcmp(device_info, Target_Device_Name) == 0) {
            Device_Detected        = true;
            Target_Device_ID       = Device_IDs[ui];
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected device               : " << Target_Device_Name << endl << endl;
            #endif
        }
        delete[] device_info;
    }
    if (!Device_Detected) return EXIT_FAILURE;

    // ------------------------------------------------------------------------------------
    // Step 2.3 & 2.4: Create Context and Command Queue
    // ------------------------------------------------------------------------------------
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);

    // ============================================================================
    // Step 3: Create Program and Kernel
    // ============================================================================
    unsigned char *xclbin_Memory;
    int program_length = loadFile2Memory(xclbinFilename, (char **) &xclbin_Memory);
    if (program_length < 0) return EXIT_FAILURE;

    size_t Program_Length_in_Bytes = program_length;
    cl_program Program;
    cl_int Binary_Status;
    
    Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes, 
                                        (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);

    // Updated Kernel Name
    cl_kernel K_multihead_attention = clCreateKernel(Program, "multihead_attention", &errCode);
    if (errCode != CL_SUCCESS) {
        cout << "HOST-Error: Failed to create K_multihead_attention" << endl;
        return EXIT_FAILURE;
    }

    // ================================================================
    // Step 4: Prepare Data to Run Kernel
    // ================================================================
    // Input Feature Maps and LayerNorm parameters (Float)
    float *X         = nullptr;
    float *gamma     = nullptr;
    float *beta      = nullptr;

    // Quantized Weights (INT8)
    int8_t *Wq       = nullptr;
    int8_t *Wk       = nullptr;
    int8_t *Wv       = nullptr;
    int8_t *Wo       = nullptr; // Added Output Projection Weight

    // Floating-Point Biases and Dequantization Scales (Float)
    float *bq        = nullptr;
    float *bk        = nullptr;
    float *bv        = nullptr;
    float *bo        = nullptr; // Added Output Projection Bias
    
    float *scale_wq  = nullptr;
    float *scale_wk  = nullptr;
    float *scale_wv  = nullptr;
    float *scale_wo  = nullptr; // Added Output Projection Scale

    // Final Output (Float - packed into uint256_dt by HLS, received as float)
    float *Out       = nullptr; // Single Output matrix for the entire layer

    #ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                          " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    #endif

    // ============================================================================
    // Step 4.1: Allocate and Load Input Arrays from Text Files
    // ============================================================================
    void *ptr = nullptr;

    // --- 1. Load Input Feature Map X [SEQ_LEN * EMBED_DIM] ---
    cout << "HOST-Info: Allocating and loading data for X ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) return EXIT_FAILURE;
    X = reinterpret_cast<float*>(ptr);
    {
        ifstream infile("input_x.txt");
        for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) infile >> X[i];
    }
    cout << "Done" << endl;

    // --- 2. Load LayerNorm Weights (gamma, beta) [EMBED_DIM] ---
    cout << "HOST-Info: Allocating and loading data for gamma & beta ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; gamma = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; beta  = reinterpret_cast<float*>(ptr);
    {
        ifstream f_gamma("gamma.txt"), f_beta("beta.txt");
        for (int i = 0; i < EMBED_DIM; i++) { f_gamma >> gamma[i]; f_beta >> beta[i]; }
    }
    cout << "Done" << endl;

    // --- 3. Load Quantized Weights (Wq, Wk, Wv, Wo) [EMBED_DIM * EMBED_DIM] ---
    cout << "HOST-Info: Allocating and loading data for Quantized Weights (Wq, Wk, Wv, Wo) ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wq = reinterpret_cast<int8_t*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wk = reinterpret_cast<int8_t*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wv = reinterpret_cast<int8_t*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wo = reinterpret_cast<int8_t*>(ptr);
    {
        ifstream f_wq("Wq_transposed.txt"), f_wk("Wk_transposed.txt"), f_wv("Wv_transposed.txt"), f_wo("Wo_transposed.txt");
        int temp_q, temp_k, temp_v, temp_o; 
        for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
            f_wq >> temp_q; f_wk >> temp_k; f_wv >> temp_v; f_wo >> temp_o;
            Wq[i] = static_cast<int8_t>(temp_q);
            Wk[i] = static_cast<int8_t>(temp_k);
            Wv[i] = static_cast<int8_t>(temp_v);
            Wo[i] = static_cast<int8_t>(temp_o);
        }
    }
    cout << "Done" << endl;

    // --- 4. Load Biases (bq, bk, bv, bo) [EMBED_DIM] ---
    cout << "HOST-Info: Allocating and loading data for Biases (bq, bk, bv, bo) ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bq = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bk = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bv = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bo = reinterpret_cast<float*>(ptr);
    {
        ifstream f_bq("bq.txt"), f_bk("bk.txt"), f_bv("bv.txt"), f_bo("bo.txt");
        for (int i = 0; i < EMBED_DIM; i++) { f_bq >> bq[i]; f_bk >> bk[i]; f_bv >> bv[i]; f_bo >> bo[i]; }
    }
    cout << "Done" << endl;

    // --- 5. Load Dequantization Scales (scale_wq, wk, wv, wo) [EMBED_DIM] ---
    cout << "HOST-Info: Allocating and loading Dequantization Scales ... ";
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wq = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wk = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wv = reinterpret_cast<float*>(ptr);
    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wo = reinterpret_cast<float*>(ptr);
    {
        ifstream f_swq("scale_wq.txt"), f_swk("scale_wk.txt"), f_swv("scale_wv.txt"), f_swo("scale_wo.txt");
        for (int i = 0; i < EMBED_DIM; i++) { f_swq >> scale_wq[i]; f_swk >> scale_wk[i]; f_swv >> scale_wv[i]; f_swo >> scale_wo[i]; }
    }
    cout << "Done" << endl;

    // --- 6. Allocate Memory for Output Out [SEQ_LEN * EMBED_DIM] ---
    cout << "HOST-Info: Allocating memory for Output ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) return EXIT_FAILURE; 
    Out = reinterpret_cast<float*>(ptr);
    cout << "Done" << endl << endl;

    // ------------------------------------------------------------------
    // Step 4.2: Create Buffers in Global Memory to store data
    // ------------------------------------------------------------------
    cl_mem  GlobMem_BUF_X, GlobMem_BUF_gamma, GlobMem_BUF_beta,
            GlobMem_BUF_Wq, GlobMem_BUF_Wk, GlobMem_BUF_Wv, GlobMem_BUF_Wo,
            GlobMem_BUF_bq, GlobMem_BUF_bk, GlobMem_BUF_bv, GlobMem_BUF_bo,
            GlobMem_BUF_scale_wq, GlobMem_BUF_scale_wk, GlobMem_BUF_scale_wv, GlobMem_BUF_scale_wo,
            GlobMem_BUF_Out;

    GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, SEQ_LEN * EMBED_DIM * sizeof(float), X, &errCode);
    GlobMem_BUF_gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), gamma, &errCode);
    GlobMem_BUF_beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), beta, &errCode);
    
    GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wq, &errCode);
    GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wk, &errCode);
    GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wv, &errCode);
    GlobMem_BUF_Wo = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wo, &errCode);

    GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bq, &errCode);
    GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bk, &errCode);
    GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bv, &errCode);
    GlobMem_BUF_bo = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), bo, &errCode);

    GlobMem_BUF_scale_wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wq, &errCode);
    GlobMem_BUF_scale_wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wk, &errCode);
    GlobMem_BUF_scale_wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wv, &errCode);
    GlobMem_BUF_scale_wo = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, EMBED_DIM * sizeof(float), scale_wo, &errCode);

    GlobMem_BUF_Out = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, SEQ_LEN * EMBED_DIM * sizeof(float), Out, &errCode);

    // ============================================================================
    // Step 5: Set Kernel Arguments and Run the Application
    // ============================================================================
    int Nb_Of_Mem_Events = 16; 
    int Nb_Of_Exe_Events = 1;

    cl_event Mem_op_event[Nb_Of_Mem_Events];
    cl_event K_exe_event[Nb_Of_Exe_Events];

    // ------------------------------------------------------------------
    // Step 5.1: Set Kernel Arguments (Sequential Order matching Top HLS)
    // ------------------------------------------------------------------
    errCode  = CL_SUCCESS;
    // Connect OpenCL memory buffers to the corresponding hardware ports
    errCode |= clSetKernelArg(K_multihead_attention, 0,  sizeof(cl_mem), &GlobMem_BUF_X);
    errCode |= clSetKernelArg(K_multihead_attention, 1,  sizeof(cl_mem), &GlobMem_BUF_gamma);
    errCode |= clSetKernelArg(K_multihead_attention, 2,  sizeof(cl_mem), &GlobMem_BUF_beta);
    errCode |= clSetKernelArg(K_multihead_attention, 3,  sizeof(cl_mem), &GlobMem_BUF_Wq);
    errCode |= clSetKernelArg(K_multihead_attention, 4,  sizeof(cl_mem), &GlobMem_BUF_Wk);
    errCode |= clSetKernelArg(K_multihead_attention, 5,  sizeof(cl_mem), &GlobMem_BUF_Wv);
    errCode |= clSetKernelArg(K_multihead_attention, 6,  sizeof(cl_mem), &GlobMem_BUF_Wo);
    errCode |= clSetKernelArg(K_multihead_attention, 7,  sizeof(cl_mem), &GlobMem_BUF_bq);
    errCode |= clSetKernelArg(K_multihead_attention, 8,  sizeof(cl_mem), &GlobMem_BUF_bk);
    errCode |= clSetKernelArg(K_multihead_attention, 9,  sizeof(cl_mem), &GlobMem_BUF_bv);
    errCode |= clSetKernelArg(K_multihead_attention, 10, sizeof(cl_mem), &GlobMem_BUF_bo);
    errCode |= clSetKernelArg(K_multihead_attention, 11, sizeof(cl_mem), &GlobMem_BUF_scale_wq);
    errCode |= clSetKernelArg(K_multihead_attention, 12, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
    errCode |= clSetKernelArg(K_multihead_attention, 13, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
    errCode |= clSetKernelArg(K_multihead_attention, 14, sizeof(cl_mem), &GlobMem_BUF_scale_wo);
    errCode |= clSetKernelArg(K_multihead_attention, 15, sizeof(cl_mem), &GlobMem_BUF_Out);

    if (errCode != CL_SUCCESS) { cout << "Host-ERROR: Failed to set Kernel arguments" << endl; return EXIT_FAILURE; }

    // ------------------------------------------------------
    // Step 5.2: Copy Input data from Host to Global Memory
    // ------------------------------------------------------
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_X, 0, 0, NULL, &Mem_op_event[0]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_gamma, 0, 0, NULL, &Mem_op_event[1]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_beta, 0, 0, NULL, &Mem_op_event[2]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wq, 0, 0, NULL, &Mem_op_event[3]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wk, 0, 0, NULL, &Mem_op_event[4]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wv, 0, 0, NULL, &Mem_op_event[5]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wo, 0, 0, NULL, &Mem_op_event[6]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bq, 0, 0, NULL, &Mem_op_event[7]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bk, 0, 0, NULL, &Mem_op_event[8]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bv, 0, 0, NULL, &Mem_op_event[9]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bo, 0, 0, NULL, &Mem_op_event[10]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wq, 0, 0, NULL, &Mem_op_event[11]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wk, 0, 0, NULL, &Mem_op_event[12]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wv, 0, 0, NULL, &Mem_op_event[13]);
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wo, 0, 0, NULL, &Mem_op_event[14]);

    // Virtual allocation for output buffers on FPGA device without copying host memory content
    errCode |= clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[15]);

    // Sync barrier
    clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);

    // ----------------------------------------
    // Step 5.3: Submit Kernel for Execution
    // ----------------------------------------
    cout << "HOST-Info: Submitting Kernel K_multihead_attention ..." << endl;
    errCode = clEnqueueTask(Command_Queue, K_multihead_attention, 0, NULL, &K_exe_event[0]);

    // ---------------------------------------------------------
    // Step 5.4: Submit Copy Results from Global Memory to Host
    // ---------------------------------------------------------
    cl_event Mem_read_event[1];
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[0]);

    cout << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

    // ============================================================================
    // Step 6: Processing Output Results
    // ============================================================================
    char Out_File_Name[] = "Out_output.txt";
    cout << "HOST_Info: Store Final output results in: " << Out_File_Name << endl;
    
    fstream Out_File;
    Out_File.open(Out_File_Name, ios::out);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < EMBED_DIM; j++) {
            Out_File << Out[i * EMBED_DIM + j] << " ";
        }
        Out_File << endl;
    }
    Out_File.close();
    cout << "HOST-Info: Results stored successfully" << endl;

    // ============================================================================
    // Step 7: Custom Profiling
    // ============================================================================
    int Nb_Of_Kernels = Nb_Of_Exe_Events;
    int Nb_Of_Memory_Transfers = Nb_Of_Mem_Events;

    string list_of_kernel_names[Nb_Of_Kernels] = {"multihead_attention"};
    run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    clReleaseDevice(Target_Device_ID);

    for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
    for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
    clReleaseEvent(Mem_read_event[0]);

    clReleaseMemObject(GlobMem_BUF_X);
    clReleaseMemObject(GlobMem_BUF_gamma);
    clReleaseMemObject(GlobMem_BUF_beta);
    clReleaseMemObject(GlobMem_BUF_Wq);
    clReleaseMemObject(GlobMem_BUF_Wk);
    clReleaseMemObject(GlobMem_BUF_Wv);
    clReleaseMemObject(GlobMem_BUF_Wo);
    clReleaseMemObject(GlobMem_BUF_bq);
    clReleaseMemObject(GlobMem_BUF_bk);
    clReleaseMemObject(GlobMem_BUF_bv);
    clReleaseMemObject(GlobMem_BUF_bo);
    clReleaseMemObject(GlobMem_BUF_scale_wq);
    clReleaseMemObject(GlobMem_BUF_scale_wk);
    clReleaseMemObject(GlobMem_BUF_scale_wv);
    clReleaseMemObject(GlobMem_BUF_scale_wo);
    clReleaseMemObject(GlobMem_BUF_Out);

    clReleaseKernel(K_multihead_attention);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;
    free(X); free(gamma); free(beta);
    free(Wq); free(Wk); free(Wv); free(Wo);
    free(bq); free(bk); free(bv); free(bo);
    free(scale_wq); free(scale_wk); free(scale_wv); free(scale_wo);
    free(Out);

    cout << endl << "HOST-Info: DONE" << endl << endl;

    return EXIT_SUCCESS;
}