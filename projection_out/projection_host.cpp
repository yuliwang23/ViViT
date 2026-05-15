/*******************************************************************************
** HOST Code for projection_out
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

using namespace std;

#include <CL/cl.h>

#include "projection.h"

#define ALL_MESSAGES

static int loadFile2Memory(const char *filename, char **result)
{
    int size = 0;
    ifstream stream(filename, ifstream::binary);
    if (!stream) {
        return -1;
    }

    stream.seekg(0, stream.end);
    size = stream.tellg();
    stream.seekg(0, stream.beg);

    *result = new char[size + 1];
    stream.read(*result, size);
    if (!stream) {
        return -2;
    }
    stream.close();
    (*result)[size] = 0;
    return size;
}

static int8_t clamp_int8_ref(int value)
{
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return static_cast<int8_t>(value);
}

static void init_projection_data(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM])
{
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int v = (h * 17 + i * 5 + d * 3) % 251;
                attention_in[h][i][d] = (static_cast<float>(v) - 125.0f) * 0.003f;
            }
        }
    }

    for (int d = 0; d < EMBED_DIM; d++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            weight[d][o] = static_cast<int8_t>(((d * 11 + o * 7) % 255) - 127);
        }
    }

    for (int o = 0; o < EMBED_DIM; o++) {
        bias[o] = static_cast<float>((o % 19) - 9) * 0.001f;
        scale_weight[o] = 0.0004f + static_cast<float>(o % 13) * 0.00001f;
    }
}

static void projection_reference(
    float attention_in[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out_ref[SEQ_LEN][EMBED_DIM])
{
    for (int token = 0; token < SEQ_LEN; token++) {
        float max_abs = 0.0f;
        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const float value = fabsf(attention_in[h][token][d]);
                if (value > max_abs) {
                    max_abs = value;
                }
            }
        }

        const float scale_input = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        int8_t q_token[EMBED_DIM];

        for (int h = 0; h < NUM_HEADS; h++) {
            for (int d = 0; d < HEAD_DIM; d++) {
                const int flat = h * HEAD_DIM + d;
                q_token[flat] = clamp_int8_ref(
                    static_cast<int>(roundf(attention_in[h][token][d] / scale_input)));
            }
        }

        for (int o = 0; o < EMBED_DIM; o++) {
            int32_t acc = 0;
            for (int d = 0; d < EMBED_DIM; d++) {
                acc += static_cast<int32_t>(q_token[d]) *
                       static_cast<int32_t>(weight[d][o]);
            }
            const float dequant =
                static_cast<float>(acc) * scale_input * scale_weight[o] + bias[o];
            out_ref[token][o] = static_cast<int>(roundf(dequant));
        }
    }
}

int main(int argc, char *argv[])
{
    cout << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                        " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    if (argc != 4) {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info:  Usage: " << argv[0]
             << " <Platform_Vendor> <Device_Name> <XCLBIN_File>" << endl << endl;
        return EXIT_FAILURE;
    }

    const char *Target_Platform_Vendor = argv[1];
    const char *Target_Device_Name = argv[2];
    const char *xclbinFilename = argv[3];
    cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
    cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;

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
    cl_platform_id Target_Platform_ID = NULL;
    bool Platform_Detected;
    char *platform_info;

    cl_device_id *Device_IDs;
    cl_uint Nb_Of_Devices;
    cl_device_id Target_Device_ID = NULL;
    bool Device_Detected;
    char *device_info;

    cl_context Context;
    cl_command_queue Command_Queue;
    cl_int errCode;
    size_t size;

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

    Platform_Detected = false;
    for (ui = 0; ui < Nb_Of_Platforms; ui++) {
        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR size" << endl << endl;
            return EXIT_FAILURE;
        }

        platform_info = new char[size];
        if (!platform_info) {
            cout << endl << "HOST-Error: Out of Memory during platform info allocation" << endl << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR" << endl << endl;
            return EXIT_FAILURE;
        }

        if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
            Platform_Detected = true;
            Target_Platform_ID = Platform_IDs[ui];
#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl << endl;
#endif
        }
        delete[] platform_info;
    }

    if (Platform_Detected == false) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Platform_Vendor << " platform" << endl << endl;
        return EXIT_FAILURE;
    }

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

    Device_Detected = false;
    for (ui = 0; ui < Nb_Of_Devices; ui++) {
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME size" << endl << endl;
            return EXIT_FAILURE;
        }

        device_info = new char[size];
        if (!device_info) {
            cout << endl << "HOST-Error: Out of Memory during device info allocation" << endl << endl;
            return EXIT_FAILURE;
        }

        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME" << endl << endl;
            return EXIT_FAILURE;
        }

        if (strcmp(device_info, Target_Device_Name) == 0) {
            Device_Detected = true;
            Target_Device_ID = Device_IDs[ui];
        }
        delete[] device_info;
    }

    if (Device_Detected == false) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Device_Name << " device" << endl << endl;
        return EXIT_FAILURE;
    } else {
#ifdef ALL_MESSAGES
        cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl << endl;
#endif
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Context ... " << endl;
#endif
    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Command Queue ... " << endl;
#endif
    Command_Queue = clCreateCommandQueue(
        Context, Target_Device_ID,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernels                          " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    unsigned char *xclbin_Memory;
    int program_length;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
#endif
    program_length = loadFile2Memory(xclbinFilename, (char **)&xclbin_Memory);
    if (program_length < 0) {
        cout << endl << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl << endl;
        return EXIT_FAILURE;
    }

    size_t Program_Length_in_Bytes;
    cl_program Program;
    cl_int Binary_Status;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating Program with Binary ..." << endl;
#endif
    Program_Length_in_Bytes = program_length;
    Program = clCreateProgramWithBinary(
        Context, 1, &Target_Device_ID, &Program_Length_in_Bytes,
        (const unsigned char **)&xclbin_Memory, &Binary_Status, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Building the Program ..." << endl;
#endif
    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
        return EXIT_FAILURE;
    }

    cl_kernel K_projection_out;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Creating a Kernel: projection_out ..." << endl;
#endif
    K_projection_out = clCreateKernel(Program, "projection_out", &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create K_projection_out" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernel                          " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    float (*Attention_In)[SEQ_LEN][HEAD_DIM];
    int8_t (*Wout)[EMBED_DIM];
    float *bout;
    float *scale_wout;
    int (*Out)[EMBED_DIM];
    int (*Out_ref)[EMBED_DIM];
    void *ptr = NULL;

    cout << "HOST-Info: Allocating projection input/output data ... " << endl;
    if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for Attention_In" << endl << endl;
        return EXIT_FAILURE;
    }
    Attention_In = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(ptr);

    if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) {
        cout << endl << "HOST-Error: Out of Memory for Wout" << endl << endl;
        return EXIT_FAILURE;
    }
    Wout = reinterpret_cast<int8_t (*)[EMBED_DIM]>(ptr);

    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for bout" << endl << endl;
        return EXIT_FAILURE;
    }
    bout = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
        cout << endl << "HOST-Error: Out of Memory for scale_wout" << endl << endl;
        return EXIT_FAILURE;
    }
    scale_wout = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for Out" << endl << endl;
        return EXIT_FAILURE;
    }
    Out = reinterpret_cast<int (*)[EMBED_DIM]>(ptr);

    if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for Out_ref" << endl << endl;
        return EXIT_FAILURE;
    }
    Out_ref = reinterpret_cast<int (*)[EMBED_DIM]>(ptr);

    init_projection_data(Attention_In, Wout, bout, scale_wout);
    memset(Out, 0, SEQ_LEN * EMBED_DIM * sizeof(int));
    memset(Out_ref, 0, SEQ_LEN * EMBED_DIM * sizeof(int));
    projection_reference(Attention_In, Wout, bout, scale_wout, Out_ref);
    cout << "HOST-Info: Projection test data initialized" << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
#endif
    cl_mem GlobMem_BUF_Attention_In, GlobMem_BUF_Wout, GlobMem_BUF_bout,
        GlobMem_BUF_scale_wout, GlobMem_BUF_Out;

    GlobMem_BUF_Attention_In = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), Attention_In, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Attention_In" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wout = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wout, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Wout" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bout = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), bout, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_bout" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wout = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), scale_wout, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_scale_wout" << endl << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Out = clCreateBuffer(
        Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        SEQ_LEN * EMBED_DIM * sizeof(int), Out, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to allocate GlobMem_BUF_Out" << endl << endl;
        return EXIT_FAILURE;
    }

    int Nb_Of_Mem_Events = 5;
    int Nb_Of_Exe_Events = 1;
    cl_event Mem_op_event[Nb_Of_Mem_Events];
    cl_event K_exe_event[Nb_Of_Exe_Events];

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 5) Run Application                                     " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

#ifdef ALL_MESSAGES
    cout << "HOST-Info: Setting Kernel arguments ..." << endl;
#endif
    errCode = false;
    errCode |= clSetKernelArg(K_projection_out, 0, sizeof(cl_mem), &GlobMem_BUF_Attention_In);
    errCode |= clSetKernelArg(K_projection_out, 1, sizeof(cl_mem), &GlobMem_BUF_Wout);
    errCode |= clSetKernelArg(K_projection_out, 2, sizeof(cl_mem), &GlobMem_BUF_bout);
    errCode |= clSetKernelArg(K_projection_out, 3, sizeof(cl_mem), &GlobMem_BUF_scale_wout);
    errCode |= clSetKernelArg(K_projection_out, 4, sizeof(cl_mem), &GlobMem_BUF_Out);

    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
#endif
    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_Attention_In, 0, 0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to write Attention_In" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_Wout, 0, 0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to write Wout" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_bout, 0, 0, NULL, &Mem_op_event[2]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to write bout" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_scale_wout, 0, 0, NULL, &Mem_op_event[3]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to write scale_wout" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        0, NULL, &Mem_op_event[4]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to migrate output buffer without content" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to Submit BarrierWithWaitList" << endl << endl;
        return EXIT_FAILURE;
    }

    cout << "HOST-Info: Submitting Kernel K_projection_out ..." << endl;
    errCode = clEnqueueTask(Command_Queue, K_projection_out, 0, NULL, &K_exe_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to submit K_projection_out" << endl << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
#endif
    cl_event Mem_read_event[1];
    errCode = clEnqueueMigrateMemObjects(
        Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_HOST,
        1, &K_exe_event[0], &Mem_read_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_Out" << endl << endl;
        return EXIT_FAILURE;
    }

    cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 6) Check the Output Results                            " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    int max_err = 0;
    int mismatch_count = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            const int diff = abs(Out[i][o] - Out_ref[i][o]);
            if (diff > max_err) {
                max_err = diff;
            }
            if (diff != 0 && mismatch_count < 10) {
                cout << "HOST-Warn: mismatch at [" << i << "][" << o << "]"
                     << " hw=" << Out[i][o] << " ref=" << Out_ref[i][o]
                     << " diff=" << diff << endl;
                mismatch_count++;
            }
        }
    }

    cout << "HOST-Info: projection_out max int error = " << max_err << endl;
    if (max_err == 0) {
        cout << "HOST-Info: PASS" << endl;
    } else {
        cout << "HOST-Info: FAIL" << endl;
    }

    char Out_File_Name[] = "projection_out_output.txt";
    cout << "HOST_Info: Store projection output results in: " << Out_File_Name << endl;
    fstream Out_File;
    Out_File.open(Out_File_Name, ios::out);
    if (!Out_File.is_open()) {
        cout << endl << "HOST-Error: Failed to open the " << Out_File_Name << " file for write" << endl << endl;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            Out_File << Out[i][o] << " ";
        }
        Out_File << endl;
    }
    Out_File.close();

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 7) Release Allocated Resources                         " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    clReleaseDevice(Target_Device_ID);
    for (int i = 0; i < Nb_Of_Mem_Events; i++) {
        clReleaseEvent(Mem_op_event[i]);
    }
    for (int i = 0; i < Nb_Of_Exe_Events; i++) {
        clReleaseEvent(K_exe_event[i]);
    }
    clReleaseEvent(Mem_read_event[0]);

    clReleaseMemObject(GlobMem_BUF_Attention_In);
    clReleaseMemObject(GlobMem_BUF_Wout);
    clReleaseMemObject(GlobMem_BUF_bout);
    clReleaseMemObject(GlobMem_BUF_scale_wout);
    clReleaseMemObject(GlobMem_BUF_Out);

    clReleaseKernel(K_projection_out);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;
    delete[] xclbin_Memory;
    free(Attention_In);
    free(Wout);
    free(bout);
    free(scale_wout);
    free(Out);
    free(Out_ref);

    cout << endl << "HOST-Info: DONE" << endl << endl;

    return (max_err == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
