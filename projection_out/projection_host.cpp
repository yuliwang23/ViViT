#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "help_functions.h"

using namespace std;

// Host-side mirror of projection_out dimensions. Keep these in sync with
// projection.h when changing the kernel configuration.
constexpr int SEQ_LEN = 4;
constexpr int EMBED_DIM = 64;
constexpr int NUM_HEADS = 2;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;

static float (*attention_in)[SEQ_LEN][HEAD_DIM];
static int8_t (*weight)[EMBED_DIM];
static float *bias;
static float *scale_weight;
static float (*out_hw)[EMBED_DIM];
static float (*out_ref)[EMBED_DIM];

template <typename T>
static T *aligned_alloc_t(size_t count)
{
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 4096, count * sizeof(T))) {
        cerr << "HOST-Error: aligned allocation failed" << endl;
        exit(EXIT_FAILURE);
    }
    memset(ptr, 0, count * sizeof(T));
    return reinterpret_cast<T *>(ptr);
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

static void init_data()
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

static void projection_ref()
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
            out_ref[token][o] = dequant;
        }
    }
}

int main(int argc, char **argv)
{
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

    attention_in = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(
        aligned_alloc_t<float>(NUM_HEADS * SEQ_LEN * HEAD_DIM));
    weight = reinterpret_cast<int8_t (*)[EMBED_DIM]>(
        aligned_alloc_t<int8_t>(EMBED_DIM * EMBED_DIM));
    bias = aligned_alloc_t<float>(EMBED_DIM);
    scale_weight = aligned_alloc_t<float>(EMBED_DIM);
    out_hw = reinterpret_cast<float (*)[EMBED_DIM]>(
        aligned_alloc_t<float>(SEQ_LEN * EMBED_DIM));
    out_ref = reinterpret_cast<float (*)[EMBED_DIM]>(
        aligned_alloc_t<float>(SEQ_LEN * EMBED_DIM));

    init_data();
    projection_ref();

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

    errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
    if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
        cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
        return EXIT_FAILURE;
    }

    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
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
        errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_PLATFORM_VENDOR info" << endl << endl;
            return EXIT_FAILURE;
        }

        if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
            Platform_Detected = true;
            Target_Platform_ID = Platform_IDs[ui];
            cout << "HOST-Info: Selected platform : " << Target_Platform_Vendor << endl;
        }
        delete[] platform_info;
    }

    if (!Platform_Detected) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Platform_Vendor << " platform" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, 0, NULL, &Nb_Of_Devices);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get number of devices" << endl << endl;
        return EXIT_FAILURE;
    }

    Device_IDs = new cl_device_id[Nb_Of_Devices];
    errCode = clGetDeviceIDs(Target_Platform_ID, CL_DEVICE_TYPE_ALL, Nb_Of_Devices, Device_IDs, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to get available devices" << endl << endl;
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
        errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (errCode != CL_SUCCESS) {
            cout << endl << "HOST-Error: Failed to get CL_DEVICE_NAME info" << endl << endl;
            return EXIT_FAILURE;
        }

        if (strcmp(device_info, Target_Device_Name) == 0) {
            Device_Detected = true;
            Target_Device_ID = Device_IDs[ui];
        }
        delete[] device_info;
    }

    delete[] Platform_IDs;
    delete[] Device_IDs;

    if (!Device_Detected) {
        cout << endl << "HOST-Error: Failed to detect " << Target_Device_Name << " device" << endl << endl;
        return EXIT_FAILURE;
    }
    cout << "HOST-Info: Selected device : " << Target_Device_Name << endl;

    Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
        return EXIT_FAILURE;
    }

    Command_Queue = clCreateCommandQueue(
        Context, Target_Device_ID,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
        return EXIT_FAILURE;
    }

    char *binary = nullptr;
    const int binary_size = loadFile2Memory(xclbinFilename, &binary);
    if (binary_size < 0) {
        cout << endl << "HOST-Error: Failed to load " << xclbinFilename << " binary file to memory" << endl << endl;
        return EXIT_FAILURE;
    }

    const unsigned char *binary_ptr = reinterpret_cast<unsigned char *>(binary);
    size_t binary_size_t = static_cast<size_t>(binary_size);
    cl_program Program =
        clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &binary_size_t, &binary_ptr,
                                  NULL, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
        return EXIT_FAILURE;
    }

    cl_kernel K_projection_out = clCreateKernel(Program, "projection_out", &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to create K_projection_out" << endl << endl;
        return EXIT_FAILURE;
    }

    cl_mem buf_attention = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), attention_in, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to allocate buf_attention" << endl << endl;
        return EXIT_FAILURE;
    }
    cl_mem buf_weight = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), weight, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to allocate buf_weight" << endl << endl;
        return EXIT_FAILURE;
    }
    cl_mem buf_bias = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), bias, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to allocate buf_bias" << endl << endl;
        return EXIT_FAILURE;
    }
    cl_mem buf_scale_weight = clCreateBuffer(
        Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), scale_weight, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to allocate buf_scale_weight" << endl << endl;
        return EXIT_FAILURE;
    }
    cl_mem buf_out = clCreateBuffer(
        Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        SEQ_LEN * EMBED_DIM * sizeof(float), out_hw, &errCode);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to allocate buf_out" << endl << endl;
        return EXIT_FAILURE;
    }

    errCode = false;
    errCode |= clSetKernelArg(K_projection_out, 0, sizeof(cl_mem), &buf_attention);
    errCode |= clSetKernelArg(K_projection_out, 1, sizeof(cl_mem), &buf_weight);
    errCode |= clSetKernelArg(K_projection_out, 2, sizeof(cl_mem), &buf_bias);
    errCode |= clSetKernelArg(K_projection_out, 3, sizeof(cl_mem), &buf_scale_weight);
    errCode |= clSetKernelArg(K_projection_out, 4, sizeof(cl_mem), &buf_out);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to set Kernel arguments" << endl << endl;
        return EXIT_FAILURE;
    }

    const int Nb_Of_Memory_Transfers = 5;
    const int Nb_Of_Kernels = 1;
    cl_event Mem_op_event[Nb_Of_Memory_Transfers];
    cl_event K_exe_event[Nb_Of_Kernels];

    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_attention, 0, 0, NULL, &Mem_op_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to write attention_in to buf_attention" << endl << endl;
        return EXIT_FAILURE;
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_weight, 0, 0, NULL, &Mem_op_event[1]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to write weight to buf_weight" << endl << endl;
        return EXIT_FAILURE;
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_bias, 0, 0, NULL, &Mem_op_event[2]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to write bias to buf_bias" << endl << endl;
        return EXIT_FAILURE;
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_scale_weight, 0, 0, NULL, &Mem_op_event[3]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to write scale_weight to buf_scale_weight" << endl << endl;
        return EXIT_FAILURE;
    }
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_out,
                                         CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                         0, NULL, &Mem_op_event[4]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to migrate buf_out without content" << endl << endl;
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

    cl_event Mem_read_event[1];
    errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &buf_out, CL_MIGRATE_MEM_OBJECT_HOST,
                                         1, &K_exe_event[0], &Mem_read_event[0]);
    if (errCode != CL_SUCCESS) {
        cout << endl << "HOST-Error: Failed to copy results from buf_out" << endl << endl;
        return EXIT_FAILURE;
    }

    cout << "HOST-Info: Waiting for application to be completed ..." << endl;
    clFinish(Command_Queue);

    float max_err = 0.0f;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            float diff = std::fabs(out_hw[i][o] - out_ref[i][o]);
            if (diff > max_err) {
                max_err = diff;
            }
        }
    }

    cout << "HOST-Info: projection_out max fp32 error = " << max_err << endl;
    cout << ((max_err <= 1.0e-5f) ? "PASS" : "FAIL") << endl;

    string list_of_kernel_names[Nb_Of_Kernels] = {"projection_out"};
    run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers,
                         K_exe_event, Mem_op_event, list_of_kernel_names);

    for (int i = 0; i < Nb_Of_Memory_Transfers; i++) {
        clReleaseEvent(Mem_op_event[i]);
    }
    for (int i = 0; i < Nb_Of_Kernels; i++) {
        clReleaseEvent(K_exe_event[i]);
    }
    clReleaseEvent(Mem_read_event[0]);
    clReleaseMemObject(buf_attention);
    clReleaseMemObject(buf_weight);
    clReleaseMemObject(buf_bias);
    clReleaseMemObject(buf_scale_weight);
    clReleaseMemObject(buf_out);
    clReleaseKernel(K_projection_out);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);
    delete[] binary;

    free(attention_in);
    free(weight);
    free(bias);
    free(scale_weight);
    free(out_hw);
    free(out_ref);

    return (max_err <= 1.0e-5f) ? EXIT_SUCCESS : EXIT_FAILURE;
}
