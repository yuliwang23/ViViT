#include "projection.h"

#include <CL/cl.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

using namespace std;

static float (*attention_in)[SEQ_LEN][HEAD_DIM];
static int8_t (*weight)[EMBED_DIM];
static float *bias;
static float *scale_weight;
static int (*out_hw)[EMBED_DIM];
static int (*out_ref)[EMBED_DIM];

static int load_file_to_memory(const char *filename, char **result)
{
    ifstream stream(filename, ifstream::binary);
    if (!stream) {
        return -1;
    }

    stream.seekg(0, stream.end);
    int size = static_cast<int>(stream.tellg());
    stream.seekg(0, stream.beg);

    *result = new char[size];
    stream.read(*result, size);
    return stream ? size : -2;
}

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
            out_ref[token][o] = static_cast<int>(roundf(dequant));
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        cerr << "Usage: " << argv[0]
             << " <Platform_Vendor> <Device_Name> <XCLBIN_File>" << endl;
        return EXIT_FAILURE;
    }

    const char *target_vendor = argv[1];
    const char *target_device = argv[2];
    const char *xclbin_file = argv[3];

    attention_in = reinterpret_cast<float (*)[SEQ_LEN][HEAD_DIM]>(
        aligned_alloc_t<float>(NUM_HEADS * SEQ_LEN * HEAD_DIM));
    weight = reinterpret_cast<int8_t (*)[EMBED_DIM]>(
        aligned_alloc_t<int8_t>(EMBED_DIM * EMBED_DIM));
    bias = aligned_alloc_t<float>(EMBED_DIM);
    scale_weight = aligned_alloc_t<float>(EMBED_DIM);
    out_hw = reinterpret_cast<int (*)[EMBED_DIM]>(
        aligned_alloc_t<int>(SEQ_LEN * EMBED_DIM));
    out_ref = reinterpret_cast<int (*)[EMBED_DIM]>(
        aligned_alloc_t<int>(SEQ_LEN * EMBED_DIM));

    init_data();
    projection_ref();

    cl_int err = CL_SUCCESS;
    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        cerr << "HOST-Error: no OpenCL platforms found" << endl;
        return EXIT_FAILURE;
    }

    cl_platform_id *platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, nullptr);

    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    for (cl_uint p = 0; p < num_platforms && !device; p++) {
        size_t vendor_size = 0;
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, 0, nullptr, &vendor_size);
        char *vendor = new char[vendor_size];
        clGetPlatformInfo(platforms[p], CL_PLATFORM_VENDOR, vendor_size, vendor, nullptr);

        if (strstr(vendor, target_vendor) != nullptr) {
            cl_uint num_devices = 0;
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ACCELERATOR, 0, nullptr, &num_devices);
            cl_device_id *devices = new cl_device_id[num_devices];
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ACCELERATOR, num_devices, devices, nullptr);

            for (cl_uint d = 0; d < num_devices; d++) {
                size_t name_size = 0;
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, 0, nullptr, &name_size);
                char *name = new char[name_size];
                clGetDeviceInfo(devices[d], CL_DEVICE_NAME, name_size, name, nullptr);
                if (strstr(name, target_device) != nullptr) {
                    platform = platforms[p];
                    device = devices[d];
                    delete[] name;
                    break;
                }
                delete[] name;
            }
            delete[] devices;
        }
        delete[] vendor;
    }
    delete[] platforms;

    if (!platform || !device) {
        cerr << "HOST-Error: target platform/device not found" << endl;
        return EXIT_FAILURE;
    }

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    cl_command_queue queue =
        clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);

    char *binary = nullptr;
    const int binary_size = load_file_to_memory(xclbin_file, &binary);
    if (binary_size < 0) {
        cerr << "HOST-Error: failed to load xclbin" << endl;
        return EXIT_FAILURE;
    }

    const unsigned char *binary_ptr = reinterpret_cast<unsigned char *>(binary);
    size_t binary_size_t = static_cast<size_t>(binary_size);
    cl_program program =
        clCreateProgramWithBinary(context, 1, &device, &binary_size_t, &binary_ptr,
                                  nullptr, &err);
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    cl_kernel kernel = clCreateKernel(program, "projection_out", &err);

    cl_mem buf_attention = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(float), attention_in, &err);
    cl_mem buf_weight = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * EMBED_DIM * sizeof(int8_t), weight, &err);
    cl_mem buf_bias = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), bias, &err);
    cl_mem buf_scale_weight = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        EMBED_DIM * sizeof(float), scale_weight, &err);
    cl_mem buf_out = clCreateBuffer(
        context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        SEQ_LEN * EMBED_DIM * sizeof(int), out_hw, &err);

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buf_attention);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &buf_weight);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &buf_bias);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &buf_scale_weight);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &buf_out);
    if (err != CL_SUCCESS) {
        cerr << "HOST-Error: failed to set kernel args" << endl;
        return EXIT_FAILURE;
    }

    cl_mem input_bufs[] = {buf_attention, buf_weight, buf_bias, buf_scale_weight};
    err = clEnqueueMigrateMemObjects(queue, 4, input_bufs, 0, 0, nullptr, nullptr);
    err |= clEnqueueMigrateMemObjects(queue, 1, &buf_out,
                                      CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                                      0, nullptr, nullptr);
    clFinish(queue);

    cl_event kernel_event;
    err = clEnqueueTask(queue, kernel, 0, nullptr, &kernel_event);
    if (err != CL_SUCCESS) {
        cerr << "HOST-Error: failed to enqueue projection_out" << endl;
        return EXIT_FAILURE;
    }

    err = clEnqueueMigrateMemObjects(queue, 1, &buf_out, CL_MIGRATE_MEM_OBJECT_HOST,
                                     1, &kernel_event, nullptr);
    clFinish(queue);

    int max_err = 0;
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int o = 0; o < EMBED_DIM; o++) {
            int diff = std::abs(out_hw[i][o] - out_ref[i][o]);
            if (diff > max_err) {
                max_err = diff;
            }
        }
    }

    cout << "HOST-Info: projection_out max int error = " << max_err << endl;
    cout << ((max_err == 0) ? "PASS" : "FAIL") << endl;

    clReleaseMemObject(buf_attention);
    clReleaseMemObject(buf_weight);
    clReleaseMemObject(buf_bias);
    clReleaseMemObject(buf_scale_weight);
    clReleaseMemObject(buf_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    delete[] binary;

    free(attention_in);
    free(weight);
    free(bias);
    free(scale_weight);
    free(out_hw);
    free(out_ref);

    return (max_err == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
