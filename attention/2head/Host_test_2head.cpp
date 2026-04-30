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
#include <vector>
#include <string>

using namespace std;

#include <CL/cl.h>

#include "attention.h"
#include "help_functions.h"

#define ALL_MESSAGES

namespace
{
    constexpr int TEST_HEADS = 2;
    constexpr size_t HEAD_BYTES = SEQ_LEN * HEAD_DIM * sizeof(int);

    void *host_aligned_alloc(size_t size_in_bytes)
    {
        void *ptr = nullptr;
        if (posix_memalign(&ptr, 4096, size_in_bytes))
        {
            return nullptr;
        }
        return ptr;
    }

    bool load_head_input_file(const char *filename, int (*target)[HEAD_DIM])
    {
        ifstream infile(filename);
        if (!infile.is_open())
        {
            cout << endl
                 << "HOST-Error: Could not open " << filename << endl
                 << endl;
            return false;
        }

        for (int i = 0; i < SEQ_LEN; i++)
        {
            for (int j = 0; j < HEAD_DIM; j++)
            {
                if (!(infile >> target[i][j]))
                {
                    cout << endl
                         << "HOST-Error: Data mismatch in " << filename
                         << " at [" << i << "][" << j << "]" << endl
                         << endl;
                    return false;
                }
            }
        }

        return true;
    }

    void replicate_head_input(const int (*src)[HEAD_DIM], int (*dst)[SEQ_LEN][HEAD_DIM])
    {
        for (int h = 0; h < TEST_HEADS; h++)
        {
            for (int i = 0; i < SEQ_LEN; i++)
            {
                for (int j = 0; j < HEAD_DIM; j++)
                {
                    dst[h][i][j] = src[i][j];
                }
            }
        }
    }

    bool write_output_file(const char *filename, int (*out)[SEQ_LEN][HEAD_DIM])
    {
        ofstream outfile(filename);
        if (!outfile.is_open())
        {
            cout << endl
                 << "HOST-Error: Failed to open " << filename << " for write" << endl
                 << endl;
            return false;
        }

        for (int h = 0; h < TEST_HEADS; h++)
        {
            outfile << "# head " << h << "\n";
            for (int i = 0; i < SEQ_LEN; i++)
            {
                for (int j = 0; j < HEAD_DIM; j++)
                {
                    outfile << out[h][i][j];
                    if (j + 1 < HEAD_DIM)
                        outfile << " ";
                }
                outfile << "\n";
            }
        }

        return true;
    }
} // namespace

int main(int argc, char *argv[])
{
    cout << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments                        " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
#endif

    if (argc != 4)
    {
        cout << "HOST-Error: Incorrect command line syntax " << endl;
        cout << "HOST-Info: Usage: " << argv[0]
             << " <Platform_Vendor> <Device_Name> <XCLBIN_File>" << endl
             << endl;
        return EXIT_FAILURE;
    }

    const char *target_platform_vendor = argv[1];
    const char *target_device_name = argv[2];
    const char *xclbin_filename = argv[3];
    cout << "HOST-Info: Platform_Vendor   : " << target_platform_vendor << endl;
    cout << "HOST-Info: Device_Name       : " << target_device_name << endl;
    cout << "HOST-Info: XCLBIN_file       : " << xclbin_filename << endl;

    cl_uint ui;
    cl_platform_id *platform_ids;
    cl_uint num_platforms;
    cl_platform_id target_platform_id;
    bool platform_detected;
    char *platform_info;

    cl_device_id *device_ids;
    cl_uint num_devices;
    cl_device_id target_device_id;
    bool device_detected;
    char *device_info;

    cl_context context;
    cl_command_queue command_queue;

    cl_int err_code;
    size_t size;

    err_code = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err_code != CL_SUCCESS || num_platforms <= 0)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

    platform_ids = new cl_platform_id[num_platforms];
    err_code = clGetPlatformIDs(num_platforms, platform_ids, NULL);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the available platforms" << endl
             << endl;
        return EXIT_FAILURE;
    }

    platform_detected = false;
    for (ui = 0; ui < num_platforms; ui++)
    {
        err_code = clGetPlatformInfo(platform_ids[ui], CL_PLATFORM_VENDOR, 0, NULL, &size);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;

        platform_info = new char[size];
        err_code = clGetPlatformInfo(platform_ids[ui], CL_PLATFORM_VENDOR, size, platform_info, NULL);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;

        if (strcmp(platform_info, target_platform_vendor) == 0)
        {
            platform_detected = true;
            target_platform_id = platform_ids[ui];
        }
        delete[] platform_info;
    }

    if (!platform_detected)
    {
        cout << endl
             << "HOST-Error: Failed to detect platform " << target_platform_vendor << endl
             << endl;
        return EXIT_FAILURE;
    }

    err_code = clGetDeviceIDs(target_platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get the number of available devices" << endl
             << endl;
        return EXIT_FAILURE;
    }

    device_ids = new cl_device_id[num_devices];
    err_code = clGetDeviceIDs(target_platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_ids, NULL);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to get available devices" << endl
             << endl;
        return EXIT_FAILURE;
    }

    device_detected = false;
    for (ui = 0; ui < num_devices; ui++)
    {
        err_code = clGetDeviceInfo(device_ids[ui], CL_DEVICE_NAME, 0, NULL, &size);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;

        device_info = new char[size];
        err_code = clGetDeviceInfo(device_ids[ui], CL_DEVICE_NAME, size, device_info, NULL);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;

        if (strcmp(device_info, target_device_name) == 0)
        {
            device_detected = true;
            target_device_id = device_ids[ui];
        }
        delete[] device_info;
    }

    if (!device_detected)
    {
        cout << endl
             << "HOST-Error: Failed to detect device " << target_device_name << endl
             << endl;
        return EXIT_FAILURE;
    }

    context = clCreateContext(0, 1, &target_device_id, NULL, NULL, &err_code);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Context" << endl
             << endl;
        return EXIT_FAILURE;
    }

    command_queue = clCreateCommandQueue(
        context,
        target_device_id,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE,
        &err_code);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create a Command Queue" << endl
             << endl;
        return EXIT_FAILURE;
    }

    unsigned char *xclbin_memory;
    int program_length = loadFile2Memory(xclbin_filename, (char **)&xclbin_memory);
    if (program_length < 0)
    {
        cout << endl
             << "HOST-Error: Failed to load " << xclbin_filename << endl
             << endl;
        return EXIT_FAILURE;
    }

    size_t program_length_in_bytes = program_length;
    cl_int binary_status;
    cl_program program = clCreateProgramWithBinary(
        context,
        1,
        &target_device_id,
        &program_length_in_bytes,
        (const unsigned char **)&xclbin_memory,
        &binary_status,
        &err_code);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to create program from binary" << endl
             << endl;
        return EXIT_FAILURE;
    }

    err_code = clBuildProgram(program, 1, &target_device_id, NULL, NULL, NULL);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to build program executable" << endl
             << endl;
        return EXIT_FAILURE;
    }

    cl_kernel kernels[TEST_HEADS];
    for (int h = 0; h < TEST_HEADS; h++)
    {
        kernels[h] = clCreateKernel(program, "attention_opt", &err_code);
        if (err_code != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to create attention_opt kernel " << h << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }

    int (*q_seed)[HEAD_DIM] = reinterpret_cast<int (*)[HEAD_DIM]>(host_aligned_alloc(HEAD_BYTES));
    int (*k_seed)[HEAD_DIM] = reinterpret_cast<int (*)[HEAD_DIM]>(host_aligned_alloc(HEAD_BYTES));
    int (*v_seed)[HEAD_DIM] = reinterpret_cast<int (*)[HEAD_DIM]>(host_aligned_alloc(HEAD_BYTES));

    int (*q_all)[SEQ_LEN][HEAD_DIM] = reinterpret_cast<int (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(TEST_HEADS * HEAD_BYTES));
    int (*k_all)[SEQ_LEN][HEAD_DIM] = reinterpret_cast<int (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(TEST_HEADS * HEAD_BYTES));
    int (*v_all)[SEQ_LEN][HEAD_DIM] = reinterpret_cast<int (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(TEST_HEADS * HEAD_BYTES));
    int (*out_all)[SEQ_LEN][HEAD_DIM] = reinterpret_cast<int (*)[SEQ_LEN][HEAD_DIM]>(host_aligned_alloc(TEST_HEADS * HEAD_BYTES));

    if (!q_seed || !k_seed || !v_seed || !q_all || !k_all || !v_all || !out_all)
    {
        cout << endl
             << "HOST-Error: Failed to allocate aligned host buffers" << endl
             << endl;
        return EXIT_FAILURE;
    }

    if (!load_head_input_file("input_q.txt", q_seed) ||
        !load_head_input_file("input_k.txt", k_seed) ||
        !load_head_input_file("input_v.txt", v_seed))
    {
        return EXIT_FAILURE;
    }

    replicate_head_input(q_seed, q_all);
    replicate_head_input(k_seed, k_all);
    replicate_head_input(v_seed, v_all);
    memset(out_all, 0, TEST_HEADS * HEAD_BYTES);

    cl_mem buf_q[TEST_HEADS];
    cl_mem buf_k[TEST_HEADS];
    cl_mem buf_v[TEST_HEADS];
    cl_mem buf_out[TEST_HEADS];

    for (int h = 0; h < TEST_HEADS; h++)
    {
        buf_q[h] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, HEAD_BYTES, q_all[h], &err_code);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        buf_k[h] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, HEAD_BYTES, k_all[h], &err_code);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        buf_v[h] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, HEAD_BYTES, v_all[h], &err_code);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        buf_out[h] = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, HEAD_BYTES, out_all[h], &err_code);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
    }

    const int num_mem_events = TEST_HEADS * 4 + 1;
    const int num_kernel_events = TEST_HEADS;
    cl_event mem_events[num_mem_events];
    cl_event kernel_events[num_kernel_events];

    int mem_event_idx = 0;
    for (int h = 0; h < TEST_HEADS; h++)
    {
        err_code = clSetKernelArg(kernels[h], 0, sizeof(cl_mem), &buf_q[h]);
        err_code |= clSetKernelArg(kernels[h], 1, sizeof(cl_mem), &buf_k[h]);
        err_code |= clSetKernelArg(kernels[h], 2, sizeof(cl_mem), &buf_v[h]);
        err_code |= clSetKernelArg(kernels[h], 3, sizeof(cl_mem), &buf_out[h]);
        if (err_code != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to set kernel args for head " << h << endl
                 << endl;
            return EXIT_FAILURE;
        }

        err_code = clEnqueueMigrateMemObjects(command_queue, 1, &buf_q[h], 0, 0, NULL, &mem_events[mem_event_idx++]);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        err_code = clEnqueueMigrateMemObjects(command_queue, 1, &buf_k[h], 0, 0, NULL, &mem_events[mem_event_idx++]);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        err_code = clEnqueueMigrateMemObjects(command_queue, 1, &buf_v[h], 0, 0, NULL, &mem_events[mem_event_idx++]);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
        err_code = clEnqueueMigrateMemObjects(
            command_queue,
            1,
            &buf_out[h],
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
            0,
            NULL,
            &mem_events[mem_event_idx++]);
        if (err_code != CL_SUCCESS)
            return EXIT_FAILURE;
    }

    err_code = clEnqueueBarrierWithWaitList(command_queue, 0, NULL, &mem_events[mem_event_idx++]);
    if (err_code != CL_SUCCESS)
    {
        cout << endl
             << "HOST-Error: Failed to enqueue barrier" << endl
             << endl;
        return EXIT_FAILURE;
    }

    cl_event kernel_wait_list[] = {mem_events[num_mem_events - 1]};
    for (int h = 0; h < TEST_HEADS; h++)
    {
        err_code = clEnqueueTask(command_queue, kernels[h], 1, kernel_wait_list, &kernel_events[h]);
        if (err_code != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to submit attention_opt for head " << h << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }

    vector<cl_event> readback_events(TEST_HEADS);
    for (int h = 0; h < TEST_HEADS; h++)
    {
        err_code = clEnqueueMigrateMemObjects(
            command_queue,
            1,
            &buf_out[h],
            CL_MIGRATE_MEM_OBJECT_HOST,
            1,
            &kernel_events[h],
            &readback_events[h]);
        if (err_code != CL_SUCCESS)
        {
            cout << endl
                 << "HOST-Error: Failed to migrate output for head " << h << endl
                 << endl;
            return EXIT_FAILURE;
        }
    }

    clWaitForEvents(TEST_HEADS, readback_events.data());
    clFinish(command_queue);

    if (!write_output_file("Out_output_2head.txt", out_all))
    {
        return EXIT_FAILURE;
    }

    string kernel_names[num_kernel_events] = {
        "attention_opt_head_0",
        "attention_opt_head_1"};
    run_custom_profiling(num_kernel_events, num_mem_events, kernel_events, mem_events, kernel_names);

    for (int h = 0; h < TEST_HEADS; h++)
    {
        clReleaseEvent(readback_events[h]);
        clReleaseMemObject(buf_q[h]);
        clReleaseMemObject(buf_k[h]);
        clReleaseMemObject(buf_v[h]);
        clReleaseMemObject(buf_out[h]);
        clReleaseKernel(kernels[h]);
    }

    for (int i = 0; i < num_mem_events; i++)
        clReleaseEvent(mem_events[i]);
    for (int i = 0; i < num_kernel_events; i++)
        clReleaseEvent(kernel_events[i]);

    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    clReleaseDevice(target_device_id);

    delete[] platform_ids;
    delete[] device_ids;
    free(q_seed);
    free(k_seed);
    free(v_seed);
    free(q_all);
    free(k_all);
    free(v_all);
    free(out_all);

    cout << endl
         << "HOST-Info: Wrote Out_output_2head.txt" << endl
         << "HOST-Info: DONE" << endl
         << endl;

    return EXIT_SUCCESS;
}
