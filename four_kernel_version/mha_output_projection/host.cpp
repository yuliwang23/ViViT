/*******************************************************************************
** HOST Code for mha_output_projection
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <CL/cl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "help_functions.h"
#include "mha_output_projection.h"

using namespace std;

#define ALL_MESSAGES

constexpr size_t FEATURE_MAP_BYTES =
    static_cast<size_t>(SEQ_LEN) *
    static_cast<size_t>(EMBED_DIM) *
    sizeof(float);

constexpr size_t ATTENTION_MAX_BYTES =
    static_cast<size_t>(SEQ_LEN) *
    sizeof(float);

constexpr size_t WEIGHT_BYTES =
    static_cast<size_t>(EMBED_DIM) *
    static_cast<size_t>(EMBED_DIM) *
    sizeof(int8_t);

constexpr size_t PARAM_BYTES =
    static_cast<size_t>(EMBED_DIM) *
    sizeof(float);

constexpr int NB_OF_MEM_EVENTS = 7;
constexpr int NB_OF_EXE_EVENTS = 1;
constexpr int NB_OF_READ_EVENTS = 1;

static bool allocate_aligned(
    void **ptr,
    size_t bytes
) {
    *ptr = nullptr;

    return
        posix_memalign(
            ptr,
            4096,
            bytes
        ) == 0;
}

static bool load_float_file(
    const char *filename,
    float *destination,
    size_t count
) {
    ifstream input_file(filename);

    if (!input_file) {
        return false;
    }

    for (size_t i = 0; i < count; i++) {
        if (!(input_file >> destination[i])) {
            return false;
        }
    }

    return true;
}

static bool load_int8_file(
    const char *filename,
    int8_t *destination,
    size_t count
) {
    ifstream input_file(filename);

    if (!input_file) {
        return false;
    }

    int value = 0;

    for (size_t i = 0; i < count; i++) {
        if (!(input_file >> value)) {
            return false;
        }

        destination[i] =
            static_cast<int8_t>(value);
    }

    return true;
}

int main(int argc, char *argv[])
{
    cout << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    if (argc != 4)
    {
        cout << "HOST-Error: Incorrect command line syntax" << endl;
        cout << "HOST-Info:  Usage: "
             << argv[0]
             << " <Platform_Vendor> <Device_Name> <XCLBIN_File>"
             << endl
             << endl;

        return EXIT_FAILURE;
    }

    const char *Target_Platform_Vendor = argv[1];
    const char *Target_Device_Name = argv[2];
    const char *xclbinFilename = argv[3];

    cout << "HOST-Info: Platform_Vendor   : "
         << Target_Platform_Vendor << endl;

    cout << "HOST-Info: Device_Name       : "
         << Target_Device_Name << endl;

    cout << "HOST-Info: XCLBIN_file       : "
         << xclbinFilename << endl;

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device" << endl;
    cout << "HOST-Info:          Create Context and Command Queue" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    cl_uint ui = 0;

    cl_platform_id *Platform_IDs = nullptr;
    cl_uint Nb_Of_Platforms = 0;
    cl_platform_id Target_Platform_ID = nullptr;
    bool Platform_Detected = false;

    cl_device_id *Device_IDs = nullptr;
    cl_uint Nb_Of_Devices = 0;
    cl_device_id Target_Device_ID = nullptr;
    bool Device_Detected = false;

    cl_context Context = nullptr;
    cl_command_queue Command_Queue = nullptr;

    cl_int errCode = CL_SUCCESS;
    size_t size = 0;

    errCode = clGetPlatformIDs(
        0,
        nullptr,
        &Nb_Of_Platforms
    );

    if (errCode != CL_SUCCESS || Nb_Of_Platforms == 0)
    {
        cout << "HOST-Error: No OpenCL platform found" << endl;
        return EXIT_FAILURE;
    }

    Platform_IDs =
        new cl_platform_id[Nb_Of_Platforms];

    errCode = clGetPlatformIDs(
        Nb_Of_Platforms,
        Platform_IDs,
        nullptr
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: clGetPlatformIDs failed" << endl;
        return EXIT_FAILURE;
    }

    for (ui = 0; ui < Nb_Of_Platforms; ui++)
    {
        clGetPlatformInfo(
            Platform_IDs[ui],
            CL_PLATFORM_VENDOR,
            0,
            nullptr,
            &size
        );

        char *platform_info =
            new char[size];

        clGetPlatformInfo(
            Platform_IDs[ui],
            CL_PLATFORM_VENDOR,
            size,
            platform_info,
            nullptr
        );

        if (
            strcmp(
                platform_info,
                Target_Platform_Vendor
            ) == 0
        )
        {
            Platform_Detected = true;
            Target_Platform_ID =
                Platform_IDs[ui];

#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected platform             : "
                 << Target_Platform_Vendor << endl;
#endif
        }

        delete[] platform_info;
    }

    if (!Platform_Detected)
    {
        cout << "HOST-Error: Requested platform not found" << endl;
        return EXIT_FAILURE;
    }

    errCode = clGetDeviceIDs(
        Target_Platform_ID,
        CL_DEVICE_TYPE_ALL,
        0,
        nullptr,
        &Nb_Of_Devices
    );

    if (errCode != CL_SUCCESS || Nb_Of_Devices == 0)
    {
        cout << "HOST-Error: No OpenCL device found" << endl;
        return EXIT_FAILURE;
    }

    Device_IDs =
        new cl_device_id[Nb_Of_Devices];

    errCode = clGetDeviceIDs(
        Target_Platform_ID,
        CL_DEVICE_TYPE_ALL,
        Nb_Of_Devices,
        Device_IDs,
        nullptr
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: clGetDeviceIDs failed" << endl;
        return EXIT_FAILURE;
    }

    for (ui = 0; ui < Nb_Of_Devices; ui++)
    {
        clGetDeviceInfo(
            Device_IDs[ui],
            CL_DEVICE_NAME,
            0,
            nullptr,
            &size
        );

        char *device_info =
            new char[size];

        clGetDeviceInfo(
            Device_IDs[ui],
            CL_DEVICE_NAME,
            size,
            device_info,
            nullptr
        );

        if (
            strcmp(
                device_info,
                Target_Device_Name
            ) == 0
        )
        {
            Device_Detected = true;
            Target_Device_ID =
                Device_IDs[ui];

#ifdef ALL_MESSAGES
            cout << "HOST-Info: Selected device               : "
                 << Target_Device_Name << endl
                 << endl;
#endif
        }

        delete[] device_info;
    }

    if (!Device_Detected)
    {
        cout << "HOST-Error: Requested device not found" << endl;
        return EXIT_FAILURE;
    }

    Context = clCreateContext(
        nullptr,
        1,
        &Target_Device_ID,
        nullptr,
        nullptr,
        &errCode
    );

    if (errCode != CL_SUCCESS || Context == nullptr)
    {
        cout << "HOST-Error: Failed to create context" << endl;
        return EXIT_FAILURE;
    }

    Command_Queue = clCreateCommandQueue(
        Context,
        Target_Device_ID,
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
            CL_QUEUE_PROFILING_ENABLE,
        &errCode
    );

    if (errCode != CL_SUCCESS || Command_Queue == nullptr)
    {
        cout << "HOST-Error: Failed to create command queue" << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernel" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    unsigned char *xclbin_Memory = nullptr;

    int program_length = loadFile2Memory(
        xclbinFilename,
        reinterpret_cast<char **>(
            &xclbin_Memory
        )
    );

    if (program_length < 0)
    {
        cout << "HOST-Error: Failed to load xclbin file" << endl;
        return EXIT_FAILURE;
    }

    size_t Program_Length_in_Bytes =
        static_cast<size_t>(
            program_length
        );

    cl_program Program = nullptr;
    cl_int Binary_Status = CL_SUCCESS;

    Program = clCreateProgramWithBinary(
        Context,
        1,
        &Target_Device_ID,
        &Program_Length_in_Bytes,
        const_cast<const unsigned char **>(
            &xclbin_Memory
        ),
        &Binary_Status,
        &errCode
    );

    if (
        errCode != CL_SUCCESS ||
        Binary_Status != CL_SUCCESS ||
        Program == nullptr
    )
    {
        cout << "HOST-Error: Failed to create program from xclbin" << endl;
        return EXIT_FAILURE;
    }

    errCode = clBuildProgram(
        Program,
        1,
        &Target_Device_ID,
        nullptr,
        nullptr,
        nullptr
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to build OpenCL program" << endl;
        return EXIT_FAILURE;
    }

    cl_kernel K_mha_output_projection =
        clCreateKernel(
            Program,
            "mha_output_projection",
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        K_mha_output_projection == nullptr
    )
    {
        cout << "HOST-Error: Failed to create K_mha_output_projection"
             << endl;

        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernel" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    float *Attention_Out = nullptr;
    float *Attention_Max = nullptr;
    int8_t *Wo = nullptr;
    float *X = nullptr;
    float *bo = nullptr;
    float *scale_wo = nullptr;
    float *Out = nullptr;

    void *ptr = nullptr;

    cout << "HOST-Info: Loading HBM_Attention_Out.txt ... ";

    if (
        !allocate_aligned(
            &ptr,
            FEATURE_MAP_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    Attention_Out =
        reinterpret_cast<float *>(ptr);

    if (
        !load_float_file(
            "HBM_Attention_Out.txt",
            Attention_Out,
            static_cast<size_t>(SEQ_LEN) *
                static_cast<size_t>(EMBED_DIM)
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read HBM_Attention_Out.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Loading HBM_Attention_Max.txt ... ";

    if (
        !allocate_aligned(
            &ptr,
            ATTENTION_MAX_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    Attention_Max =
        reinterpret_cast<float *>(ptr);

    if (
        !load_float_file(
            "HBM_Attention_Max.txt",
            Attention_Max,
            SEQ_LEN
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read HBM_Attention_Max.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Loading Wo_transposed.txt ... ";

    if (
        !allocate_aligned(
            &ptr,
            WEIGHT_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    Wo = reinterpret_cast<int8_t *>(ptr);

    if (
        !load_int8_file(
            "Wo_transposed.txt",
            Wo,
            static_cast<size_t>(EMBED_DIM) *
                static_cast<size_t>(EMBED_DIM)
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read Wo_transposed.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Loading input_x.txt for residual ... ";

    if (
        !allocate_aligned(
            &ptr,
            FEATURE_MAP_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    X = reinterpret_cast<float *>(ptr);

    if (
        !load_float_file(
            "input_x.txt",
            X,
            static_cast<size_t>(SEQ_LEN) *
                static_cast<size_t>(EMBED_DIM)
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read input_x.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Loading bo.txt ... ";

    if (
        !allocate_aligned(
            &ptr,
            PARAM_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    bo = reinterpret_cast<float *>(ptr);

    if (
        !load_float_file(
            "bo.txt",
            bo,
            EMBED_DIM
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read bo.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Loading scale_wo.txt ... ";

    if (
        !allocate_aligned(
            &ptr,
            PARAM_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    scale_wo =
        reinterpret_cast<float *>(ptr);

    if (
        !load_float_file(
            "scale_wo.txt",
            scale_wo,
            EMBED_DIM
        )
    )
    {
        cout << "Failed" << endl;
        cout << "HOST-Error: Cannot read scale_wo.txt" << endl;
        return EXIT_FAILURE;
    }

    cout << "Done" << endl;

    cout << "HOST-Info: Allocating final output ... ";

    if (
        !allocate_aligned(
            &ptr,
            FEATURE_MAP_BYTES
        )
    )
    {
        cout << "Failed" << endl;
        return EXIT_FAILURE;
    }

    Out = reinterpret_cast<float *>(ptr);

    cout << "Done" << endl
         << endl;

    cl_mem GlobMem_BUF_Attention_Out = nullptr;
    cl_mem GlobMem_BUF_Attention_Max = nullptr;
    cl_mem GlobMem_BUF_Wo = nullptr;
    cl_mem GlobMem_BUF_X = nullptr;
    cl_mem GlobMem_BUF_bo = nullptr;
    cl_mem GlobMem_BUF_scale_wo = nullptr;
    cl_mem GlobMem_BUF_Out = nullptr;

    GlobMem_BUF_Attention_Out = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FEATURE_MAP_BYTES,
        Attention_Out,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Attention output buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Attention_Max = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        ATTENTION_MAX_BYTES,
        Attention_Max,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Attention max buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wo = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        WEIGHT_BYTES,
        Wo,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Wo buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_X = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FEATURE_MAP_BYTES,
        X,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create residual X buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bo = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        bo,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create bo buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wo = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        scale_wo,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create scale_wo buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Out = clCreateBuffer(
        Context,
        CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
        FEATURE_MAP_BYTES,
        Out,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create output buffer" << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 5) Set Kernel Arguments and Run" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    cl_event Mem_op_event[NB_OF_MEM_EVENTS] = {};
    cl_event K_exe_event[NB_OF_EXE_EVENTS] = {};
    cl_event Mem_read_event[NB_OF_READ_EVENTS] = {};

    errCode = CL_SUCCESS;

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        0,
        sizeof(cl_mem),
        &GlobMem_BUF_Attention_Out
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        1,
        sizeof(cl_mem),
        &GlobMem_BUF_Attention_Max
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        2,
        sizeof(cl_mem),
        &GlobMem_BUF_Wo
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        3,
        sizeof(cl_mem),
        &GlobMem_BUF_X
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        4,
        sizeof(cl_mem),
        &GlobMem_BUF_bo
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        5,
        sizeof(cl_mem),
        &GlobMem_BUF_scale_wo
    );

    errCode |= clSetKernelArg(
        K_mha_output_projection,
        6,
        sizeof(cl_mem),
        &GlobMem_BUF_Out
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to set kernel arguments, errCode = "
             << errCode << endl;

        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Attention_Out,
        0,
        0,
        nullptr,
        &Mem_op_event[0]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Attention_Max,
        0,
        0,
        nullptr,
        &Mem_op_event[1]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Wo,
        0,
        0,
        nullptr,
        &Mem_op_event[2]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_X,
        0,
        0,
        nullptr,
        &Mem_op_event[3]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_bo,
        0,
        0,
        nullptr,
        &Mem_op_event[4]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_scale_wo,
        0,
        0,
        nullptr,
        &Mem_op_event[5]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Out,
        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        0,
        nullptr,
        &Mem_op_event[6]
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to migrate memory objects, errCode = "
             << errCode << endl;

        return EXIT_FAILURE;
    }

    errCode = clEnqueueBarrierWithWaitList(
        Command_Queue,
        0,
        nullptr,
        nullptr
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to enqueue memory barrier" << endl;
        return EXIT_FAILURE;
    }

    cout << "HOST-Info: Submitting Kernel K_mha_output_projection ..."
         << endl;

    errCode = clEnqueueTask(
        Command_Queue,
        K_mha_output_projection,
        0,
        nullptr,
        &K_exe_event[0]
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to enqueue kernel, errCode = "
             << errCode << endl;

        return EXIT_FAILURE;
    }

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Out,
        CL_MIGRATE_MEM_OBJECT_HOST,
        1,
        &K_exe_event[0],
        &Mem_read_event[0]
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to migrate final output to Host"
             << endl;

        return EXIT_FAILURE;
    }

    cout << "HOST-Info: Waiting for application to be completed ..."
         << endl;

    errCode = clFinish(Command_Queue);

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: clFinish failed" << endl;
        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 6) Store Output Results" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    const char Out_File_Name[] =
        "Out_output.txt";

    cout << "HOST-Info: Store final output results in: "
         << Out_File_Name << endl;

    ofstream Out_File(
        Out_File_Name,
        ios::out
    );

    if (!Out_File)
    {
        cout << "HOST-Error: Cannot open output file" << endl;
        return EXIT_FAILURE;
    }

    Out_File << setprecision(9);

    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < EMBED_DIM; j++)
        {
            Out_File
                << Out[
                    i * EMBED_DIM + j
                ];

            if (j + 1 < EMBED_DIM)
            {
                Out_File << ' ';
            }
        }

        Out_File << '\n';
    }

    Out_File.close();

    cout << "HOST-Info: Results stored successfully" << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 7) Custom Profiling" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    string list_of_kernel_names[NB_OF_EXE_EVENTS] =
    {
        "mha_output_projection"
    };

    run_custom_profiling(
        NB_OF_EXE_EVENTS,
        NB_OF_MEM_EVENTS,
        K_exe_event,
        Mem_op_event,
        list_of_kernel_names
    );

    for (int i = 0; i < NB_OF_MEM_EVENTS; i++)
    {
        if (Mem_op_event[i] != nullptr)
        {
            clReleaseEvent(Mem_op_event[i]);
        }
    }

    for (int i = 0; i < NB_OF_EXE_EVENTS; i++)
    {
        if (K_exe_event[i] != nullptr)
        {
            clReleaseEvent(K_exe_event[i]);
        }
    }

    for (int i = 0; i < NB_OF_READ_EVENTS; i++)
    {
        if (Mem_read_event[i] != nullptr)
        {
            clReleaseEvent(Mem_read_event[i]);
        }
    }

    clReleaseMemObject(GlobMem_BUF_Attention_Out);
    clReleaseMemObject(GlobMem_BUF_Attention_Max);
    clReleaseMemObject(GlobMem_BUF_Wo);
    clReleaseMemObject(GlobMem_BUF_X);
    clReleaseMemObject(GlobMem_BUF_bo);
    clReleaseMemObject(GlobMem_BUF_scale_wo);
    clReleaseMemObject(GlobMem_BUF_Out);

    clReleaseKernel(K_mha_output_projection);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;

    free(Attention_Out);
    free(Attention_Max);
    free(Wo);
    free(X);
    free(bo);
    free(scale_wo);
    free(Out);

    free(xclbin_Memory);

    cout << endl
         << "HOST-Info: DONE"
         << endl
         << endl;

    return EXIT_SUCCESS;
}
