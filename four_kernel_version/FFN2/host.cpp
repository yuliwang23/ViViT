/*******************************************************************************
** HOST Code for FFN2
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
#include "FFN2.h"

using namespace std;

#define ALL_MESSAGES

constexpr size_t FEATURE_MAP_ELEMENTS =
    static_cast<size_t>(SEQ_LEN) *
    static_cast<size_t>(EMBED_DIM);

constexpr size_t INTERMEDIATE_ELEMENTS =
    static_cast<size_t>(SEQ_LEN) *
    static_cast<size_t>(MLP_DIM);

constexpr size_t W2_ELEMENTS =
    static_cast<size_t>(EMBED_DIM) *
    static_cast<size_t>(MLP_DIM);

constexpr size_t FEATURE_MAP_BYTES =
    FEATURE_MAP_ELEMENTS *
    sizeof(float);

constexpr size_t INTERMEDIATE_BYTES =
    INTERMEDIATE_ELEMENTS *
    sizeof(float);

constexpr size_t GELU_MAX_BYTES =
    static_cast<size_t>(SEQ_LEN) *
    sizeof(float);

constexpr size_t W2_BYTES =
    W2_ELEMENTS *
    sizeof(int8_t);

constexpr size_t EMBED_PARAM_BYTES =
    static_cast<size_t>(EMBED_DIM) *
    sizeof(float);

constexpr int NB_OF_MEM_EVENTS =
    7;

constexpr int NB_OF_EXE_EVENTS =
    1;

template <typename T>
static bool allocate_aligned(
    T **ptr,
    size_t count
) {
    void *raw_ptr =
        nullptr;

    if (
        posix_memalign(
            &raw_ptr,
            4096,
            count * sizeof(T)
        ) != 0
    ) {
        return false;
    }

    *ptr =
        reinterpret_cast<T *>(raw_ptr);

    return true;
}

static bool load_float_file(
    const char *filename,
    float *data,
    size_t count
) {
    ifstream input_file(
        filename
    );

    if (!input_file) {
        cout
            << "HOST-Error: Cannot open "
            << filename
            << endl;

        return false;
    }

    for (size_t i = 0; i < count; i++) {
        if (!(input_file >> data[i])) {
            cout
                << "HOST-Error: Invalid or incomplete file: "
                << filename
                << endl;

            return false;
        }
    }

    return true;
}

static bool load_int8_file(
    const char *filename,
    int8_t *data,
    size_t count
) {
    ifstream input_file(
        filename
    );

    if (!input_file) {
        cout
            << "HOST-Error: Cannot open "
            << filename
            << endl;

        return false;
    }

    int value =
        0;

    for (size_t i = 0; i < count; i++) {
        if (!(input_file >> value)) {
            cout
                << "HOST-Error: Invalid or incomplete file: "
                << filename
                << endl;

            return false;
        }

        data[i] =
            static_cast<int8_t>(
                value
            );
    }

    return true;
}

static bool store_output_file(
    const char *filename,
    const float *data
) {
    ofstream output_file(
        filename,
        ios::out
    );

    if (!output_file) {
        cout
            << "HOST-Error: Cannot open "
            << filename
            << endl;

        return false;
    }

    output_file
        << setprecision(9);

    for (int token = 0; token < SEQ_LEN; token++) {
        for (
            int channel = 0;
            channel < EMBED_DIM;
            channel++
        ) {
            output_file
                << data[
                    static_cast<size_t>(token) *
                    static_cast<size_t>(EMBED_DIM) +
                    static_cast<size_t>(channel)
                ];

            if (
                channel + 1 <
                EMBED_DIM
            ) {
                output_file
                    << ' ';
            }
        }

        output_file
            << '\n';
    }

    return true;
}

int main(
    int argc,
    char *argv[]
) {
    cout
        << endl;

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 1) Check Command Line Arguments" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    if (argc != 4) {
        cout
            << "HOST-Error: Incorrect command line syntax"
            << endl;

        cout
            << "HOST-Info:  Usage: "
            << argv[0]
            << " <Platform_Vendor> <Device_Name> <XCLBIN_File>"
            << endl
            << endl;

        return EXIT_FAILURE;
    }

    const char *Target_Platform_Vendor =
        argv[1];

    const char *Target_Device_Name =
        argv[2];

    const char *xclbinFilename =
        argv[3];

    cout
        << "HOST-Info: Platform_Vendor   : "
        << Target_Platform_Vendor
        << endl;

    cout
        << "HOST-Info: Device_Name       : "
        << Target_Device_Name
        << endl;

    cout
        << "HOST-Info: XCLBIN_file       : "
        << xclbinFilename
        << endl;

    cl_uint ui =
        0;

    cl_platform_id *Platform_IDs =
        nullptr;

    cl_uint Nb_Of_Platforms =
        0;

    cl_platform_id Target_Platform_ID =
        nullptr;

    bool Platform_Detected =
        false;

    cl_device_id *Device_IDs =
        nullptr;

    cl_uint Nb_Of_Devices =
        0;

    cl_device_id Target_Device_ID =
        nullptr;

    bool Device_Detected =
        false;

    cl_context Context =
        nullptr;

    cl_command_queue Command_Queue =
        nullptr;

    cl_program Program =
        nullptr;

    cl_kernel K_FFN2 =
        nullptr;

    unsigned char *xclbin_Memory =
        nullptr;

    cl_int errCode =
        CL_SUCCESS;

    size_t size =
        0;

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 2) Detect Platform and Device" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    errCode =
        clGetPlatformIDs(
            0,
            nullptr,
            &Nb_Of_Platforms
        );

    if (
        errCode != CL_SUCCESS ||
        Nb_Of_Platforms == 0
    ) {
        cout
            << "HOST-Error: No OpenCL platform found"
            << endl;

        return EXIT_FAILURE;
    }

    Platform_IDs =
        new cl_platform_id[
            Nb_Of_Platforms
        ];

    errCode =
        clGetPlatformIDs(
            Nb_Of_Platforms,
            Platform_IDs,
            nullptr
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: clGetPlatformIDs failed"
            << endl;

        return EXIT_FAILURE;
    }

    for (
        ui = 0;
        ui < Nb_Of_Platforms;
        ui++
    ) {
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
        ) {
            Platform_Detected =
                true;

            Target_Platform_ID =
                Platform_IDs[ui];
        }

        delete[]
            platform_info;
    }

    if (!Platform_Detected) {
        cout
            << "HOST-Error: Requested platform not found"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        clGetDeviceIDs(
            Target_Platform_ID,
            CL_DEVICE_TYPE_ALL,
            0,
            nullptr,
            &Nb_Of_Devices
        );

    if (
        errCode != CL_SUCCESS ||
        Nb_Of_Devices == 0
    ) {
        cout
            << "HOST-Error: No OpenCL device found"
            << endl;

        return EXIT_FAILURE;
    }

    Device_IDs =
        new cl_device_id[
            Nb_Of_Devices
        ];

    errCode =
        clGetDeviceIDs(
            Target_Platform_ID,
            CL_DEVICE_TYPE_ALL,
            Nb_Of_Devices,
            Device_IDs,
            nullptr
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: clGetDeviceIDs failed"
            << endl;

        return EXIT_FAILURE;
    }

    for (
        ui = 0;
        ui < Nb_Of_Devices;
        ui++
    ) {
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
        ) {
            Device_Detected =
                true;

            Target_Device_ID =
                Device_IDs[ui];
        }

        delete[]
            device_info;
    }

    if (!Device_Detected) {
        cout
            << "HOST-Error: Requested device not found"
            << endl;

        return EXIT_FAILURE;
    }

    Context =
        clCreateContext(
            nullptr,
            1,
            &Target_Device_ID,
            nullptr,
            nullptr,
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        Context == nullptr
    ) {
        cout
            << "HOST-Error: Failed to create context"
            << endl;

        return EXIT_FAILURE;
    }

    Command_Queue =
        clCreateCommandQueue(
            Context,
            Target_Device_ID,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE |
            CL_QUEUE_PROFILING_ENABLE,
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        Command_Queue == nullptr
    ) {
        cout
            << "HOST-Error: Failed to create command queue"
            << endl;

        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 3) Create Program and Kernel" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    int program_length =
        loadFile2Memory(
            xclbinFilename,
            reinterpret_cast<char **>(
                &xclbin_Memory
            )
        );

    if (program_length < 0) {
        cout
            << "HOST-Error: Failed to load xclbin"
            << endl;

        return EXIT_FAILURE;
    }

    size_t Program_Length_in_Bytes =
        static_cast<size_t>(
            program_length
        );

    cl_int Binary_Status =
        CL_SUCCESS;

    Program =
        clCreateProgramWithBinary(
            Context,
            1,
            &Target_Device_ID,
            &Program_Length_in_Bytes,
            const_cast<
                const unsigned char **
            >(
                &xclbin_Memory
            ),
            &Binary_Status,
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        Binary_Status != CL_SUCCESS ||
        Program == nullptr
    ) {
        cout
            << "HOST-Error: Failed to create program"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        clBuildProgram(
            Program,
            1,
            &Target_Device_ID,
            nullptr,
            nullptr,
            nullptr
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to build program"
            << endl;

        return EXIT_FAILURE;
    }

    K_FFN2 =
        clCreateKernel(
            Program,
            "FFN2",
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        K_FFN2 == nullptr
    ) {
        cout
            << "HOST-Error: Failed to create FFN2 kernel"
            << endl;

        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 4) Allocate and Load Data" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    float *HBM_FP32_Intermediate =
        nullptr;

    float *HBM_GELU_Max =
        nullptr;

    int8_t *W2 =
        nullptr;

    float *X =
        nullptr;

    float *b2 =
        nullptr;

    float *scale_w2 =
        nullptr;

    float *Out =
        nullptr;

    if (
        !allocate_aligned(
            &HBM_FP32_Intermediate,
            INTERMEDIATE_ELEMENTS
        ) ||
        !allocate_aligned(
            &HBM_GELU_Max,
            SEQ_LEN
        ) ||
        !allocate_aligned(
            &W2,
            W2_ELEMENTS
        ) ||
        !allocate_aligned(
            &X,
            FEATURE_MAP_ELEMENTS
        ) ||
        !allocate_aligned(
            &b2,
            EMBED_DIM
        ) ||
        !allocate_aligned(
            &scale_w2,
            EMBED_DIM
        ) ||
        !allocate_aligned(
            &Out,
            FEATURE_MAP_ELEMENTS
        )
    ) {
        cout
            << "HOST-Error: Host memory allocation failed"
            << endl;

        return EXIT_FAILURE;
    }

    if (
        !load_float_file(
            "HBM_FP32_Intermediate.txt",
            HBM_FP32_Intermediate,
            INTERMEDIATE_ELEMENTS
        ) ||
        !load_float_file(
            "HBM_GELU_Max.txt",
            HBM_GELU_Max,
            SEQ_LEN
        ) ||
        !load_int8_file(
            "W2_transposed.txt",
            W2,
            W2_ELEMENTS
        ) ||
        !load_float_file(
            "input_x.txt",
            X,
            FEATURE_MAP_ELEMENTS
        ) ||
        !load_float_file(
            "b2.txt",
            b2,
            EMBED_DIM
        ) ||
        !load_float_file(
            "scale_w2.txt",
            scale_w2,
            EMBED_DIM
        )
    ) {
        return EXIT_FAILURE;
    }

    cl_mem GlobMem_BUF_HBM_Intermediate =
        nullptr;

    cl_mem GlobMem_BUF_HBM_GELU_Max =
        nullptr;

    cl_mem GlobMem_BUF_W2 =
        nullptr;

    cl_mem GlobMem_BUF_X =
        nullptr;

    cl_mem GlobMem_BUF_b2 =
        nullptr;

    cl_mem GlobMem_BUF_scale_w2 =
        nullptr;

    cl_mem GlobMem_BUF_Out =
        nullptr;

    GlobMem_BUF_HBM_Intermediate =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            INTERMEDIATE_BYTES,
            HBM_FP32_Intermediate,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create intermediate buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_HBM_GELU_Max =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            GELU_MAX_BYTES,
            HBM_GELU_Max,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create GELU max buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_W2 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            W2_BYTES,
            W2,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create W2 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_X =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            FEATURE_MAP_BYTES,
            X,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create X buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_b2 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            EMBED_PARAM_BYTES,
            b2,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create b2 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_w2 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            EMBED_PARAM_BYTES,
            scale_w2,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create scale_w2 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_Out =
        clCreateBuffer(
            Context,
            CL_MEM_WRITE_ONLY |
            CL_MEM_USE_HOST_PTR,
            FEATURE_MAP_BYTES,
            Out,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create output buffer"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        CL_SUCCESS;

    errCode |=
        clSetKernelArg(
            K_FFN2,
            0,
            sizeof(cl_mem),
            &GlobMem_BUF_HBM_Intermediate
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            1,
            sizeof(cl_mem),
            &GlobMem_BUF_HBM_GELU_Max
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            2,
            sizeof(cl_mem),
            &GlobMem_BUF_W2
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            3,
            sizeof(cl_mem),
            &GlobMem_BUF_X
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            4,
            sizeof(cl_mem),
            &GlobMem_BUF_b2
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            5,
            sizeof(cl_mem),
            &GlobMem_BUF_scale_w2
        );

    errCode |=
        clSetKernelArg(
            K_FFN2,
            6,
            sizeof(cl_mem),
            &GlobMem_BUF_Out
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to set FFN2 arguments"
            << endl;

        return EXIT_FAILURE;
    }

    cl_event Mem_op_event[
        NB_OF_MEM_EVENTS
    ] = {};

    cl_event K_exe_event[
        NB_OF_EXE_EVENTS
    ] = {};

    cl_event Mem_read_event =
        nullptr;

    errCode =
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_HBM_Intermediate,
            0,
            0,
            nullptr,
            &Mem_op_event[0]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_HBM_GELU_Max,
            0,
            0,
            nullptr,
            &Mem_op_event[1]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_W2,
            0,
            0,
            nullptr,
            &Mem_op_event[2]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_X,
            0,
            0,
            nullptr,
            &Mem_op_event[3]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_b2,
            0,
            0,
            nullptr,
            &Mem_op_event[4]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_scale_w2,
            0,
            0,
            nullptr,
            &Mem_op_event[5]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_Out,
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
            0,
            nullptr,
            &Mem_op_event[6]
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to migrate FFN2 buffers"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        clEnqueueBarrierWithWaitList(
            Command_Queue,
            0,
            nullptr,
            nullptr
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to enqueue barrier"
            << endl;

        return EXIT_FAILURE;
    }

    cout
        << "HOST-Info: Submitting Kernel FFN2 ..."
        << endl;

    errCode =
        clEnqueueTask(
            Command_Queue,
            K_FFN2,
            0,
            nullptr,
            &K_exe_event[0]
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to enqueue FFN2"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_Out,
            CL_MIGRATE_MEM_OBJECT_HOST,
            1,
            &K_exe_event[0],
            &Mem_read_event
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to copy FFN2 output to Host"
            << endl;

        return EXIT_FAILURE;
    }

    errCode =
        clFinish(
            Command_Queue
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: clFinish failed"
            << endl;

        return EXIT_FAILURE;
    }

    const char Output_File_Name[] =
        "Out_mlp_output.txt";

    cout
        << "HOST-Info: Store FFN2 output in: "
        << Output_File_Name
        << endl;

    if (
        !store_output_file(
            Output_File_Name,
            Out
        )
    ) {
        return EXIT_FAILURE;
    }

    string list_of_kernel_names[
        NB_OF_EXE_EVENTS
    ] =
    {
        "FFN2"
    };

    run_custom_profiling(
        NB_OF_EXE_EVENTS,
        NB_OF_MEM_EVENTS,
        K_exe_event,
        Mem_op_event,
        list_of_kernel_names
    );

    for (
        int i = 0;
        i < NB_OF_MEM_EVENTS;
        i++
    ) {
        if (
            Mem_op_event[i] !=
            nullptr
        ) {
            clReleaseEvent(
                Mem_op_event[i]
            );
        }
    }

    if (
        K_exe_event[0] !=
        nullptr
    ) {
        clReleaseEvent(
            K_exe_event[0]
        );
    }

    if (
        Mem_read_event !=
        nullptr
    ) {
        clReleaseEvent(
            Mem_read_event
        );
    }

    clReleaseMemObject(
        GlobMem_BUF_HBM_Intermediate
    );

    clReleaseMemObject(
        GlobMem_BUF_HBM_GELU_Max
    );

    clReleaseMemObject(
        GlobMem_BUF_W2
    );

    clReleaseMemObject(
        GlobMem_BUF_X
    );

    clReleaseMemObject(
        GlobMem_BUF_b2
    );

    clReleaseMemObject(
        GlobMem_BUF_scale_w2
    );

    clReleaseMemObject(
        GlobMem_BUF_Out
    );

    clReleaseKernel(
        K_FFN2
    );

    clReleaseProgram(
        Program
    );

    clReleaseCommandQueue(
        Command_Queue
    );

    clReleaseContext(
        Context
    );

    delete[]
        Platform_IDs;

    delete[]
        Device_IDs;

    free(HBM_FP32_Intermediate);
    free(HBM_GELU_Max);
    free(W2);
    free(X);
    free(b2);
    free(scale_w2);
    free(Out);
    free(xclbin_Memory);

    cout
        << endl
        << "HOST-Info: FFN2 DONE"
        << endl
        << endl;

    return EXIT_SUCCESS;
}
