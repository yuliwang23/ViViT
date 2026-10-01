/*******************************************************************************
** HOST Code for FFN1
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
#include "FFN1.h"

using namespace std;

#define ALL_MESSAGES

constexpr size_t FEATURE_MAP_ELEMENTS =
    static_cast<size_t>(SEQ_LEN) *
    static_cast<size_t>(EMBED_DIM);

constexpr size_t INTERMEDIATE_ELEMENTS =
    static_cast<size_t>(SEQ_LEN) *
    static_cast<size_t>(MLP_DIM);

constexpr size_t W1_ELEMENTS =
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

constexpr size_t W1_BYTES =
    W1_ELEMENTS *
    sizeof(int8_t);

constexpr size_t EMBED_PARAM_BYTES =
    static_cast<size_t>(EMBED_DIM) *
    sizeof(float);

constexpr size_t MLP_PARAM_BYTES =
    static_cast<size_t>(MLP_DIM) *
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

static bool store_intermediate_file(
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

    for (
        int tile = 0;
        tile < MLP1_NUM_TILES;
        tile++
    ) {
        for (
            int token = 0;
            token < SEQ_LEN;
            token++
        ) {
            size_t base =
                (
                    static_cast<size_t>(tile) *
                    static_cast<size_t>(SEQ_LEN) +
                    static_cast<size_t>(token)
                ) *
                static_cast<size_t>(TILE_OUT_DIM);

            for (
                int channel = 0;
                channel < TILE_OUT_DIM;
                channel++
            ) {
                output_file
                    << data[
                        base +
                        static_cast<size_t>(channel)
                    ];

                if (
                    channel + 1 <
                    TILE_OUT_DIM
                ) {
                    output_file
                        << ' ';
                }
            }

            output_file
                << '\n';
        }
    }

    return true;
}

static bool store_gelu_max_file(
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

    for (int i = 0; i < SEQ_LEN; i++) {
        output_file
            << data[i]
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

    cl_kernel K_FFN1 =
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

    K_FFN1 =
        clCreateKernel(
            Program,
            "FFN1",
            &errCode
        );

    if (
        errCode != CL_SUCCESS ||
        K_FFN1 == nullptr
    ) {
        cout
            << "HOST-Error: Failed to create FFN1 kernel"
            << endl;

        return EXIT_FAILURE;
    }

#ifdef ALL_MESSAGES
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 4) Allocate and Load Data" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    float *X =
        nullptr;

    float *gamma =
        nullptr;

    float *beta =
        nullptr;

    int8_t *W1 =
        nullptr;

    float *b1 =
        nullptr;

    float *scale_w1 =
        nullptr;

    float *HBM_FP32_Intermediate =
        nullptr;

    float *HBM_GELU_Max =
        nullptr;

    if (
        !allocate_aligned(
            &X,
            FEATURE_MAP_ELEMENTS
        ) ||
        !allocate_aligned(
            &gamma,
            EMBED_DIM
        ) ||
        !allocate_aligned(
            &beta,
            EMBED_DIM
        ) ||
        !allocate_aligned(
            &W1,
            W1_ELEMENTS
        ) ||
        !allocate_aligned(
            &b1,
            MLP_DIM
        ) ||
        !allocate_aligned(
            &scale_w1,
            MLP_DIM
        ) ||
        !allocate_aligned(
            &HBM_FP32_Intermediate,
            INTERMEDIATE_ELEMENTS
        ) ||
        !allocate_aligned(
            &HBM_GELU_Max,
            SEQ_LEN
        )
    ) {
        cout
            << "HOST-Error: Host memory allocation failed"
            << endl;

        return EXIT_FAILURE;
    }

    if (
        !load_float_file(
            "input_x.txt",
            X,
            FEATURE_MAP_ELEMENTS
        ) ||
        !load_float_file(
            "gamma.txt",
            gamma,
            EMBED_DIM
        ) ||
        !load_float_file(
            "beta.txt",
            beta,
            EMBED_DIM
        ) ||
        !load_int8_file(
            "W1_transposed.txt",
            W1,
            W1_ELEMENTS
        ) ||
        !load_float_file(
            "b1.txt",
            b1,
            MLP_DIM
        ) ||
        !load_float_file(
            "scale_w1.txt",
            scale_w1,
            MLP_DIM
        )
    ) {
        return EXIT_FAILURE;
    }

    cl_mem GlobMem_BUF_X =
        nullptr;

    cl_mem GlobMem_BUF_gamma =
        nullptr;

    cl_mem GlobMem_BUF_beta =
        nullptr;

    cl_mem GlobMem_BUF_W1 =
        nullptr;

    cl_mem GlobMem_BUF_b1 =
        nullptr;

    cl_mem GlobMem_BUF_scale_w1 =
        nullptr;

    cl_mem GlobMem_BUF_HBM_Intermediate =
        nullptr;

    cl_mem GlobMem_BUF_HBM_GELU_Max =
        nullptr;

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

    GlobMem_BUF_gamma =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            EMBED_PARAM_BYTES,
            gamma,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create gamma buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_beta =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            EMBED_PARAM_BYTES,
            beta,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create beta buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_W1 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            W1_BYTES,
            W1,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create W1 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_b1 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            MLP_PARAM_BYTES,
            b1,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create b1 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_w1 =
        clCreateBuffer(
            Context,
            CL_MEM_READ_ONLY |
            CL_MEM_USE_HOST_PTR,
            MLP_PARAM_BYTES,
            scale_w1,
            &errCode
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to create scale_w1 buffer"
            << endl;

        return EXIT_FAILURE;
    }

    GlobMem_BUF_HBM_Intermediate =
        clCreateBuffer(
            Context,
            CL_MEM_WRITE_ONLY |
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
            CL_MEM_WRITE_ONLY |
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

    errCode =
        CL_SUCCESS;

    errCode |=
        clSetKernelArg(
            K_FFN1,
            0,
            sizeof(cl_mem),
            &GlobMem_BUF_X
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            1,
            sizeof(cl_mem),
            &GlobMem_BUF_gamma
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            2,
            sizeof(cl_mem),
            &GlobMem_BUF_beta
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            3,
            sizeof(cl_mem),
            &GlobMem_BUF_W1
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            4,
            sizeof(cl_mem),
            &GlobMem_BUF_b1
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            5,
            sizeof(cl_mem),
            &GlobMem_BUF_scale_w1
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            6,
            sizeof(cl_mem),
            &GlobMem_BUF_HBM_Intermediate
        );

    errCode |=
        clSetKernelArg(
            K_FFN1,
            7,
            sizeof(cl_mem),
            &GlobMem_BUF_HBM_GELU_Max
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to set FFN1 arguments"
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
            &GlobMem_BUF_X,
            0,
            0,
            nullptr,
            &Mem_op_event[0]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_gamma,
            0,
            0,
            nullptr,
            &Mem_op_event[1]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_beta,
            0,
            0,
            nullptr,
            &Mem_op_event[2]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_W1,
            0,
            0,
            nullptr,
            &Mem_op_event[3]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_b1,
            0,
            0,
            nullptr,
            &Mem_op_event[4]
        );

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            1,
            &GlobMem_BUF_scale_w1,
            0,
            0,
            nullptr,
            &Mem_op_event[5]
        );

    cl_mem Undefined_Output_Buffers[2] =
    {
        GlobMem_BUF_HBM_Intermediate,
        GlobMem_BUF_HBM_GELU_Max
    };

    errCode |=
        clEnqueueMigrateMemObjects(
            Command_Queue,
            2,
            Undefined_Output_Buffers,
            CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
            0,
            nullptr,
            &Mem_op_event[6]
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to migrate FFN1 buffers"
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
        << "HOST-Info: Submitting Kernel FFN1 ..."
        << endl;

    errCode =
        clEnqueueTask(
            Command_Queue,
            K_FFN1,
            0,
            nullptr,
            &K_exe_event[0]
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to enqueue FFN1"
            << endl;

        return EXIT_FAILURE;
    }

    cl_mem Output_Buffers[2] =
    {
        GlobMem_BUF_HBM_Intermediate,
        GlobMem_BUF_HBM_GELU_Max
    };

    errCode =
        clEnqueueMigrateMemObjects(
            Command_Queue,
            2,
            Output_Buffers,
            CL_MIGRATE_MEM_OBJECT_HOST,
            1,
            &K_exe_event[0],
            &Mem_read_event
        );

    if (errCode != CL_SUCCESS) {
        cout
            << "HOST-Error: Failed to copy FFN1 outputs to Host"
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

    const char Intermediate_File_Name[] =
        "HBM_FP32_Intermediate.txt";

    const char GELU_Max_File_Name[] =
        "HBM_GELU_Max.txt";

    cout
        << "HOST-Info: Store intermediate output in: "
        << Intermediate_File_Name
        << endl;

    if (
        !store_intermediate_file(
            Intermediate_File_Name,
            HBM_FP32_Intermediate
        )
    ) {
        return EXIT_FAILURE;
    }

    cout
        << "HOST-Info: Store GELU max in: "
        << GELU_Max_File_Name
        << endl;

    if (
        !store_gelu_max_file(
            GELU_Max_File_Name,
            HBM_GELU_Max
        )
    ) {
        return EXIT_FAILURE;
    }

    string list_of_kernel_names[
        NB_OF_EXE_EVENTS
    ] =
    {
        "FFN1"
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
        GlobMem_BUF_X
    );

    clReleaseMemObject(
        GlobMem_BUF_gamma
    );

    clReleaseMemObject(
        GlobMem_BUF_beta
    );

    clReleaseMemObject(
        GlobMem_BUF_W1
    );

    clReleaseMemObject(
        GlobMem_BUF_b1
    );

    clReleaseMemObject(
        GlobMem_BUF_scale_w1
    );

    clReleaseMemObject(
        GlobMem_BUF_HBM_Intermediate
    );

    clReleaseMemObject(
        GlobMem_BUF_HBM_GELU_Max
    );

    clReleaseKernel(
        K_FFN1
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

    free(X);
    free(gamma);
    free(beta);
    free(W1);
    free(b1);
    free(scale_w1);
    free(HBM_FP32_Intermediate);
    free(HBM_GELU_Max);
    free(xclbin_Memory);

    cout
        << endl
        << "HOST-Info: FFN1 DONE"
        << endl
        << endl;

    return EXIT_SUCCESS;
}
