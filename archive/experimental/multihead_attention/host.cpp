/*******************************************************************************
** HOST Code for multihead_attention
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <CL/cl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "help_functions.h"
#include "multihead_attention.h"

using namespace std;

#define ALL_MESSAGES

// ============================================================================
// Buffer Size Constants
// ============================================================================

constexpr size_t FEATURE_MAP_BYTES =
    static_cast<size_t>(SEQ_LEN)
    * static_cast<size_t>(EMBED_DIM)
    * sizeof(float);

constexpr size_t ATTENTION_INTERMEDIATE_BYTES =
    FEATURE_MAP_BYTES;

constexpr size_t WEIGHT_BYTES =
    static_cast<size_t>(EMBED_DIM)
    * static_cast<size_t>(EMBED_DIM)
    * sizeof(int8_t);

constexpr size_t PARAM_BYTES =
    static_cast<size_t>(EMBED_DIM)
    * sizeof(float);

// 15 input migrations + 1 Attention intermediate allocation
// + 1 final output allocation
constexpr int NB_OF_MEM_EVENTS = 17;
constexpr int NB_OF_EXE_EVENTS = 1;

// ============================================================================
// Main
// ============================================================================

int main(int argc, char *argv[])
{
    cout << endl;

    // ========================================================================
    // Step 1: Check Command Line Arguments
    // ========================================================================

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
    const char *Target_Device_Name     = argv[2];
    const char *xclbinFilename         = argv[3];

    cout << "HOST-Info: Platform_Vendor   : "
         << Target_Platform_Vendor << endl;

    cout << "HOST-Info: Device_Name       : "
         << Target_Device_Name << endl;

    cout << "HOST-Info: XCLBIN_file       : "
         << xclbinFilename << endl;

    // ========================================================================
    // Step 2: Detect Platform and Device
    // ========================================================================

    cout << endl;

#ifdef ALL_MESSAGES
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

    // ------------------------------------------------------------------------
    // Step 2.1: Get Platforms
    // ------------------------------------------------------------------------

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

    Platform_IDs = new cl_platform_id[Nb_Of_Platforms];

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

        char *platform_info = new char[size];

        clGetPlatformInfo(
            Platform_IDs[ui],
            CL_PLATFORM_VENDOR,
            size,
            platform_info,
            nullptr
        );

        if (strcmp(platform_info, Target_Platform_Vendor) == 0)
        {
            Platform_Detected  = true;
            Target_Platform_ID = Platform_IDs[ui];

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

    // ------------------------------------------------------------------------
    // Step 2.2: Get Devices
    // ------------------------------------------------------------------------

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

    Device_IDs = new cl_device_id[Nb_Of_Devices];

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

        char *device_info = new char[size];

        clGetDeviceInfo(
            Device_IDs[ui],
            CL_DEVICE_NAME,
            size,
            device_info,
            nullptr
        );

        if (strcmp(device_info, Target_Device_Name) == 0)
        {
            Device_Detected  = true;
            Target_Device_ID = Device_IDs[ui];

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

    // ------------------------------------------------------------------------
    // Step 2.3: Create Context
    // ------------------------------------------------------------------------

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

    // ------------------------------------------------------------------------
    // Step 2.4: Create Out-of-Order Command Queue
    // ------------------------------------------------------------------------

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

    // ========================================================================
    // Step 3: Create Program and Kernel
    // ========================================================================

    unsigned char *xclbin_Memory = nullptr;

    int program_length = loadFile2Memory(
        xclbinFilename,
        reinterpret_cast<char **>(&xclbin_Memory)
    );

    if (program_length < 0)
    {
        cout << "HOST-Error: Failed to load xclbin file" << endl;
        return EXIT_FAILURE;
    }

    size_t Program_Length_in_Bytes =
        static_cast<size_t>(program_length);

    cl_program Program = nullptr;
    cl_int Binary_Status = CL_SUCCESS;

    Program = clCreateProgramWithBinary(
        Context,
        1,
        &Target_Device_ID,
        &Program_Length_in_Bytes,
        const_cast<const unsigned char **>(&xclbin_Memory),
        &Binary_Status,
        &errCode
    );

    if (errCode != CL_SUCCESS ||
        Binary_Status != CL_SUCCESS ||
        Program == nullptr)
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

    cl_kernel K_multihead_attention =
        clCreateKernel(
            Program,
            "multihead_attention",
            &errCode
        );

    if (errCode != CL_SUCCESS ||
        K_multihead_attention == nullptr)
    {
        cout << "HOST-Error: Failed to create K_multihead_attention"
             << endl;

        return EXIT_FAILURE;
    }

    // ========================================================================
    // Step 4: Prepare Host Data
    // ========================================================================

    float *X     = nullptr;
    float *gamma = nullptr;
    float *beta  = nullptr;

    int8_t *Wq = nullptr;
    int8_t *Wk = nullptr;
    int8_t *Wv = nullptr;
    int8_t *Wo = nullptr;

    float *bq = nullptr;
    float *bk = nullptr;
    float *bv = nullptr;
    float *bo = nullptr;

    float *scale_wq = nullptr;
    float *scale_wk = nullptr;
    float *scale_wv = nullptr;
    float *scale_wo = nullptr;

    float *Out = nullptr;

#ifdef ALL_MESSAGES
    cout << endl;
    cout << "HOST-Info: =============================================================" << endl;
    cout << "HOST-Info: (Step 4) Prepare Data to Run Kernel" << endl;
    cout << "HOST-Info: =============================================================" << endl;
#endif

    void *ptr = nullptr;

    // ------------------------------------------------------------------------
    // Step 4.1.1: Input Feature Map
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating and loading data for X ... ";

    if (posix_memalign(
            &ptr,
            4096,
            FEATURE_MAP_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    X = reinterpret_cast<float *>(ptr);

    {
        ifstream infile("input_x.txt");

        if (!infile)
        {
            cout << "Failed" << endl;
            cout << "HOST-Error: Cannot open input_x.txt" << endl;
            return EXIT_FAILURE;
        }

        for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++)
        {
            infile >> X[i];
        }
    }

    cout << "Done" << endl;

    // ------------------------------------------------------------------------
    // Step 4.1.2: Gamma and Beta
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating and loading data for gamma & beta ... ";

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    gamma = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    beta = reinterpret_cast<float *>(ptr);

    {
        ifstream f_gamma("gamma.txt");
        ifstream f_beta("beta.txt");

        if (!f_gamma || !f_beta)
        {
            cout << "Failed" << endl;
            cout << "HOST-Error: Cannot open gamma.txt or beta.txt"
                 << endl;

            return EXIT_FAILURE;
        }

        for (int i = 0; i < EMBED_DIM; i++)
        {
            f_gamma >> gamma[i];
            f_beta  >> beta[i];
        }
    }

    cout << "Done" << endl;

    // ------------------------------------------------------------------------
    // Step 4.1.3: INT8 Weights
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating and loading Quantized Weights "
         << "(Wq, Wk, Wv, Wo) ... ";

    if (posix_memalign(&ptr, 4096, WEIGHT_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    Wq = reinterpret_cast<int8_t *>(ptr);

    if (posix_memalign(&ptr, 4096, WEIGHT_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    Wk = reinterpret_cast<int8_t *>(ptr);

    if (posix_memalign(&ptr, 4096, WEIGHT_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    Wv = reinterpret_cast<int8_t *>(ptr);

    if (posix_memalign(&ptr, 4096, WEIGHT_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    Wo = reinterpret_cast<int8_t *>(ptr);

    {
        ifstream f_wq("Wq_transposed.txt");
        ifstream f_wk("Wk_transposed.txt");
        ifstream f_wv("Wv_transposed.txt");
        ifstream f_wo("Wo_transposed.txt");

        if (!f_wq || !f_wk || !f_wv || !f_wo)
        {
            cout << "Failed" << endl;
            cout << "HOST-Error: Cannot open one or more weight files"
                 << endl;

            return EXIT_FAILURE;
        }

        int temp_q = 0;
        int temp_k = 0;
        int temp_v = 0;
        int temp_o = 0;

        for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++)
        {
            f_wq >> temp_q;
            f_wk >> temp_k;
            f_wv >> temp_v;
            f_wo >> temp_o;

            Wq[i] = static_cast<int8_t>(temp_q);
            Wk[i] = static_cast<int8_t>(temp_k);
            Wv[i] = static_cast<int8_t>(temp_v);
            Wo[i] = static_cast<int8_t>(temp_o);
        }
    }

    cout << "Done" << endl;

    // ------------------------------------------------------------------------
    // Step 4.1.4: Biases
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating and loading Biases "
         << "(bq, bk, bv, bo) ... ";

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    bq = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    bk = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    bv = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    bo = reinterpret_cast<float *>(ptr);

    {
        ifstream f_bq("bq.txt");
        ifstream f_bk("bk.txt");
        ifstream f_bv("bv.txt");
        ifstream f_bo("bo.txt");

        if (!f_bq || !f_bk || !f_bv || !f_bo)
        {
            cout << "Failed" << endl;
            cout << "HOST-Error: Cannot open one or more bias files"
                 << endl;

            return EXIT_FAILURE;
        }

        for (int i = 0; i < EMBED_DIM; i++)
        {
            f_bq >> bq[i];
            f_bk >> bk[i];
            f_bv >> bv[i];
            f_bo >> bo[i];
        }
    }

    cout << "Done" << endl;

    // ------------------------------------------------------------------------
    // Step 4.1.5: Weight Scales
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating and loading Dequantization Scales ... ";

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    scale_wq = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    scale_wk = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    scale_wv = reinterpret_cast<float *>(ptr);

    if (posix_memalign(&ptr, 4096, PARAM_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    scale_wo = reinterpret_cast<float *>(ptr);

    {
        ifstream f_swq("scale_wq.txt");
        ifstream f_swk("scale_wk.txt");
        ifstream f_swv("scale_wv.txt");
        ifstream f_swo("scale_wo.txt");

        if (!f_swq || !f_swk || !f_swv || !f_swo)
        {
            cout << "Failed" << endl;
            cout << "HOST-Error: Cannot open one or more scale files"
                 << endl;

            return EXIT_FAILURE;
        }

        for (int i = 0; i < EMBED_DIM; i++)
        {
            f_swq >> scale_wq[i];
            f_swk >> scale_wk[i];
            f_swv >> scale_wv[i];
            f_swo >> scale_wo[i];
        }
    }

    cout << "Done" << endl;

    // ------------------------------------------------------------------------
    // Step 4.1.6: Final Output
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Allocating memory for final Output ... ";

    if (posix_memalign(
            &ptr,
            4096,
            FEATURE_MAP_BYTES) != 0)
    {
        return EXIT_FAILURE;
    }

    Out = reinterpret_cast<float *>(ptr);

    cout << "Done" << endl
         << endl;

    // ========================================================================
    // Step 4.2: Create OpenCL Global Memory Buffers
    // ========================================================================

    cl_mem GlobMem_BUF_X = nullptr;
    cl_mem GlobMem_BUF_gamma = nullptr;
    cl_mem GlobMem_BUF_beta = nullptr;

    cl_mem GlobMem_BUF_Wq = nullptr;
    cl_mem GlobMem_BUF_Wk = nullptr;
    cl_mem GlobMem_BUF_Wv = nullptr;
    cl_mem GlobMem_BUF_Wo = nullptr;

    cl_mem GlobMem_BUF_bq = nullptr;
    cl_mem GlobMem_BUF_bk = nullptr;
    cl_mem GlobMem_BUF_bv = nullptr;
    cl_mem GlobMem_BUF_bo = nullptr;

    cl_mem GlobMem_BUF_scale_wq = nullptr;
    cl_mem GlobMem_BUF_scale_wk = nullptr;
    cl_mem GlobMem_BUF_scale_wv = nullptr;
    cl_mem GlobMem_BUF_scale_wo = nullptr;

    // Device-side FP32 Softmax × V intermediate
    cl_mem GlobMem_BUF_Attention_Out = nullptr;

    // Final FP32 output
    cl_mem GlobMem_BUF_Out = nullptr;

    GlobMem_BUF_X = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        FEATURE_MAP_BYTES,
        X,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create X buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_gamma = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        gamma,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create gamma buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_beta = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        beta,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create beta buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wq = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        WEIGHT_BYTES,
        Wq,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Wq buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wk = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        WEIGHT_BYTES,
        Wk,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Wk buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_Wv = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        WEIGHT_BYTES,
        Wv,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Wv buffer" << endl;
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

    GlobMem_BUF_bq = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        bq,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create bq buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bk = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        bk,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create bk buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_bv = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        bv,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create bv buffer" << endl;
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

    GlobMem_BUF_scale_wq = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        scale_wq,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create scale_wq buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wk = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        scale_wk,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create scale_wk buffer" << endl;
        return EXIT_FAILURE;
    }

    GlobMem_BUF_scale_wv = clCreateBuffer(
        Context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        PARAM_BYTES,
        scale_wv,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create scale_wv buffer" << endl;
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

    // ------------------------------------------------------------------------
    // New FP32 Attention Intermediate Buffer
    //
    // Size:
    //
    //   197 × 768 × 4 bytes
    //   = 605184 bytes
    //
    // Kernel first writes Softmax × V into this buffer, then reads it again
    // during the output projection stage.
    // ------------------------------------------------------------------------

    GlobMem_BUF_Attention_Out = clCreateBuffer(
        Context,
        CL_MEM_READ_WRITE,
        ATTENTION_INTERMEDIATE_BYTES,
        nullptr,
        &errCode
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to create Attention intermediate buffer"
             << endl;

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
        cout << "HOST-Error: Failed to create final output buffer" << endl;
        return EXIT_FAILURE;
    }

    // ========================================================================
    // Step 5: Set Kernel Arguments and Run
    // ========================================================================

    cl_event Mem_op_event[NB_OF_MEM_EVENTS] = {};
    cl_event K_exe_event[NB_OF_EXE_EVENTS] = {};

    // ------------------------------------------------------------------------
    // Step 5.1: Set Kernel Arguments
    //
    //  0  x
    //  1  gamma
    //  2  beta
    //  3  Wq
    //  4  Wk
    //  5  Wv
    //  6  Wo
    //  7  bq
    //  8  bk
    //  9  bv
    // 10  bo
    // 11  scale_wq
    // 12  scale_wk
    // 13  scale_wv
    // 14  scale_wo
    // 15  HBM_Attention_Out
    // 16  out
    // ------------------------------------------------------------------------

    errCode = CL_SUCCESS;

    errCode |= clSetKernelArg(
        K_multihead_attention,
        0,
        sizeof(cl_mem),
        &GlobMem_BUF_X
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        1,
        sizeof(cl_mem),
        &GlobMem_BUF_gamma
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        2,
        sizeof(cl_mem),
        &GlobMem_BUF_beta
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        3,
        sizeof(cl_mem),
        &GlobMem_BUF_Wq
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        4,
        sizeof(cl_mem),
        &GlobMem_BUF_Wk
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        5,
        sizeof(cl_mem),
        &GlobMem_BUF_Wv
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        6,
        sizeof(cl_mem),
        &GlobMem_BUF_Wo
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        7,
        sizeof(cl_mem),
        &GlobMem_BUF_bq
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        8,
        sizeof(cl_mem),
        &GlobMem_BUF_bk
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        9,
        sizeof(cl_mem),
        &GlobMem_BUF_bv
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        10,
        sizeof(cl_mem),
        &GlobMem_BUF_bo
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        11,
        sizeof(cl_mem),
        &GlobMem_BUF_scale_wq
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        12,
        sizeof(cl_mem),
        &GlobMem_BUF_scale_wk
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        13,
        sizeof(cl_mem),
        &GlobMem_BUF_scale_wv
    );

    errCode |= clSetKernelArg(
        K_multihead_attention,
        14,
        sizeof(cl_mem),
        &GlobMem_BUF_scale_wo
    );

    // FP32 Softmax × V intermediate
    errCode |= clSetKernelArg(
        K_multihead_attention,
        15,
        sizeof(cl_mem),
        &GlobMem_BUF_Attention_Out
    );

    // Final FP32 output
    errCode |= clSetKernelArg(
        K_multihead_attention,
        16,
        sizeof(cl_mem),
        &GlobMem_BUF_Out
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to set kernel arguments, errCode = "
             << errCode << endl;

        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------
    // Step 5.2: Copy Inputs to Global Memory
    // ------------------------------------------------------------------------

    errCode = clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_X,
        0,
        0,
        nullptr,
        &Mem_op_event[0]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_gamma,
        0,
        0,
        nullptr,
        &Mem_op_event[1]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_beta,
        0,
        0,
        nullptr,
        &Mem_op_event[2]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Wq,
        0,
        0,
        nullptr,
        &Mem_op_event[3]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Wk,
        0,
        0,
        nullptr,
        &Mem_op_event[4]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Wv,
        0,
        0,
        nullptr,
        &Mem_op_event[5]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Wo,
        0,
        0,
        nullptr,
        &Mem_op_event[6]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_bq,
        0,
        0,
        nullptr,
        &Mem_op_event[7]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_bk,
        0,
        0,
        nullptr,
        &Mem_op_event[8]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_bv,
        0,
        0,
        nullptr,
        &Mem_op_event[9]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_bo,
        0,
        0,
        nullptr,
        &Mem_op_event[10]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_scale_wq,
        0,
        0,
        nullptr,
        &Mem_op_event[11]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_scale_wk,
        0,
        0,
        nullptr,
        &Mem_op_event[12]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_scale_wv,
        0,
        0,
        nullptr,
        &Mem_op_event[13]
    );

    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_scale_wo,
        0,
        0,
        nullptr,
        &Mem_op_event[14]
    );

    // Allocate intermediate buffer in device global memory without copying
    // undefined Host data.
    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Attention_Out,
        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        0,
        nullptr,
        &Mem_op_event[15]
    );

    // Allocate final output buffer in device global memory without copying
    // the uninitialized Host output array.
    errCode |= clEnqueueMigrateMemObjects(
        Command_Queue,
        1,
        &GlobMem_BUF_Out,
        CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
        0,
        nullptr,
        &Mem_op_event[16]
    );

    if (errCode != CL_SUCCESS)
    {
        cout << "HOST-Error: Failed to migrate memory objects, errCode = "
             << errCode << endl;

        return EXIT_FAILURE;
    }

    // The queue is out-of-order, so insert a barrier before launching
    // the kernel.
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

    // ------------------------------------------------------------------------
    // Step 5.3: Submit Kernel
    // ------------------------------------------------------------------------

    cout << "HOST-Info: Submitting Kernel K_multihead_attention ..."
         << endl;

    errCode = clEnqueueTask(
        Command_Queue,
        K_multihead_attention,
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

    // ------------------------------------------------------------------------
    // Step 5.4: Copy Final Output to Host
    // ------------------------------------------------------------------------

    cl_event Mem_read_event[1] = {};

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

    // ========================================================================
    // Step 6: Store Output Results
    // ========================================================================

    const char Out_File_Name[] = "Out_output.txt";

    cout << "HOST-Info: Store final output results in: "
         << Out_File_Name << endl;

    fstream Out_File;

    Out_File.open(
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
                << Out[i * EMBED_DIM + j]
                << " ";
        }

        Out_File << '\n';
    }

    Out_File.close();

    cout << "HOST-Info: Results stored successfully" << endl;

    // ========================================================================
    // Step 7: Custom Profiling
    // ========================================================================

    constexpr int Nb_Of_Kernels = NB_OF_EXE_EVENTS;
    constexpr int Nb_Of_Memory_Transfers = NB_OF_MEM_EVENTS;

    string list_of_kernel_names[Nb_Of_Kernels] =
    {
        "multihead_attention"
    };

    run_custom_profiling(
        Nb_Of_Kernels,
        Nb_Of_Memory_Transfers,
        K_exe_event,
        Mem_op_event,
        list_of_kernel_names
    );

    // ========================================================================
    // Step 8: Release OpenCL Resources
    // ========================================================================

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

    if (Mem_read_event[0] != nullptr)
    {
        clReleaseEvent(Mem_read_event[0]);
    }

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

    // Release the new device-side Attention intermediate buffer.
    clReleaseMemObject(GlobMem_BUF_Attention_Out);

    clReleaseMemObject(GlobMem_BUF_Out);

    clReleaseKernel(K_multihead_attention);
    clReleaseProgram(Program);
    clReleaseCommandQueue(Command_Queue);
    clReleaseContext(Context);

    delete[] Platform_IDs;
    delete[] Device_IDs;

    free(X);
    free(gamma);
    free(beta);

    free(Wq);
    free(Wk);
    free(Wv);
    free(Wo);

    free(bq);
    free(bk);
    free(bv);
    free(bo);

    free(scale_wq);
    free(scale_wk);
    free(scale_wv);
    free(scale_wo);

    free(Out);

    free(xclbin_Memory);

    cout << endl
         << "HOST-Info: DONE"
         << endl
         << endl;

    return EXIT_SUCCESS;
}