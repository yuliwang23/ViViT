/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
** HOST Code for linear_qkv_multihead
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
#include <CL/cl_ext_xilinx.h>

using namespace std;

#include <CL/cl.h>

#include "help_functions.h"
#include "linear_qkv_multihead.h"

#define ALL_MESSAGES

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //

int main(int argc, char* argv[])
{
	cout << endl;


	// ============================================================================
	// Step 1: Check Command Line Arguments
	// ============================================================================
	//    o) argv[1] Platfrom Vendor
	//    o) argv[2] Device Name
	//    o) argv[3] XCLBIN file
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
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
	// Step 2: Detect Target Platform and Target Device in a system. 
	//         Create Context and Command Queue.
	// ============================================================================
	cout << endl;
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
	cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	cl_uint         ui;

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
	// Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
	// ------------------------------------------------------------------------------------
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
			cout << endl << "HOST-Error: Failed to get the size of the Platofrm parameter " << "CL_PLATFORM_VENDOR" << " value " << endl << endl;
			return EXIT_FAILURE;
		}

		platform_info = new char[size];
		if (!platform_info) {
			cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform Parameter " << "CL_PLATFORM_VENDOR" << endl << endl;
			return EXIT_FAILURE;
		}

		errCode = clGetPlatformInfo(Platform_IDs[ui], CL_PLATFORM_VENDOR, size, platform_info , NULL);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the " << "CL_PLATFORM_VENDOR" << " platform info" << endl << endl;
			return EXIT_FAILURE;
		}

		if (strcmp(platform_info, Target_Platform_Vendor) == 0) {
			Platform_Detected        = true;
			Target_Platform_ID       = Platform_IDs[ui];
			#ifdef ALL_MESSAGES
			cout << "HOST-Info: Selected platform            : " << Target_Platform_Vendor << endl << endl;
			#endif
		}
	}

	if (Platform_Detected == false) {
		cout << endl << "HOST-Error: Failed to get detect " << Target_Platform_Vendor << " platform" << endl << endl;
		return EXIT_FAILURE;
	}


	// ------------------------------------------------------------------------------------
	// Step 2.2: Get All Devices for selected platform Target_Platform_ID
	// ------------------------------------------------------------------------------------
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
			cout << endl << "HOST-Error: Failed to get the size of the Device parameter value " << "CL_DEVICE_NAME" << endl << endl;
			return EXIT_FAILURE;
		}
				
		device_info = new char[size];
		if (!device_info) {
			cout << endl << "HOST-Error: Out of Memory during memory allocation for Device parameter " << "CL_DEVICE_NAME" << " value " << endl << endl;
			return EXIT_FAILURE;
		}
				
		errCode = clGetDeviceInfo(Device_IDs[ui], CL_DEVICE_NAME, size, device_info, NULL);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get the " << "CL_DEVICE_NAME" << " device info" << endl << endl;
			return EXIT_FAILURE;
		}

		if (strcmp(device_info, Target_Device_Name) == 0) {
			Device_Detected        = true;
			Target_Device_ID       = Device_IDs[ui];
		}
	}

	if (Device_Detected == false) {
		cout << endl << "HOST-Error: Failed to get detect " << Target_Device_Name << " device" << endl << endl;
		return EXIT_FAILURE;
	} else {
		#ifdef ALL_MESSAGES
		cout << "HOST-Info: Selected device              : " << Target_Device_Name << endl << endl;
		#endif
	}

	// ------------------------------------------------------------------------------------
	// Step 2.3: Create Context
	// ------------------------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Context ... " << endl;
	#endif
	Context = clCreateContext(0, 1, &Target_Device_ID, NULL, NULL, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Context" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------------------------------------
	// Step 2.4: Create Command Queue (commands are executed in-order)
	// ------------------------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Command Queue ... " << endl;
	#endif
	Command_Queue = clCreateCommandQueue(Context, Target_Device_ID, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Command Queue" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 3: Create Program and Kernel
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 3.1: Load Binary File from a disk to Memory
	// ------------------------------------------------------------------
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

	// ------------------------------------------------------------
	// Step 3.2: Create a program using a Binary File
	// ------------------------------------------------------------
	size_t     Program_Length_in_Bytes;
	cl_program Program;
	cl_int     Binary_Status;
	
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Program with Binary ..." << endl;
	#endif
	Program_Length_in_Bytes = program_length;
	Program = clCreateProgramWithBinary(Context, 1, &Target_Device_ID, &Program_Length_in_Bytes, 
                                        (const unsigned char **) &xclbin_Memory, &Binary_Status, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create a Program from a Binary" << endl << endl;
		return EXIT_FAILURE;
	}

	// ----------------------------------------------------------------------
	// Step 3.3: Build (compiles and links) a program executable from binary
	// ----------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Building the Program ..." << endl;
	#endif

	errCode = clBuildProgram(Program, 1, &Target_Device_ID, NULL, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to build a Program Executable" << endl << endl;
		return EXIT_FAILURE;
	}

	// -------------------------------------------------------------
	// Step 3.4: Create Kernel
	// -------------------------------------------------------------
	cl_kernel K_linear_qkv_multihead;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: linear_qkv_multihead ..." << endl;
	#endif
	K_linear_qkv_multihead = clCreateKernel(Program, "linear_qkv_multihead", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_linear_qkv_multihead" << endl << endl;
		return EXIT_FAILURE;
	}

	// ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	//   o) Allocate and initialize input arrays: X, Wq, bq, Wk, bk, Wv, bv
	//   o) Allocate Memory to store the results: Q, K, V arrays
	//   o) Create Buffers in Global Memory to store data
	// ================================================================
	int (*X)[EMBED_DIM];
	int (*Wq)[EMBED_DIM];
	int *bq;
	int (*Wk)[EMBED_DIM];
	int *bk;
	int (*Wv)[EMBED_DIM];
	int *bv;
	int (*Q)[SEQ_LEN][HEAD_DIM];
	int (*K_out)[SEQ_LEN][HEAD_DIM];
	int (*V)[SEQ_LEN][HEAD_DIM];

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 4.1: Allocate and Initialize Input Arrays
	// ------------------------------------------------------------------
	void *ptr = nullptr;

	// Allocate X: [SEQ_LEN][EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for X ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for X array" << endl << endl;
		return EXIT_FAILURE;
	}
	X = reinterpret_cast<int(*)[EMBED_DIM]>(ptr);
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int j = 0; j < EMBED_DIM; j++) {
			X[i][j] = rand() % 101;
		}
	}
	cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " elements)" << endl;

	// Allocate Wq: [EMBED_DIM][EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for Wq ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Wq array" << endl << endl;
		return EXIT_FAILURE;
	}
	Wq = reinterpret_cast<int(*)[EMBED_DIM]>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		for (int j = 0; j < EMBED_DIM; j++) {
			Wq[i][j] = rand() % 101;
		}
	}
	cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

	// Allocate bq: [EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for bq ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for bq array" << endl << endl;
		return EXIT_FAILURE;
	}
	bq = reinterpret_cast<int*>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		bq[i] = rand() % 101;
	}
	cout << "Done (" << EMBED_DIM << " elements)" << endl;

	// Allocate Wk: [EMBED_DIM][EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for Wk ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Wk array" << endl << endl;
		return EXIT_FAILURE;
	}
	Wk = reinterpret_cast<int(*)[EMBED_DIM]>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		for (int j = 0; j < EMBED_DIM; j++) {
			Wk[i][j] = rand() % 101;
		}
	}
	cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

	// Allocate bk: [EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for bk ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for bk array" << endl << endl;
		return EXIT_FAILURE;
	}
	bk = reinterpret_cast<int*>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		bk[i] = rand() % 101;
	}
	cout << "Done (" << EMBED_DIM << " elements)" << endl;

	// Allocate Wv: [EMBED_DIM][EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for Wv ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Wv array" << endl << endl;
		return EXIT_FAILURE;
	}
	Wv = reinterpret_cast<int(*)[EMBED_DIM]>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		for (int j = 0; j < EMBED_DIM; j++) {
			Wv[i][j] = rand() % 101;
		}
	}
	cout << "Done (" << EMBED_DIM << "x" << EMBED_DIM << " elements)" << endl;

	// Allocate bv: [EMBED_DIM]
	cout << "HOST-Info: Allocating and initializing data for bv ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for bv array" << endl << endl;
		return EXIT_FAILURE;
	}
	bv = reinterpret_cast<int*>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		bv[i] = rand() % 101;
	}
	cout << "Done (" << EMBED_DIM << " elements)" << endl;

	// Allocate Q: [NUM_HEADS][SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating memory for Q    ... ";
	if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Q array" << endl << endl;
		return EXIT_FAILURE;
	}
	Q = reinterpret_cast<int(*)[SEQ_LEN][HEAD_DIM]>(ptr);
	cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	// Allocate K: [NUM_HEADS][SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating memory for K    ... ";
	if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for K array" << endl << endl;
		return EXIT_FAILURE;
	}
	K_out = reinterpret_cast<int(*)[SEQ_LEN][HEAD_DIM]>(ptr);
	cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	// Allocate V: [NUM_HEADS][SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating memory for V    ... ";
	if (posix_memalign(&ptr, 4096, NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for V array" << endl << endl;
		return EXIT_FAILURE;
	}
	V = reinterpret_cast<int(*)[SEQ_LEN][HEAD_DIM]>(ptr);
	cout << "Done (" << NUM_HEADS << "x" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	cout << endl;

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory to store data
	//             o) GlobMem_BUF_X      - stores X (R)  → HBM[0]
	//             o) GlobMem_BUF_Wq     - stores Wq (R) → HBM[1]
	//             o) GlobMem_BUF_Wk     - stores Wk (R) → HBM[2]
	//             o) GlobMem_BUF_Wv     - stores Wv (R) → HBM[3]
	//             o) GlobMem_BUF_bq     - stores bq (R) → HBM[4] (shared with Q)
	//             o) GlobMem_BUF_Q      - stores Q  (W) → HBM[4] (shared with bq)
	//             o) GlobMem_BUF_bk     - stores bk (R) → HBM[5] (shared with K)
	//             o) GlobMem_BUF_K      - stores K  (W) → HBM[5] (shared with bk)
	//             o) GlobMem_BUF_bv     - stores bv (R) → HBM[6] (shared with V)
	//             o) GlobMem_BUF_V      - stores V  (W) → HBM[6] (shared with bv)
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
	#endif

	/*cl_mem	GlobMem_BUF_X, GlobMem_BUF_Wq, GlobMem_BUF_bq,
			GlobMem_BUF_Wk, GlobMem_BUF_bk,
			GlobMem_BUF_Wv, GlobMem_BUF_bv,
			GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V;

	// ext_ptr 宣告：指定各 buffer 對應的 HBM bank
	// .obj = nullptr 因為使用 ALLOC_HOST_PTR，由 XRT 管理記憶體
	cl_mem_ext_ptr_t ext_X, ext_Wq, ext_Wk, ext_Wv;
	cl_mem_ext_ptr_t ext_bq, ext_bk, ext_bv;
	cl_mem_ext_ptr_t ext_Q,  ext_K,  ext_V;

	ext_X.flags  = XCL_MEM_TOPOLOGY | 0;  ext_X.obj  = nullptr;  ext_X.param  = 0;
	ext_Wq.flags = XCL_MEM_TOPOLOGY | 1;  ext_Wq.obj = nullptr;  ext_Wq.param = 0;
	ext_Wk.flags = XCL_MEM_TOPOLOGY | 2;  ext_Wk.obj = nullptr;  ext_Wk.param = 0;
	ext_Wv.flags = XCL_MEM_TOPOLOGY | 3;  ext_Wv.obj = nullptr;  ext_Wv.param = 0;
	ext_bq.flags = XCL_MEM_TOPOLOGY | 4;  ext_bq.obj = nullptr;  ext_bq.param = 0;
	ext_Q.flags  = XCL_MEM_TOPOLOGY | 4;  ext_Q.obj  = nullptr;  ext_Q.param  = 0;
	ext_bk.flags = XCL_MEM_TOPOLOGY | 5;  ext_bk.obj = nullptr;  ext_bk.param = 0;
	ext_K.flags  = XCL_MEM_TOPOLOGY | 5;  ext_K.obj  = nullptr;  ext_K.param  = 0;
	ext_bv.flags = XCL_MEM_TOPOLOGY | 6;  ext_bv.obj = nullptr;  ext_bv.param = 0;
	ext_V.flags  = XCL_MEM_TOPOLOGY | 6;  ext_V.obj  = nullptr;  ext_V.param  = 0;

	// Allocate Global Memory for GlobMem_BUF_X
	GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		SEQ_LEN * EMBED_DIM * sizeof(int), &ext_X, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_X" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_Wq
	GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wq, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wq" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_bq
	GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * sizeof(int), &ext_bq, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bq" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_Wk
	GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wk, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wk" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_bk
	GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * sizeof(int), &ext_bk, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bk" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_Wv
	GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wv, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Wv" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_bv
	GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX,
		EMBED_DIM * sizeof(int), &ext_bv, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_bv" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_Q
	GlobMem_BUF_Q = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_Q, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Q" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_K
	GlobMem_BUF_K = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_K, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_K" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_V
	GlobMem_BUF_V = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_V, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_V" << endl << endl;
		return EXIT_FAILURE;
	}*/
// ------------------------------------------------------------------
// Step 4.2: Create Buffers - 修正 XRT Bank Flags 與對齊問題
// ------------------------------------------------------------------

cl_mem GlobMem_BUF_X, GlobMem_BUF_Wq, GlobMem_BUF_bq,
       GlobMem_BUF_Wk, GlobMem_BUF_bk,
       GlobMem_BUF_Wv, GlobMem_BUF_bv,
       GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V;

// 修正後的 ext_ptr 宣告：必須使用 XCL_MEM_TOPOLOGY
// 格式：{ Bank_ID | XCL_MEM_TOPOLOGY, Host_Ptr, 0 }
// 使用 static_cast 轉換為 unsigned int 避開編譯器的 narrowing 檢查
cl_mem_ext_ptr_t ext_X  = { static_cast<unsigned int>(0 | XCL_MEM_TOPOLOGY), X, 0 };      // HBM[0]
cl_mem_ext_ptr_t ext_Wq = { static_cast<unsigned int>(1 | XCL_MEM_TOPOLOGY), Wq, 0 };     // HBM[1]
cl_mem_ext_ptr_t ext_Wk = { static_cast<unsigned int>(2 | XCL_MEM_TOPOLOGY), Wk, 0 };     // HBM[2]
cl_mem_ext_ptr_t ext_Wv = { static_cast<unsigned int>(3 | XCL_MEM_TOPOLOGY), Wv, 0 };     // HBM[3]

// 共用 Bank 的部分
/*cl_mem_ext_ptr_t ext_bq = { static_cast<unsigned int>(4 | XCL_MEM_TOPOLOGY), bq, 0 };     // HBM[4]
cl_mem_ext_ptr_t ext_Q  = { static_cast<unsigned int>(4 | XCL_MEM_TOPOLOGY), Q, 0 };      // HBM[4]

cl_mem_ext_ptr_t ext_bk = { static_cast<unsigned int>(5 | XCL_MEM_TOPOLOGY), bk, 0 };     // HBM[5]
cl_mem_ext_ptr_t ext_K  = { static_cast<unsigned int>(5 | XCL_MEM_TOPOLOGY), K_out, 0 };  // HBM[5]

cl_mem_ext_ptr_t ext_bv = { static_cast<unsigned int>(6 | XCL_MEM_TOPOLOGY), bv, 0 };     // HBM[6]
cl_mem_ext_ptr_t ext_V  = { static_cast<unsigned int>(6 | XCL_MEM_TOPOLOGY), V, 0 };      // HBM[6]*/
// 共用 Bank 的部分
cl_mem_ext_ptr_t ext_bq = { static_cast<unsigned int>(1 | XCL_MEM_TOPOLOGY), bq, 0 };     // HBM[4]
cl_mem_ext_ptr_t ext_Q  = { static_cast<unsigned int>(1 | XCL_MEM_TOPOLOGY), Q, 0 };      // HBM[4]

cl_mem_ext_ptr_t ext_bk = { static_cast<unsigned int>(2 | XCL_MEM_TOPOLOGY), bk, 0 };     // HBM[5]
cl_mem_ext_ptr_t ext_K  = { static_cast<unsigned int>(2 | XCL_MEM_TOPOLOGY), K_out, 0 };  // HBM[5]

cl_mem_ext_ptr_t ext_bv = { static_cast<unsigned int>(3 | XCL_MEM_TOPOLOGY), bv, 0 };     // HBM[6]
cl_mem_ext_ptr_t ext_V  = { static_cast<unsigned int>(3 | XCL_MEM_TOPOLOGY), V, 0 };      // HBM[6]

// 使用 XILINX 擴展分配 Buffer
// 注意：CL_MEM_EXT_PTR_XILINX 必須搭配正確的 ext_ptr.flags
#define HBM_MEM_FLAGS (CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR)

// 分配各個 Buffer 並檢查錯誤代碼
GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                SEQ_LEN * EMBED_DIM * sizeof(int), &ext_X, &errCode);
if (errCode != CL_SUCCESS) cout << "Error: BUF_X (" << errCode << ")" << endl;

GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wq, &errCode);
if (errCode != CL_SUCCESS) cout << "Error: BUF_Wq (" << errCode << ")" << endl;

GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wk, &errCode);
if (errCode != CL_SUCCESS) cout << "Error: BUF_Wk (" << errCode << ")" << endl;

GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * EMBED_DIM * sizeof(int), &ext_Wv, &errCode);
if (errCode != CL_SUCCESS) cout << "Error: BUF_Wv (" << errCode << ")" << endl;

// Bank 4, 5, 6
GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * sizeof(int), &ext_bq, &errCode);
GlobMem_BUF_Q  = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | HBM_MEM_FLAGS, 
                                 NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_Q, &errCode);

GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * sizeof(int), &ext_bk, &errCode);
GlobMem_BUF_K  = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | HBM_MEM_FLAGS, 
                                 NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_K, &errCode);

GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | HBM_MEM_FLAGS, 
                                 EMBED_DIM * sizeof(int), &ext_bv, &errCode);
GlobMem_BUF_V  = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | HBM_MEM_FLAGS, 
                                 NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), &ext_V, &errCode);

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         o) Set Kernel Arguments
	// 				----------------------------------------------------
	// 				 Kernel	  		        Argument Nb		Description
	// 				----------------------------------------------------
	//  			 linear_qkv_multihead	0			GlobMem_BUF_X
	//				 linear_qkv_multihead	1			GlobMem_BUF_Wq
	//				 linear_qkv_multihead	2			GlobMem_BUF_bq
	//				 linear_qkv_multihead	3			GlobMem_BUF_Wk
	//				 linear_qkv_multihead	4			GlobMem_BUF_bk
	//				 linear_qkv_multihead	5			GlobMem_BUF_Wv
	//				 linear_qkv_multihead	6			GlobMem_BUF_bv
	//				 linear_qkv_multihead	7			GlobMem_BUF_Q
	//				 linear_qkv_multihead	8			GlobMem_BUF_K
	//				 linear_qkv_multihead	9			GlobMem_BUF_V
	// 				----------------------------------------------------
	//         o) Copy Input Data from Host to Global Memory
	//         o) Submit Kernels for Execution
	//         o) Copy Results from Global Memory to Host
	// ============================================================================
	int Nb_Of_Mem_Events = 10;
	int Nb_Of_Exe_Events = 1;

	cl_event Mem_op_event[Nb_Of_Mem_Events];
	cl_event K_exe_event[Nb_Of_Exe_Events];


	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif


	// ----------------------------------------
	// Step 5.1: Set Kernel Arguments
	// ----------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Setting Kernel arguments ..." << endl;
	#endif
	errCode  = false;

	errCode |= clSetKernelArg(K_linear_qkv_multihead, 0, sizeof(cl_mem), &GlobMem_BUF_X);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 1, sizeof(cl_mem), &GlobMem_BUF_Wq);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 2, sizeof(cl_mem), &GlobMem_BUF_bq);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 3, sizeof(cl_mem), &GlobMem_BUF_Wk);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 4, sizeof(cl_mem), &GlobMem_BUF_bk);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 5, sizeof(cl_mem), &GlobMem_BUF_Wv);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 6, sizeof(cl_mem), &GlobMem_BUF_bv);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 7, sizeof(cl_mem), &GlobMem_BUF_Q);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 8, sizeof(cl_mem), &GlobMem_BUF_K);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 9, sizeof(cl_mem), &GlobMem_BUF_V);

	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------
	// Step 5.2: Copy Input data from Host to Global Memory
	//           使用 clEnqueueWriteBuffer 將 host 資料寫入 HBM
	// ------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
	#endif

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_X, CL_FALSE, 0,
		SEQ_LEN * EMBED_DIM * sizeof(int), X, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write X to GlobMem_BUF_X" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wq, CL_FALSE, 0,
		EMBED_DIM * EMBED_DIM * sizeof(int), Wq, 0, NULL, &Mem_op_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write Wq to GlobMem_BUF_Wq" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bq, CL_FALSE, 0,
		EMBED_DIM * sizeof(int), bq, 0, NULL, &Mem_op_event[2]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write bq to GlobMem_BUF_bq" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wk, CL_FALSE, 0,
		EMBED_DIM * EMBED_DIM * sizeof(int), Wk, 0, NULL, &Mem_op_event[3]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write Wk to GlobMem_BUF_Wk" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bk, CL_FALSE, 0,
		EMBED_DIM * sizeof(int), bk, 0, NULL, &Mem_op_event[4]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write bk to GlobMem_BUF_bk" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_Wv, CL_FALSE, 0,
		EMBED_DIM * EMBED_DIM * sizeof(int), Wv, 0, NULL, &Mem_op_event[5]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write Wv to GlobMem_BUF_Wv" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_bv, CL_FALSE, 0,
		EMBED_DIM * sizeof(int), bv, 0, NULL, &Mem_op_event[6]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write bv to GlobMem_BUF_bv" << endl << endl;
		return EXIT_FAILURE;
	}

	// 輸出 buffer 不需要寫入內容，用 clEnqueueMarker 佔用 event slot 以維持 Nb_Of_Mem_Events=10
	errCode = clEnqueueMarkerWithWaitList(Command_Queue, 0, NULL, &Mem_op_event[7]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to enqueue marker for Mem_op_event[7]" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMarkerWithWaitList(Command_Queue, 0, NULL, &Mem_op_event[8]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to enqueue marker for Mem_op_event[8]" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMarkerWithWaitList(Command_Queue, 0, NULL, &Mem_op_event[9]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to enqueue marker for Mem_op_event[9]" << endl << endl;
		return EXIT_FAILURE;
	}

	// --------------------------------------------------------
	errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to Submit BarrierWithWaitList" << endl << endl;
		return EXIT_FAILURE;
	}

	// ----------------------------------------
	// Step 5.3: Submit Kernel for Execution
	// ----------------------------------------
	cout << "HOST-Info: Submitting Kernel K_linear_qkv_multihead ..." << endl;

	errCode = clEnqueueTask(Command_Queue, K_linear_qkv_multihead, 0, NULL, &K_exe_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to submit K_linear_qkv_multihead" << endl << endl;
		return EXIT_FAILURE;
	}

	// ---------------------------------------------------------
	// Step 5.4: Copy Results from Global Memory to Host
	//           使用 clEnqueueReadBuffer 將結果從 HBM 讀回 host
	// ---------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
	#endif

	cl_event Mem_read_event[3];

	errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_Q, CL_FALSE, 0,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), Q,
		1, &K_exe_event[0], &Mem_read_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_Q to Q" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_K, CL_FALSE, 0,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), K_out,
		1, &K_exe_event[0], &Mem_read_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_K to K" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_V, CL_FALSE, 0,
		NUM_HEADS * SEQ_LEN * HEAD_DIM * sizeof(int), V,
		1, &K_exe_event[0], &Mem_read_event[2]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_V to V" << endl << endl;
		return EXIT_FAILURE;
	}

	cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
	clFinish(Command_Queue);


	// ============================================================================
	// Step 6: Processing Output Results
	//         o) Store output results to files
	//         o) Check correctness of the output results (optional)
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 6) Store and Check the Output Results                   " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------
	// Step 6.1: Store output Results to files
	// ------------------------------------------------------
	char Q_File_Name[] = "Q_output.txt";
	char K_File_Name[] = "K_output.txt";
	char V_File_Name[] = "V_output.txt";
	
	cout << "HOST_Info: Store Q output results in: " << Q_File_Name << endl;
	fstream Q_File;
	Q_File.open(Q_File_Name, ios::out);
	if (!Q_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Q_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	for (int h = 0; h < NUM_HEADS; h++) {
		for (int i = 0; i < SEQ_LEN; i++) {
			for (int d = 0; d < HEAD_DIM; d++) {
				Q_File << Q[h][i][d] << " ";
			}
			Q_File << endl;
		}
	}
	Q_File.close();

	cout << "HOST_Info: Store K output results in: " << K_File_Name << endl;
	fstream K_File;
	K_File.open(K_File_Name, ios::out);
	if (!K_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << K_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	for (int h = 0; h < NUM_HEADS; h++) {
		for (int i = 0; i < SEQ_LEN; i++) {
			for (int d = 0; d < HEAD_DIM; d++) {
				K_File << K_out[h][i][d] << " ";
			}
			K_File << endl;
		}
	}
	K_File.close();

	cout << "HOST_Info: Store V output results in: " << V_File_Name << endl;
	fstream V_File;
	V_File.open(V_File_Name, ios::out);
	if (!V_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << V_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	for (int h = 0; h < NUM_HEADS; h++) {
		for (int i = 0; i < SEQ_LEN; i++) {
			for (int d = 0; d < HEAD_DIM; d++) {
				V_File << V[h][i][d] << " ";
			}
			V_File << endl;
		}
	}
	V_File.close();

	// ------------------------------------------------------
	// Step 6.2: Check correctness of the output results
	// ------------------------------------------------------
	cout << "HOST-Info: Results stored successfully" << endl;

	// ============================================================================
	// Step 7: Custom Profiling
	// ============================================================================
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	
	int Nb_Of_Kernels = Nb_Of_Exe_Events;
	int Nb_Of_Memory_Transfers = Nb_Of_Mem_Events;

	string list_of_kernel_names[Nb_Of_Kernels] = {"linear_qkv_multihead"};
	run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

	// ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
	clReleaseDevice(Target_Device_ID);

	for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
	for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
	for (int i = 0; i < 3; i++) clReleaseEvent(Mem_read_event[i]);

	clReleaseMemObject(GlobMem_BUF_X);
	clReleaseMemObject(GlobMem_BUF_Wq);
	clReleaseMemObject(GlobMem_BUF_bq);
	clReleaseMemObject(GlobMem_BUF_Wk);
	clReleaseMemObject(GlobMem_BUF_bk);
	clReleaseMemObject(GlobMem_BUF_Wv);
	clReleaseMemObject(GlobMem_BUF_bv);
	clReleaseMemObject(GlobMem_BUF_Q);
	clReleaseMemObject(GlobMem_BUF_K);
	clReleaseMemObject(GlobMem_BUF_V);

	clReleaseKernel(K_linear_qkv_multihead);

	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	delete[] Platform_IDs;
	delete[] Device_IDs;
	free(X);
	free(Wq);
	free(bq);
	free(Wk);
	free(bk);
	free(Wv);
	free(bv);
	free(Q);
	free(K_out);
	free(V);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return EXIT_SUCCESS;
}