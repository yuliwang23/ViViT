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
	// Variables:
	//   o) Target_Platform_Vendor[] - defined as main() input argument 
	//   o) Target_Device_Name[]     - defined as main() input argument
	// 
	// After that
	//   o) Create a Context
	//   o) Create a Command Queue
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
	
	// Get the number of platforms
	// ..................................................
	errCode = clGetPlatformIDs(0, NULL, &Nb_Of_Platforms);
	if (errCode != CL_SUCCESS || Nb_Of_Platforms <= 0) {
		cout << endl << "HOST-Error: Failed to get the number of available platforms" << endl << endl;
		return EXIT_FAILURE;
	}

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Number of detected platforms : " << Nb_Of_Platforms << endl;
	#endif

	// Allocate memory to store platforms
	// ..................................................
	Platform_IDs = new cl_platform_id[Nb_Of_Platforms];
	if (!Platform_IDs) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Platform_IDs" << endl << endl;
		return EXIT_FAILURE;
	}

	// Get and store all PLATFORMS
	// ..................................................
	errCode = clGetPlatformIDs(Nb_Of_Platforms, Platform_IDs, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to get the available platforms" << endl << endl;
		return EXIT_FAILURE;
	}
 
	// Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
	// ....................................................................................
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

		// Check if the current platform matches Target_Platform_Vendor
		// .............................................................
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
	// Step 2.2:  Get All Devices for selected platform Target_Platform_ID
	//            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
	// ------------------------------------------------------------------------------------

	// Get the Number of Devices
	// ............................................................................
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

	// Search for CL_DEVICE_NAME = Target_Device_Name
	// ............................................................................
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

		// Check if the current device matches Target_Device_Name
		// ............................................................................
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
	//   o) Create a Program from a Binary File and Build it
	//   o) Create a Kernel
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
	// Step 4: Prepare Data to Run Kernel (Refactored for Quantized QKV)
	// ================================================================
	//  o) Allocate and initialize input arrays: X, gamma, beta, Wq, Wk, Wv, biases, scales
	//  o) Allocate Memory to store the results: Q, K, V arrays
	//  o) Create Buffers in Global Memory to store data
	// ================================================================
	
	// Input Feature Maps and LayerNorm parameters (Float)
	float *X         = nullptr;
	float *gamma     = nullptr;
	float *beta      = nullptr;

	// Quantized Weights (INT8)
	int8_t *Wq       = nullptr;
	int8_t *Wk       = nullptr;
	int8_t *Wv       = nullptr;

	// Floating-Point Biases and Dequantization Scales (Float)
	float *bq        = nullptr;
	float *bk        = nullptr;
	float *bv        = nullptr;
	float *scale_wq  = nullptr;
	float *scale_wk  = nullptr;
	float *scale_wv  = nullptr;

	// Outputs (Float - packed into uint256_dt by HLS, received as float)
	float *Q         = nullptr;
	float *K_out     = nullptr;
	float *V         = nullptr;

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ============================================================================
	// Step 4.1: Allocate and Load Input Arrays from Text Files
	// ============================================================================
	void *ptr = nullptr;

	// --- 1. Load Input Feature Map X [SEQ_LEN * EMBED_DIM] (Float) ---
	cout << "HOST-Info: Allocating and loading data for X ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory for X array" << endl; return EXIT_FAILURE;
	}
	X = reinterpret_cast<float*>(ptr);
	{
		ifstream infile("input_x.txt");
		if (!infile.is_open()) { cout << endl << "HOST-Error: Could not open input_x.txt" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < SEQ_LEN * EMBED_DIM; i++) {
			if (!(infile >> X[i])) {
				cout << endl << "HOST-Error: Data mismatch in input_x.txt at element " << i << endl; return EXIT_FAILURE;
			}
		}
	}
	cout << "Done (Loaded from input_x.txt)" << endl;

	// --- 2. Load LayerNorm Weights (gamma, beta) [EMBED_DIM] (Float) ---
	cout << "HOST-Info: Allocating and loading data for gamma & beta ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; gamma = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; beta  = reinterpret_cast<float*>(ptr);
	{
		ifstream f_gamma("gamma.txt"), f_beta("beta.txt");
		if (!f_gamma.is_open() || !f_beta.is_open()) { cout << endl << "HOST-Error: Could not open gamma/beta files" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < EMBED_DIM; i++) {
			if (!(f_gamma >> gamma[i]) || !(f_beta >> beta[i])) {
				cout << endl << "HOST-Error: Data mismatch in gamma/beta.txt at index " << i << endl; return EXIT_FAILURE;
			}
		}
	}
	cout << "Done" << endl;

	// --- 3. Load Quantized Weights (Wq, Wk, Wv) [EMBED_DIM * EMBED_DIM] (INT8) ---
	cout << "HOST-Info: Allocating and loading data for Quantized Weights (Wq, Wk, Wv) ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wq = reinterpret_cast<int8_t*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wk = reinterpret_cast<int8_t*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * EMBED_DIM * sizeof(int8_t))) return EXIT_FAILURE; Wv = reinterpret_cast<int8_t*>(ptr);
	{
		ifstream f_wq("Wq_transposed.txt"), f_wk("Wk_transposed.txt"), f_wv("Wv_transposed.txt");
		if (!f_wq.is_open() || !f_wk.is_open() || !f_wv.is_open()) { cout << endl << "HOST-Error: Could not open Wq/Wk/Wv text files" << endl; return EXIT_FAILURE; }
		
		int temp_q, temp_k, temp_v; // Temporary integer to prevent char-parsing issue
		for (int i = 0; i < EMBED_DIM * EMBED_DIM; i++) {
			if (!(f_wq >> temp_q) || !(f_wk >> temp_k) || !(f_wv >> temp_v)) {
				cout << endl << "HOST-Error: Data mismatch in Weight text files at element " << i << endl; return EXIT_FAILURE;
			}
			Wq[i] = static_cast<int8_t>(temp_q);
			Wk[i] = static_cast<int8_t>(temp_k);
			Wv[i] = static_cast<int8_t>(temp_v);
		}
	}
	cout << "Done (Loaded pre-transposed layouts)" << endl;

	// --- 4. Load Biases (bq, bk, bv) [EMBED_DIM] (Float) ---
	cout << "HOST-Info: Allocating and loading data for Biases (bq, bk, bv) ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bq = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bk = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; bv = reinterpret_cast<float*>(ptr);
	{
		ifstream f_bq("bq.txt"), f_bk("bk.txt"), f_bv("bv.txt");
		if (!f_bq.is_open() || !f_bk.is_open() || !f_bv.is_open()) { cout << endl << "HOST-Error: Could not open bias text files" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < EMBED_DIM; i++) {
			if (!(f_bq >> bq[i]) || !(f_bk >> bk[i]) || !(f_bv >> bv[i])) {
				cout << endl << "HOST-Error: Data mismatch in bias text files at index " << i << endl; return EXIT_FAILURE;
			}
		}
	}
	cout << "Done" << endl;

	// --- 5. Load Dequantization Scales (scale_wq, scale_wk, scale_wv) [EMBED_DIM] (Float) ---
	cout << "HOST-Info: Allocating and loading Dequantization Scales ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wq = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wk = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) return EXIT_FAILURE; scale_wv = reinterpret_cast<float*>(ptr);
	{
		ifstream f_swq("scale_wq.txt"), f_swk("scale_wk.txt"), f_swv("scale_wv.txt");
		if (!f_swq.is_open() || !f_swk.is_open() || !f_swv.is_open()) { cout << endl << "HOST-Error: Could not open scale text files" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < EMBED_DIM; i++) {
			if (!(f_swq >> scale_wq[i]) || !(f_swk >> scale_wk[i]) || !(f_swv >> scale_wv[i])) {
				cout << endl << "HOST-Error: Data mismatch in scale text files at index " << i << endl; return EXIT_FAILURE;
			}
		}
	}
	cout << "Done" << endl;

	// --- 6. Allocate Memory for Outputs Q, K, V [SEQ_LEN * EMBED_DIM] (Float) ---
	cout << "HOST-Info: Allocating memory for Outputs (Q, K, V) ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) return EXIT_FAILURE; Q = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) return EXIT_FAILURE; K_out = reinterpret_cast<float*>(ptr);
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) return EXIT_FAILURE; V = reinterpret_cast<float*>(ptr);
	cout << "Done" << endl << endl;

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory to store data (Refactored)
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
	#endif
	cl_mem	GlobMem_BUF_X, GlobMem_BUF_gamma, GlobMem_BUF_beta,
			GlobMem_BUF_Wk, GlobMem_BUF_Wv, GlobMem_BUF_Wq,
			GlobMem_BUF_bq, GlobMem_BUF_bk, GlobMem_BUF_bv,
			GlobMem_BUF_scale_wq, GlobMem_BUF_scale_wk, GlobMem_BUF_scale_wv,
			GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V;

	// --- 1. Input Feature Map Buffer (Float) ---
	GlobMem_BUF_X = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * EMBED_DIM * sizeof(float), X, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_X" << endl; return EXIT_FAILURE; }

	// --- 2. LayerNorm Buffers (Float) ---
	GlobMem_BUF_gamma = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), gamma, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_gamma" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_beta = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), beta, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_beta" << endl; return EXIT_FAILURE; }

	// --- 3. Quantized Weight Buffers (INT8 - int8_t) ---
	GlobMem_BUF_Wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wk, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_Wk" << endl; return EXIT_FAILURE; }

	// English Comment: Wv and Wq are also mapped to 8-bit integer buffers to match HLS quantized MAC operations
	GlobMem_BUF_Wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wv, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_Wv" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_Wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * EMBED_DIM * sizeof(int8_t), Wq, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_Wq" << endl; return EXIT_FAILURE; }

	// --- 4. Bias Buffers (Float) ---
	GlobMem_BUF_bq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), bq, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_bq" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_bk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), bk, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_bk" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_bv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), bv, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_bv" << endl; return EXIT_FAILURE; }

	// --- 5. Dequantization Scale Buffers (Float) ---
	GlobMem_BUF_scale_wq = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), scale_wq, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_scale_wq" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_scale_wk = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), scale_wk, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_scale_wk" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_scale_wv = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		EMBED_DIM * sizeof(float), scale_wv, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_scale_wv" << endl; return EXIT_FAILURE; }

	// --- 6. Output Buffers (Float - Flattended 1D structure) ---
	GlobMem_BUF_Q = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * EMBED_DIM * sizeof(float), Q, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_Q" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_K = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * EMBED_DIM * sizeof(float), K_out, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_K" << endl; return EXIT_FAILURE; }

	GlobMem_BUF_V = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * EMBED_DIM * sizeof(float), V, &errCode);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to allocate GlobMem_BUF_V" << endl; return EXIT_FAILURE; }

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application (Refactored)
	// ============================================================================
	int Nb_Of_Mem_Events = 15; 
	int Nb_Of_Exe_Events = 1;

	cl_event Mem_op_event[Nb_Of_Mem_Events];
	cl_event K_exe_event[Nb_Of_Exe_Events];

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 5.1: Set Kernel Arguments (Sequential Order matching Top HLS)
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Setting Kernel arguments ..." << endl;
	#endif
	errCode  = CL_SUCCESS;

	// English Comment: Connect OpenCL memory buffers to the corresponding hardware ports
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 0,  sizeof(cl_mem), &GlobMem_BUF_X);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 1,  sizeof(cl_mem), &GlobMem_BUF_gamma);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 2,  sizeof(cl_mem), &GlobMem_BUF_beta);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 3,  sizeof(cl_mem), &GlobMem_BUF_Wk);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 4,  sizeof(cl_mem), &GlobMem_BUF_Wv);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 5,  sizeof(cl_mem), &GlobMem_BUF_Wq);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 6,  sizeof(cl_mem), &GlobMem_BUF_bq);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 7,  sizeof(cl_mem), &GlobMem_BUF_bk);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 8,  sizeof(cl_mem), &GlobMem_BUF_bv);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 9,  sizeof(cl_mem), &GlobMem_BUF_scale_wq);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 10, sizeof(cl_mem), &GlobMem_BUF_scale_wk);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 11, sizeof(cl_mem), &GlobMem_BUF_scale_wv);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 12, sizeof(cl_mem), &GlobMem_BUF_Q);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 13, sizeof(cl_mem), &GlobMem_BUF_K);
	errCode |= clSetKernelArg(K_linear_qkv_multihead, 14, sizeof(cl_mem), &GlobMem_BUF_V);

	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------
	// Step 5.2: Copy Input data from Host to Global Memory
	// ------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
	#endif
	
	// Enqueue migration for 12 input data objects
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_X, 0, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate X" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_gamma, 0, 0, NULL, &Mem_op_event[1]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate gamma" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_beta, 0, 0, NULL, &Mem_op_event[2]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate beta" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wk, 0, 0, NULL, &Mem_op_event[3]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate Wk" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wv, 0, 0, NULL, &Mem_op_event[4]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate Wv" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Wq, 0, 0, NULL, &Mem_op_event[5]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate Wq" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bq, 0, 0, NULL, &Mem_op_event[6]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate bq" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bk, 0, 0, NULL, &Mem_op_event[7]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate bk" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_bv, 0, 0, NULL, &Mem_op_event[8]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate bv" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wq, 0, 0, NULL, &Mem_op_event[9]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate scale_wq" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wk, 0, 0, NULL, &Mem_op_event[10]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate scale_wk" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scale_wv, 0, 0, NULL, &Mem_op_event[11]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed to migrate scale_wv" << endl; return EXIT_FAILURE; }

	// English Comment: Virtual allocation for output buffers on FPGA device without copying host memory content
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Q, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[12]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed virtual migrate for Q" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_K, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[13]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed virtual migrate for K" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_V, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[14]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed virtual migrate for V" << endl; return EXIT_FAILURE; }

	// --------------------------------------------------------
	// Sync barrier to guarantee all data migration completes before kernel execution
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
	// Step 5.4: Submit Copy Results from Global Memory to Host
	// ---------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
	#endif

	cl_event Mem_read_event[3];

	// English Comment: Retrieve computed outputs from FPGA back to host memory, waiting for kernel execution event
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Q, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[0]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed readback for Q" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_K, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[1]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed readback for K" << endl; return EXIT_FAILURE; }

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_V, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[2]);
	if (errCode != CL_SUCCESS) { cout << "Host-Error: Failed readback for V" << endl; return EXIT_FAILURE; }

	cout << endl << "HOST_Info: Waiting for application to be completed ..." << endl;
	clFinish(Command_Queue);


	// ============================================================================
	// Step 6: Processing Output Results
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 6) Store and Check the Output Results                    " << endl;
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
				// Flattened 1D mapping index calculation for row-major matrix layout [SEQ_LEN * EMBED_DIM]
				Q_File << Q[i * EMBED_DIM + h * HEAD_DIM + d] << " ";
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
				K_File << K_out[i * EMBED_DIM + h * HEAD_DIM + d] << " ";
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
				V_File << V[i * EMBED_DIM + h * HEAD_DIM + d] << " ";
			}
			V_File << endl;
		}
	}
	V_File.close();

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
	clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

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
	clReleaseMemObject(GlobMem_BUF_gamma);
	clReleaseMemObject(GlobMem_BUF_beta);
	clReleaseMemObject(GlobMem_BUF_scale_wq);
	clReleaseMemObject(GlobMem_BUF_scale_wk);
	clReleaseMemObject(GlobMem_BUF_scale_wv);
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
	free(gamma);
	free(beta);
	free(Wq);
	free(Wk);
	free(Wv);
	free(bq);
	free(bk);
	free(bv);
	free(scale_wq);
	free(scale_wk);
	free(scale_wv);
	free(Q);
	free(K_out);
	free(V);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return EXIT_SUCCESS;
}