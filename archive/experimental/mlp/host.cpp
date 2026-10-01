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
** HOST Code for mlp_q
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <sstream>
#include <cstdlib>
#include <algorithm>

using namespace std;

#include <CL/cl.h>

#include "help_functions.h"
#include "mlp.h"
//#include "mlp_table.h"

#define ALL_MESSAGES

// Number of 32-bit words in a 256-bit kernel chunk (uint256_dt).
// Used on the host side to lay out input/output/W1/W2 as plain arrays.
#define CHUNK_U32 8

// Derived sizes (in number of uint256_dt chunks).
#define INPUT_NUM_CHUNKS    ((SEQ_LEN  * EMBED_DIM)  / 8)    // FP32 packed: 8 values/chunk
#define OUTPUT_NUM_CHUNKS   ((SEQ_LEN  * EMBED_DIM)  / 8)
#define W1_NUM_CHUNKS       ((HIDDEN_DIM * EMBED_DIM) / 32)  // int8 packed: 32 int8/chunk
#define W2_NUM_CHUNKS       ((EMBED_DIM  * HIDDEN_DIM) / 32)

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
	cl_kernel K_mlp_q;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: mlp_q ..." << endl;
	#endif
	K_mlp_q = clCreateKernel(Program, "mlp_q", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_mlp_q" << endl << endl;
		return EXIT_FAILURE;
	}

	// ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	//   o) Allocate and initialize input arrays from txt files:
	//        input, W1_q, W2_q, b1, b2, scale_w1, scale_w2
	//   o) Allocate Memory to store the results: output array
	//   o) Create Buffers in Global Memory to store data
	// ================================================================
	// Host-side layout of kernel arguments:
	//   input/gamma/beta : uint32_t [N_chunks][8], FP32 packed into uint256_dt
	//   output           : uint32_t [N_chunks][8], FP32 packed into uint256_dt
	//   W1/W2            : int8_t   [N_chunks][32], int8 packed into uint256_dt
	//   b1,b2,sw1,sw2    : float    [N]
	uint32_t (*input_buf)[CHUNK_U32];
	uint32_t (*gamma_buf)[CHUNK_U32];
	uint32_t (*beta_buf)[CHUNK_U32];
	uint32_t (*output_buf)[CHUNK_U32];
	int8_t   (*W1_buf)[32];
	int8_t   (*W2_buf)[32];
	float    *b1_buf;
	float    *b2_buf;
	float    *scale_w1_buf;
	float    *scale_w2_buf;

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

	// Allocate input: [SEQ_LEN * EMBED_DIM] FP32, packed into 256-bit chunks
	cout << "HOST-Info: Allocating and initializing data for input ... ";
	if (posix_memalign(&ptr, 4096, INPUT_NUM_CHUNKS * 32)) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for input array" << endl << endl;
		return EXIT_FAILURE;
	}
	input_buf = reinterpret_cast<uint32_t(*)[CHUNK_U32]>(ptr);
	memset(input_buf, 0, INPUT_NUM_CHUNKS * 32);

	{
		ifstream fin("input.txt");
		if (!fin.is_open()) {
			perror("Failed to open input.txt");
			char cwd[1024]; getcwd(cwd, sizeof(cwd));
			cout << "Current working directory: " << cwd << endl;
			return EXIT_FAILURE;
		}
		for (int i = 0; i < SEQ_LEN; i++) {
			for (int j = 0; j < EMBED_DIM; j++) {
				float v;
				if (!(fin >> v)) {
					cerr << "Error: input.txt has too few values" << endl;
					return EXIT_FAILURE;
				}
				union { float f; uint32_t u; } cv;
				cv.f = v;
				input_buf[(i * EMBED_DIM + j) / 8][j % 8] = cv.u;
			}
		}
		fin.close();
	}
	cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " FP32 from input.txt)" << endl;

	// Allocate gamma/beta: default to gamma=1, beta=0 unless gamma.txt/beta.txt exist.
	cout << "HOST-Info: Allocating and initializing data for gamma/beta ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for gamma array" << endl << endl;
		return EXIT_FAILURE;
	}
	gamma_buf = reinterpret_cast<uint32_t(*)[CHUNK_U32]>(ptr);
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for beta array" << endl << endl;
		return EXIT_FAILURE;
	}
	beta_buf = reinterpret_cast<uint32_t(*)[CHUNK_U32]>(ptr);
	for (int i = 0; i < EMBED_DIM; i++) {
		union { float f; uint32_t u; } gamma_cv, beta_cv;
		gamma_cv.f = 1.0f;
		beta_cv.f  = 0.0f;
		gamma_buf[i / 8][i % 8] = gamma_cv.u;
		beta_buf [i / 8][i % 8] = beta_cv.u;
	}
	{
		ifstream fg("gamma.txt");
		if (fg.is_open()) {
			for (int i = 0; i < EMBED_DIM; i++) {
				float v;
				if (!(fg >> v)) { cerr << "Error: gamma.txt has too few values" << endl; return EXIT_FAILURE; }
				union { float f; uint32_t u; } cv;
				cv.f = v;
				gamma_buf[i / 8][i % 8] = cv.u;
			}
			fg.close();
		}
		ifstream fb("beta.txt");
		if (fb.is_open()) {
			for (int i = 0; i < EMBED_DIM; i++) {
				float v;
				if (!(fb >> v)) { cerr << "Error: beta.txt has too few values" << endl; return EXIT_FAILURE; }
				union { float f; uint32_t u; } cv;
				cv.f = v;
				beta_buf[i / 8][i % 8] = cv.u;
			}
			fb.close();
		}
	}
	cout << "Done (gamma.txt/beta.txt optional)" << endl;

	// Allocate W1: [HIDDEN_DIM * EMBED_DIM] int8, packed 32-per-chunk
	cout << "HOST-Info: Allocating and initializing data for W1 ... ";
	if (posix_memalign(&ptr, 4096, W1_NUM_CHUNKS * 32)) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for W1 array" << endl << endl;
		return EXIT_FAILURE;
	}
	W1_buf = reinterpret_cast<int8_t(*)[32]>(ptr);
	memset(W1_buf, 0, W1_NUM_CHUNKS * 32);
	{
		ifstream fw1("W1_q.txt");
		if (!fw1.is_open()) { cerr << "Error: Cannot open W1_q.txt" << endl; return EXIT_FAILURE; }
		for (int c = 0; c < W1_NUM_CHUNKS; c++) {
			for (int r = 0; r < 32; r++) {
				int v;
				if (!(fw1 >> v)) {
					cerr << "Error: W1_q.txt has too few values" << endl;
					return EXIT_FAILURE;
				}
				W1_buf[c][r] = (int8_t)v;
			}
		}
		fw1.close();
	}
	cout << "Done (" << HIDDEN_DIM << "x" << EMBED_DIM << " int8 from W1_q.txt)" << endl;

	// Allocate b1: [HIDDEN_DIM] floats
	cout << "HOST-Info: Allocating and initializing data for b1 ... ";
	if (posix_memalign(&ptr, 4096, HIDDEN_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for b1 array" << endl << endl;
		return EXIT_FAILURE;
	}
	b1_buf = reinterpret_cast<float*>(ptr);
	{
		ifstream fb1("b1.txt");
		if (!fb1.is_open()) { cerr << "Error: Cannot open b1.txt" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < HIDDEN_DIM; i++) {
			if (!(fb1 >> b1_buf[i])) { cerr << "Error: b1.txt has too few values" << endl; return EXIT_FAILURE; }
		}
		fb1.close();
	}
	cout << "Done (" << HIDDEN_DIM << " floats from b1.txt)" << endl;

	// Allocate W2: [EMBED_DIM * HIDDEN_DIM] int8, packed 32-per-chunk
	cout << "HOST-Info: Allocating and initializing data for W2 ... ";
	if (posix_memalign(&ptr, 4096, W2_NUM_CHUNKS * 32)) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for W2 array" << endl << endl;
		return EXIT_FAILURE;
	}
	W2_buf = reinterpret_cast<int8_t(*)[32]>(ptr);
	memset(W2_buf, 0, W2_NUM_CHUNKS * 32);
	{
		ifstream fw2("W2_q.txt");
		if (!fw2.is_open()) { cerr << "Error: Cannot open W2_q.txt" << endl; return EXIT_FAILURE; }
		for (int c = 0; c < W2_NUM_CHUNKS; c++) {
			for (int r = 0; r < 32; r++) {
				int v;
				if (!(fw2 >> v)) {
					cerr << "Error: W2_q.txt has too few values" << endl;
					return EXIT_FAILURE;
				}
				W2_buf[c][r] = (int8_t)v;
			}
		}
		fw2.close();
	}
	cout << "Done (" << EMBED_DIM << "x" << HIDDEN_DIM << " int8 from W2_q.txt)" << endl;

	// Allocate b2: [EMBED_DIM] floats
	cout << "HOST-Info: Allocating and initializing data for b2 ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for b2 array" << endl << endl;
		return EXIT_FAILURE;
	}
	b2_buf = reinterpret_cast<float*>(ptr);
	{
		ifstream fb2("b2.txt");
		if (!fb2.is_open()) { cerr << "Error: Cannot open b2.txt" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < EMBED_DIM; i++) {
			if (!(fb2 >> b2_buf[i])) { cerr << "Error: b2.txt has too few values" << endl; return EXIT_FAILURE; }
		}
		fb2.close();
	}
	cout << "Done (" << EMBED_DIM << " floats from b2.txt)" << endl;

	// Allocate scale_w1: [HIDDEN_DIM] floats
	cout << "HOST-Info: Allocating and initializing data for scale_w1 ... ";
	if (posix_memalign(&ptr, 4096, HIDDEN_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_w1 array" << endl << endl;
		return EXIT_FAILURE;
	}
	scale_w1_buf = reinterpret_cast<float*>(ptr);
	{
		ifstream fs1("scale_w1.txt");
		if (!fs1.is_open()) { cerr << "Error: Cannot open scale_w1.txt" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < HIDDEN_DIM; i++) {
			if (!(fs1 >> scale_w1_buf[i])) { cerr << "Error: scale_w1.txt has too few values" << endl; return EXIT_FAILURE; }
		}
		fs1.close();
	}
	cout << "Done (" << HIDDEN_DIM << " floats from scale_w1.txt)" << endl;

	// Allocate scale_w2: [EMBED_DIM] floats
	cout << "HOST-Info: Allocating and initializing data for scale_w2 ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for scale_w2 array" << endl << endl;
		return EXIT_FAILURE;
	}
	scale_w2_buf = reinterpret_cast<float*>(ptr);
	{
		ifstream fs2("scale_w2.txt");
		if (!fs2.is_open()) { cerr << "Error: Cannot open scale_w2.txt" << endl; return EXIT_FAILURE; }
		for (int i = 0; i < EMBED_DIM; i++) {
			if (!(fs2 >> scale_w2_buf[i])) { cerr << "Error: scale_w2.txt has too few values" << endl; return EXIT_FAILURE; }
		}
		fs2.close();
	}
	cout << "Done (" << EMBED_DIM << " floats from scale_w2.txt)" << endl;

	// Allocate output: [SEQ_LEN * EMBED_DIM] floats, packed 8-per-chunk
	cout << "HOST-Info: Allocating memory for output ... ";
	if (posix_memalign(&ptr, 4096, OUTPUT_NUM_CHUNKS * 32)) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for output array" << endl << endl;
		return EXIT_FAILURE;
	}
	output_buf = reinterpret_cast<uint32_t(*)[CHUNK_U32]>(ptr);
	memset(output_buf, 0, OUTPUT_NUM_CHUNKS * 32);
	cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " floats)" << endl;

	// ------------------------------------------------------------------
	// Step 4.4: Zero-weight test mode
	//   Set W1=0, W2=0, scale_w1=1, scale_w2=1
	//   Expected output[i][j] = GELU(b1[j%HIDDEN]) * 0 + b2[j]
	//                          = b2[j]   (since W2=0, matmul=0)
	//   So output should equal b2 broadcast across all SEQ_LEN tokens.
	//   If output is still all-zero but b2 is not, kernel store is broken.
	// ------------------------------------------------------------------
	#define ZERO_WEIGHT_TEST 0
	#if ZERO_WEIGHT_TEST
	cout << "HOST-Info: [TEST MODE] Setting W1=W2=0, scale=1 to verify kernel store ..." << endl;
	memset(W1_buf,  0, W1_NUM_CHUNKS * 32);
	memset(W2_buf,  0, W2_NUM_CHUNKS * 32);
	for (int i = 0; i < HIDDEN_DIM; i++) scale_w1_buf[i] = 1.0f;
	for (int i = 0; i < EMBED_DIM;  i++) scale_w2_buf[i] = 1.0f;
	cout << "HOST-Info: [TEST MODE] Expected output[0..3] = b2[0..3] = ";
	for (int i = 0; i < 4; i++) cout << b2_buf[i] << " ";
	cout << endl;
	#endif
	// Step 4.3: Sanity Check - Print first few values of each input
	// ------------------------------------------------------------------
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4.3) Sanity Check: First values of each input buffer   " << endl;
	cout << "HOST-Info: ============================================================= " << endl;

	// input/gamma/beta: print first packed FP32 chunk.
	cout << "HOST-Info: input[0..7] (FP32) = ";
	for (int k = 0; k < 8; k++) { union { uint32_t u; float f; } cv; cv.u = input_buf[0][k]; cout << cv.f << " "; }
	cout << endl;
	cout << "HOST-Info: gamma[0..7] (FP32) = ";
	for (int k = 0; k < 8; k++) { union { uint32_t u; float f; } cv; cv.u = gamma_buf[0][k]; cout << cv.f << " "; }
	cout << endl;
	cout << "HOST-Info: beta[0..7]  (FP32) = ";
	for (int k = 0; k < 8; k++) { union { uint32_t u; float f; } cv; cv.u = beta_buf[0][k]; cout << cv.f << " "; }
	cout << endl;

	// W1: first 32 int8 values (one chunk)
	cout << "HOST-Info: W1_q[0..31] (int8) = ";
	for (int r = 0; r < 32; r++) cout << (int)W1_buf[0][r] << " ";
	cout << endl;

	// W2: first 32 int8 values (one chunk)
	cout << "HOST-Info: W2_q[0..31] (int8) = ";
	for (int r = 0; r < 32; r++) cout << (int)W2_buf[0][r] << " ";
	cout << endl;

	// b1: first 4 values
	cout << "HOST-Info: b1[0..3]   (FP32) = ";
	for (int i = 0; i < 4; i++) cout << b1_buf[i] << " ";
	cout << endl;

	// b2: first 4 values
	cout << "HOST-Info: b2[0..3]   (FP32) = ";
	for (int i = 0; i < 4; i++) cout << b2_buf[i] << " ";
	cout << endl;

	// scale_w1: first 4 values
	cout << "HOST-Info: scale_w1[0..3] = ";
	for (int i = 0; i < 4; i++) cout << scale_w1_buf[i] << " ";
	cout << endl;

	// scale_w2: first 4 values
	cout << "HOST-Info: scale_w2[0..3] = ";
	for (int i = 0; i < 4; i++) cout << scale_w2_buf[i] << " ";
	cout << endl;

	cout << "HOST-Info: Total chunks - input: " << INPUT_NUM_CHUNKS
	     << "  W1: " << W1_NUM_CHUNKS
	     << "  W2: " << W2_NUM_CHUNKS << endl;
	cout << endl;
	//             o) GlobMem_BUF_input     - stores quantized input (R)
	//             o) GlobMem_BUF_gamma     - stores LayerNorm gamma (R)
	//             o) GlobMem_BUF_beta      - stores LayerNorm beta (R)
	//             o) GlobMem_BUF_W1        - stores W1 (R)
	//             o) GlobMem_BUF_W2        - stores W2 (R)
	//             o) GlobMem_BUF_b1        - stores b1 (R)
	//             o) GlobMem_BUF_b2        - stores b2 (R)
	//             o) GlobMem_BUF_sw1       - stores scale_w1 (R)
	//             o) GlobMem_BUF_sw2       - stores scale_w2 (R)
	//             o) GlobMem_BUF_output    - stores output (W)
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
	#endif
	// ------------------------------------------------------------------
	// HBM bank assignment (must match connectivity.cfg):
	//   HBM[0]: input, gamma/beta, output, b1, b2, scale_w1, scale_w2  (bundle gmem0)
	//   HBM[2]: W1                                          (bundle gmem1)
	//   HBM[4]: W2                                          (bundle gmem2)
	//
	// cl_mem_ext_ptr_t:
	//   .flags = HBM bank index
	//   .obj   = host pointer
	//   .param = 0 (unused)
	// ------------------------------------------------------------------
	// ------------------------------------------------------------------
	// Simple buffer allocation — all on HBM[0] via connectivity.cfg.
	// XRT/Vitis places buffers on HBM[0] as directed by connectivity.cfg.
	// Data transfer uses clEnqueueWriteBuffer / clEnqueueReadBuffer.
	// ------------------------------------------------------------------
	cl_mem	GlobMem_BUF_input, GlobMem_BUF_gamma, GlobMem_BUF_beta, GlobMem_BUF_W1, GlobMem_BUF_W2,
			GlobMem_BUF_b1, GlobMem_BUF_b2,
			GlobMem_BUF_sw1, GlobMem_BUF_sw2,
			GlobMem_BUF_output;

	// Allocate Global Memory for GlobMem_BUF_input
	// .......................................................
	GlobMem_BUF_input = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, INPUT_NUM_CHUNKS * 32, nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_input" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_gamma
	// .......................................................
	GlobMem_BUF_gamma = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, EMBED_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_gamma" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_beta
	// .......................................................
	GlobMem_BUF_beta = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, EMBED_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_beta" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_W1
	// .......................................................
	GlobMem_BUF_W1 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, W1_NUM_CHUNKS * 32, nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_W1" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_W2
	// .......................................................
	GlobMem_BUF_W2 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, W2_NUM_CHUNKS * 32, nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_W2" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_b1
	// .......................................................
	GlobMem_BUF_b1 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, HIDDEN_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_b1" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_b2
	// .......................................................
	GlobMem_BUF_b2 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, EMBED_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_b2" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_sw1
	// .......................................................
	GlobMem_BUF_sw1 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, HIDDEN_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_sw1" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_sw2
	// .......................................................
	GlobMem_BUF_sw2 = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, EMBED_DIM * sizeof(float), nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_sw2" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_output
	// .......................................................
	GlobMem_BUF_output = clCreateBuffer(Context,
		CL_MEM_READ_WRITE, OUTPUT_NUM_CHUNKS * 32, nullptr, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_output" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         o) Set Kernel Arguments
	// 				----------------------------------------------------
	// 				 Kernel	  		        Argument Nb		Description
	// 				----------------------------------------------------
	//  			 mlp_q	                0			GlobMem_BUF_input
	//  			 mlp_q	                1			GlobMem_BUF_gamma
	//  			 mlp_q	                2			GlobMem_BUF_beta
	//				 mlp_q	                3			GlobMem_BUF_W1
	//				 mlp_q	                4			GlobMem_BUF_W2
	//				 mlp_q	                5			GlobMem_BUF_b1
	//				 mlp_q	                6			GlobMem_BUF_b2
	//				 mlp_q	                7			GlobMem_BUF_sw1
	//				 mlp_q	                8			GlobMem_BUF_sw2
	//				 mlp_q	                9			GlobMem_BUF_output
	//				 mlp_q	               10			token_offset
	//				 mlp_q	               11			valid_seq_len
	// 				----------------------------------------------------
	//         o) Copy Input Data from Host to Global Memory
	//         o) Submit Kernels for Execution
	//         o) Copy Results from Global Memory to Host
	// ============================================================================
	int Nb_Of_Mem_Events = 9;   // input,gamma,beta,W1,W2,b1,b2,sw1,sw2
	int Nb_Of_Exe_Events = 1;

	cl_event Mem_op_event[9];
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

	int token_offset = 0;
	int valid_seq_len = SEQ_LEN;

	errCode |= clSetKernelArg(K_mlp_q, 0, sizeof(cl_mem), &GlobMem_BUF_input);
	errCode |= clSetKernelArg(K_mlp_q, 1, sizeof(cl_mem), &GlobMem_BUF_gamma);
	errCode |= clSetKernelArg(K_mlp_q, 2, sizeof(cl_mem), &GlobMem_BUF_beta);
	errCode |= clSetKernelArg(K_mlp_q, 3, sizeof(cl_mem), &GlobMem_BUF_W1);
	errCode |= clSetKernelArg(K_mlp_q, 4, sizeof(cl_mem), &GlobMem_BUF_W2);
	errCode |= clSetKernelArg(K_mlp_q, 5, sizeof(cl_mem), &GlobMem_BUF_b1);
	errCode |= clSetKernelArg(K_mlp_q, 6, sizeof(cl_mem), &GlobMem_BUF_b2);
	errCode |= clSetKernelArg(K_mlp_q, 7, sizeof(cl_mem), &GlobMem_BUF_sw1);
	errCode |= clSetKernelArg(K_mlp_q, 8, sizeof(cl_mem), &GlobMem_BUF_sw2);
	errCode |= clSetKernelArg(K_mlp_q, 9, sizeof(cl_mem), &GlobMem_BUF_output);
	errCode |= clSetKernelArg(K_mlp_q, 10, sizeof(int), &token_offset);
	errCode |= clSetKernelArg(K_mlp_q, 11, sizeof(int), &valid_seq_len);

	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-ERROR: Failed to set Kernel arguments" << endl << endl;
		return EXIT_FAILURE;
	}

	// ------------------------------------------------------
	// Step 5.2: Copy Input data from Host to Global Memory
	//           Using clEnqueueWriteBuffer (compatible with
	//           CL_MEM_EXT_PTR_XILINX / XRT HBM allocation)
	// ------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Copy Input data to Global Memory ..." << endl;
	#endif

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_input, CL_FALSE, 0,
		INPUT_NUM_CHUNKS * 32, input_buf, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write input to GlobMem_BUF_input" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_gamma, CL_FALSE, 0,
		EMBED_DIM * sizeof(float), gamma_buf, 0, NULL, &Mem_op_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write gamma to GlobMem_BUF_gamma" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_beta, CL_FALSE, 0,
		EMBED_DIM * sizeof(float), beta_buf, 0, NULL, &Mem_op_event[2]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write beta to GlobMem_BUF_beta" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_W1, CL_FALSE, 0,
		W1_NUM_CHUNKS * 32, W1_buf, 0, NULL, &Mem_op_event[3]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write W1 to GlobMem_BUF_W1" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_W2, CL_FALSE, 0,
		W2_NUM_CHUNKS * 32, W2_buf, 0, NULL, &Mem_op_event[4]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write W2 to GlobMem_BUF_W2" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_b1, CL_FALSE, 0,
		HIDDEN_DIM * sizeof(float), b1_buf, 0, NULL, &Mem_op_event[5]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write b1 to GlobMem_BUF_b1" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_b2, CL_FALSE, 0,
		EMBED_DIM * sizeof(float), b2_buf, 0, NULL, &Mem_op_event[6]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write b2 to GlobMem_BUF_b2" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_sw1, CL_FALSE, 0,
		HIDDEN_DIM * sizeof(float), scale_w1_buf, 0, NULL, &Mem_op_event[7]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write scale_w1 to GlobMem_BUF_sw1" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueWriteBuffer(Command_Queue, GlobMem_BUF_sw2, CL_FALSE, 0,
		EMBED_DIM * sizeof(float), scale_w2_buf, 0, NULL, &Mem_op_event[8]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write scale_w2 to GlobMem_BUF_sw2" << endl << endl;
		return EXIT_FAILURE;
	}

	// output buffer: kernel will write to it, no host-to-device transfer needed.
	// Nb_Of_Mem_Events tracks the 9 host-to-device transfers above.

	// --------------------------------------------------------
	errCode = clEnqueueBarrierWithWaitList(Command_Queue, 0, NULL, NULL);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to Submit BarrierWithWaitList" << endl << endl;
		return EXIT_FAILURE;
	}

	// ----------------------------------------
	// Step 5.3: Submit Kernel for Execution
	// ----------------------------------------
	cout << "HOST-Info: Submitting Kernel K_mlp_q ..." << endl;

	errCode = clEnqueueTask(Command_Queue, K_mlp_q, 0, NULL, &K_exe_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to submit K_mlp_q" << endl << endl;
		return EXIT_FAILURE;
	}

	// ---------------------------------------------------------
	// Step 5.4: Submit Copy Results from Global Memory to Host
	// ---------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
	#endif

	cl_event Mem_read_event[1];

	errCode = clEnqueueReadBuffer(Command_Queue, GlobMem_BUF_output, CL_FALSE, 0,
		OUTPUT_NUM_CHUNKS * 32, output_buf, 1, &K_exe_event[0], &Mem_read_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to read output from GlobMem_BUF_output" << endl << endl;
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
	char Output_File_Name[] = "output.txt";

	// Debug: print raw uint32 bits and float values of first chunk
	cout << "HOST-Info: output_buf[0] raw uint32 = ";
	for (int k = 0; k < 8; k++) cout << output_buf[0][k] << " ";
	cout << endl;
	cout << "HOST-Info: output_buf[0] as float   = ";
	for (int k = 0; k < 8; k++) {
		union { uint32_t u; float f; } cv; cv.u = output_buf[0][k];
		cout << cv.f << " ";
	}
	cout << endl;
	// Also check if ALL chunks are zero
	bool all_zero = true;
	for (int c = 0; c < OUTPUT_NUM_CHUNKS && all_zero; c++)
		for (int k = 0; k < 8; k++)
			if (output_buf[c][k] != 0) { all_zero = false; break; }
	cout << "HOST-Info: output buffer is " << (all_zero ? "ALL ZERO" : "non-zero") << endl;

	cout << "HOST_Info: Store output results in: " << Output_File_Name << endl;
	fstream Output_File;
	Output_File.open(Output_File_Name, ios::out);
	if (!Output_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Output_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	Output_File << scientific << setprecision(9);
	// Unpack output_buf back to FP32 and write space-separated values
	for (int c = 0; c < OUTPUT_NUM_CHUNKS; c++) {
		for (int k = 0; k < 8; k++) {
			union { uint32_t u; float f; } cv;
			cv.u = output_buf[c][k];
			if (c > 0 || k > 0) Output_File << " ";
			Output_File << cv.f;
		}
	}
	Output_File << endl;
	Output_File.close();

	// ------------------------------------------------------
	// Step 6.2: Check correctness of the output results
	// ------------------------------------------------------
	// Optional comparison against the legacy FP32 golden in golden_output.txt.
	// This now includes the expected error from LayerNorm-style input quantization.
	{
		ifstream fg("golden_layernorm_mlp_output.txt");
		if (!fg.is_open()) {
			cout << "HOST-Info: golden_layernorm_mlp_output.txt not found, skipping comparison" << endl;
		} else {
			double sum_abs = 0.0, sum_sq = 0.0;
			float  max_abs_err = 0.0f, max_ref_abs = 0.0f;
			int    worst_idx = -1;
			int    N = SEQ_LEN * EMBED_DIM;
			for (int i = 0; i < N; i++) {
				float ref;
				if (!(fg >> ref)) {
					cerr << "Error: golden_layernorm_mlp_output.txt too short" << endl;
					return EXIT_FAILURE;
				}
				int c = i / 8, k = i % 8;
				union { uint32_t u; float f; } cv;
				cv.u = output_buf[c][k];
				float got = cv.f;
				float d   = got - ref;
				float a   = fabsf(d);
				if (a > max_abs_err) { max_abs_err = a; worst_idx = i; }
				sum_abs += a;
				sum_sq  += (double)d * (double)d;
				if (fabsf(ref) > max_ref_abs) max_ref_abs = fabsf(ref);
			}
			fg.close();
			float mean_abs = (float)(sum_abs / N);
			float rmse     = (float)sqrt(sum_sq / N);
			float rel_max  = (max_ref_abs > 0) ? max_abs_err / max_ref_abs : 0.0f;
			cout << "  max |ref|        : " << max_ref_abs << endl;
			cout << "  max abs error    : " << max_abs_err << "  (at idx " << worst_idx << ")" << endl;
			cout << "  mean abs error   : " << mean_abs << endl;
			cout << "  rmse             : " << rmse << endl;
			cout << "  relative max err : " << rel_max * 100.0f << " %" << endl;
		}
	}

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

	string list_of_kernel_names[Nb_Of_Kernels] = {"mlp_q"};
	run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

	// ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
	clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

	for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
	for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
	for (int i = 0; i < 1; i++) clReleaseEvent(Mem_read_event[i]);

	clReleaseMemObject(GlobMem_BUF_input);
	clReleaseMemObject(GlobMem_BUF_gamma);
	clReleaseMemObject(GlobMem_BUF_beta);
	clReleaseMemObject(GlobMem_BUF_W1);
	clReleaseMemObject(GlobMem_BUF_W2);
	clReleaseMemObject(GlobMem_BUF_b1);
	clReleaseMemObject(GlobMem_BUF_b2);
	clReleaseMemObject(GlobMem_BUF_sw1);
	clReleaseMemObject(GlobMem_BUF_sw2);
	clReleaseMemObject(GlobMem_BUF_output);

	clReleaseKernel(K_mlp_q);

	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	delete[] Platform_IDs;
	delete[] Device_IDs;
	free(input_buf);
	free(gamma_buf);
	free(beta_buf);
	free(W1_buf);
	free(W2_buf);
	free(b1_buf);
	free(b2_buf);
	free(scale_w1_buf);
	free(scale_w2_buf);
	free(output_buf);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return EXIT_SUCCESS;
}
