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
** HOST Code for layernorm
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

using namespace std;

#include <CL/cl.h>
#include <CL/cl_ext_xilinx.h>   // Changed: added for XCL_MEM_TOPOLOGY (HBM bank binding)

#include "help_functions.h"
#include "layernorm.h"  // Changed: include layernorm header

#define ALL_MESSAGES

// Changed: HBM bank index definitions, must match connectivity.cfg sp= settings
#define HBM_BANK0  (0 | XCL_MEM_TOPOLOGY)   // gmem0: x, gamma
#define HBM_BANK2  (2 | XCL_MEM_TOPOLOGY)   // gmem1: beta, out
#define HBM_BANK4  (4 | XCL_MEM_TOPOLOGY)   // gmem2: scaler

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
	// Changed: kernel name from attention_opt to layernorm
	cl_kernel K_layernorm;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: layernorm ..." << endl;
	#endif
	K_layernorm = clCreateKernel(Program, "layernorm", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_layernorm" << endl << endl;
		return EXIT_FAILURE;
	}

	// ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	//   o) Allocate and initialize input arrays: x, gamma, beta
	//   o) Allocate Memory to store the results: out array, scaler array  // Changed: added scaler
	//   o) Create Buffers in Global Memory to store data
	// ================================================================

	// Changed: declare pointers matching layernorm signature
	// x[SEQ_LEN][EMBED_DIM]: float input
	// gamma[EMBED_DIM], beta[EMBED_DIM]: float weights
	// out[SEQ_LEN][EMBED_DIM]: int8_dt output
	// scaler[SEQ_LEN]: float output  // Changed: added scaler
	float  (*x_in)[EMBED_DIM];
	float  *gamma_in;
	float  *beta_in;
	int8_t (*out_data)[EMBED_DIM];
	float  *scaler_out;  // Changed: added scaler output

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
    // Step 4.1: Allocate and Initialize Input Arrays from Files
    // ------------------------------------------------------------------
    void *ptr = nullptr;

	// Allocate x: [SEQ_LEN][EMBED_DIM]
	cout << "HOST-Info: Allocating and loading data for x ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory for x array" << endl;
		return EXIT_FAILURE;
	}
	x_in = reinterpret_cast<float(*)[EMBED_DIM]>(ptr);
	{
		ifstream infile("input_x.txt");
		if (!infile.is_open()) {
			cout << endl << "HOST-Error: Could not open input_x.txt" << endl;
			return EXIT_FAILURE;
		}
		for (int i = 0; i < SEQ_LEN; i++)
			for (int j = 0; j < EMBED_DIM; j++)
				if (!(infile >> x_in[i][j])) {
					cout << endl << "HOST-Error: Data mismatch in input_x.txt at [" << i << "][" << j << "]" << endl;
					return EXIT_FAILURE;
				}
	}
	cout << "Done (Loaded from input_x.txt)" << endl;

	// Allocate gamma: [EMBED_DIM]
	cout << "HOST-Info: Allocating and loading data for gamma ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory for gamma array" << endl;
		return EXIT_FAILURE;
	}
	gamma_in = reinterpret_cast<float*>(ptr);
	{
		ifstream infile("input_gamma.txt");
		if (!infile.is_open()) {
			cout << endl << "HOST-Error: Could not open input_gamma.txt" << endl;
			return EXIT_FAILURE;
		}
		for (int j = 0; j < EMBED_DIM; j++)
			if (!(infile >> gamma_in[j])) {
				cout << endl << "HOST-Error: Data mismatch in input_gamma.txt at [" << j << "]" << endl;
				return EXIT_FAILURE;
			}
	}
	cout << "Done (Loaded from input_gamma.txt)" << endl;

	// Allocate beta: [EMBED_DIM]
	cout << "HOST-Info: Allocating and loading data for beta ... ";
	if (posix_memalign(&ptr, 4096, EMBED_DIM * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory for beta array" << endl;
		return EXIT_FAILURE;
	}
	beta_in = reinterpret_cast<float*>(ptr);
	{
		ifstream infile("input_beta.txt");
		if (!infile.is_open()) {
			cout << endl << "HOST-Error: Could not open input_beta.txt" << endl;
			return EXIT_FAILURE;
		}
		for (int j = 0; j < EMBED_DIM; j++)
			if (!(infile >> beta_in[j])) {
				cout << endl << "HOST-Error: Data mismatch in input_beta.txt at [" << j << "]" << endl;
				return EXIT_FAILURE;
			}
	}
	cout << "Done (Loaded from input_beta.txt)" << endl;

	// Allocate out: [SEQ_LEN][EMBED_DIM]
	cout << "HOST-Info: Allocating memory for out ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * EMBED_DIM * sizeof(int8_t))) {
		cout << endl << "HOST-Error: Out of Memory for out array" << endl;
		return EXIT_FAILURE;
	}
	out_data = reinterpret_cast<int8_t(*)[EMBED_DIM]>(ptr);
	cout << "Done (" << SEQ_LEN << "x" << EMBED_DIM << " elements)" << endl;

	// Changed: Allocate scaler: [SEQ_LEN]
	cout << "HOST-Info: Allocating memory for scaler ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * sizeof(float))) {
		cout << endl << "HOST-Error: Out of Memory for scaler array" << endl;
		return EXIT_FAILURE;
	}
	scaler_out = reinterpret_cast<float*>(ptr);
	cout << "Done (" << SEQ_LEN << " elements)" << endl;

    cout << endl;

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory to store data
	//             o) GlobMem_BUF_x      - stores x      (R)
	//             o) GlobMem_BUF_gamma  - stores gamma  (R)
	//             o) GlobMem_BUF_beta   - stores beta   (R)
	//             o) GlobMem_BUF_out    - stores out    (W)
	//             o) GlobMem_BUF_scaler - stores scaler (W)  // Changed: added scaler
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
	#endif

	// Changed: added HBM bank binding via cl_mem_ext_ptr_t for all buffers
	cl_mem_ext_ptr_t ext_x, ext_gamma, ext_beta, ext_out, ext_scaler;
	ext_x.flags      = HBM_BANK0;  ext_x.obj      = x_in;       ext_x.param      = 0;
	ext_gamma.flags  = HBM_BANK0;  ext_gamma.obj  = gamma_in;   ext_gamma.param  = 0;
	ext_beta.flags   = HBM_BANK2;  ext_beta.obj   = beta_in;    ext_beta.param   = 0;
	ext_out.flags    = HBM_BANK2;  ext_out.obj    = out_data;   ext_out.param    = 0;
	ext_scaler.flags = HBM_BANK4;  ext_scaler.obj = scaler_out; ext_scaler.param = 0;

	// Changed: four buffers for x, gamma, beta, out, scaler
	cl_mem GlobMem_BUF_x, GlobMem_BUF_gamma, GlobMem_BUF_beta, GlobMem_BUF_out;
	cl_mem GlobMem_BUF_scaler;  // Changed: added scaler buffer

	// Allocate Global Memory for GlobMem_BUF_x
	// .......................................................
	GlobMem_BUF_x = clCreateBuffer(Context,
		CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,  // Changed: added HBM flags
		SEQ_LEN * EMBED_DIM * sizeof(float), &ext_x, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_x" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_gamma
	// .......................................................
	GlobMem_BUF_gamma = clCreateBuffer(Context,
		CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,  // Changed: added HBM flags
		EMBED_DIM * sizeof(float), &ext_gamma, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_gamma" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_beta
	// .......................................................
	GlobMem_BUF_beta = clCreateBuffer(Context,
		CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,  // Changed: added HBM flags
		EMBED_DIM * sizeof(float), &ext_beta, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_beta" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_out
	// .......................................................
	GlobMem_BUF_out = clCreateBuffer(Context,
		CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,  // Changed: added HBM flags
		SEQ_LEN * EMBED_DIM * sizeof(int8_t), &ext_out, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_out" << endl << endl;
		return EXIT_FAILURE;
	}

	// Changed: Allocate Global Memory for GlobMem_BUF_scaler
	// .......................................................
	GlobMem_BUF_scaler = clCreateBuffer(Context,
		CL_MEM_WRITE_ONLY | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
		SEQ_LEN * sizeof(float), &ext_scaler, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_scaler" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         o) Set Kernel Arguments
	// 				----------------------------------------------------
	// 				 Kernel	  	Argument Nb	Description
	// 				----------------------------------------------------
	//  			 layernorm	0		GlobMem_BUF_x
	//			 	 layernorm	1		GlobMem_BUF_gamma
	//			 	 layernorm	2		GlobMem_BUF_beta
	//			 	 layernorm	3		GlobMem_BUF_out
	//			 	 layernorm	4		GlobMem_BUF_scaler  // Changed: added scaler
	// 				----------------------------------------------------
	//         o) Copy Input Data from Host to Global Memory
	//         o) Submit Kernels for Execution
	//         o) Copy Results from Global Memory to Host
	// ============================================================================
	int Nb_Of_Mem_Events = 5;  // Changed: 4 → 5 (added scaler migrate)
	int Nb_Of_Exe_Events = 1;

	cl_event Mem_op_event[5];  // Changed: 4 → 5
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

	// Changed: set arguments for layernorm kernel (x, gamma, beta, out, scaler)
	errCode |= clSetKernelArg(K_layernorm, 0, sizeof(cl_mem), &GlobMem_BUF_x);
	errCode |= clSetKernelArg(K_layernorm, 1, sizeof(cl_mem), &GlobMem_BUF_gamma);
	errCode |= clSetKernelArg(K_layernorm, 2, sizeof(cl_mem), &GlobMem_BUF_beta);
	errCode |= clSetKernelArg(K_layernorm, 3, sizeof(cl_mem), &GlobMem_BUF_out);
	errCode |= clSetKernelArg(K_layernorm, 4, sizeof(cl_mem), &GlobMem_BUF_scaler);  // Changed: added scaler

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
	
	// Changed: migrate x, gamma, beta input buffers
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_x, 0, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write x to GlobMem_BUF_x" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_gamma, 0, 0, NULL, &Mem_op_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write gamma to GlobMem_BUF_gamma" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_beta, 0, 0, NULL, &Mem_op_event[2]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write beta to GlobMem_BUF_beta" << endl << endl;
		return EXIT_FAILURE;
	}

	// Migrate output buffer without content
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_out, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[3]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_out without migrating content" << endl << endl;
		return EXIT_FAILURE;
	}

	// Changed: Migrate scaler output buffer without content
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scaler, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[4]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_scaler without migrating content" << endl << endl;
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
	// Changed: submit K_layernorm
	cout << "HOST-Info: Submitting Kernel K_layernorm ..." << endl;

	errCode = clEnqueueTask(Command_Queue, K_layernorm, 0, NULL, &K_exe_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to submit K_layernorm" << endl << endl;
		return EXIT_FAILURE;
	}

	// ---------------------------------------------------------
	// Step 5.4: Submit Copy Results from Global Memory to Host
	// ---------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
	#endif

	cl_event Mem_read_event[2];  // Changed: 1 → 2 (added scaler read-back)

	// Changed: read back GlobMem_BUF_out
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_out, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_out to out_data" << endl << endl;
		return EXIT_FAILURE;
	}

	// Changed: read back GlobMem_BUF_scaler
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_scaler, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_scaler to scaler_out" << endl << endl;
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
	// Changed: output_out.txt  - one int8 per line (verify_output_layernorm.cpp compatible)
	//          output_scaler.txt - one float per line
	char Out_File_Name[]    = "output_out.txt";
	char Scaler_File_Name[] = "output_scaler.txt";  // Changed: added scaler output file
	
	cout << "HOST_Info: Store layernorm out results in: " << Out_File_Name << endl;
	fstream Out_File;
	Out_File.open(Out_File_Name, ios::out);
	if (!Out_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Out_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int d = 0; d < EMBED_DIM; d++) {
			Out_File << static_cast<int>(out_data[i][d]) << "\n";  // Changed: one value per line
		}
	}
	Out_File.close();

	// Changed: write scaler output, one float per line
	cout << "HOST_Info: Store layernorm scaler results in: " << Scaler_File_Name << endl;
	fstream Scaler_File;
	Scaler_File.open(Scaler_File_Name, ios::out);
	if (!Scaler_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Scaler_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	Scaler_File << fixed << setprecision(9);
	for (int i = 0; i < SEQ_LEN; i++) {
		Scaler_File << scaler_out[i] << "\n";
	}
	Scaler_File.close();

	// ------------------------------------------------------
	// Step 6.2: Check correctness of the output results
	// ------------------------------------------------------
	// Note: You can implement golden reference check here if needed
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

	// Changed: kernel name string updated
	string list_of_kernel_names[Nb_Of_Kernels] = {"layernorm"};
	run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

	// ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
	clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

	for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
	for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
	for (int i = 0; i < 2; i++) clReleaseEvent(Mem_read_event[i]);  // Changed: 1 → 2

	// Changed: release layernorm buffers
	clReleaseMemObject(GlobMem_BUF_x);
	clReleaseMemObject(GlobMem_BUF_gamma);
	clReleaseMemObject(GlobMem_BUF_beta);
	clReleaseMemObject(GlobMem_BUF_out);
	clReleaseMemObject(GlobMem_BUF_scaler);  // Changed: added scaler release

	// Changed: release layernorm kernel
	clReleaseKernel(K_layernorm);

	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	free(Platform_IDs);
	free(Device_IDs);
	// Changed: free layernorm arrays
	free(x_in);
	free(gamma_in);
	free(beta_in);
	free(out_data);
	free(scaler_out);  // Changed: added scaler free

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return EXIT_SUCCESS;
}