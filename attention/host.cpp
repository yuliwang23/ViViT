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
** HOST Code for attention_opt
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

#include "help_functions.h"
#include "attention_v0.h"

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
	cl_kernel K_attention_opt;

	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: attention_opt ..." << endl;
	#endif
	K_attention_opt = clCreateKernel(Program, "attention_opt", &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to create K_attention_opt" << endl << endl;
		return EXIT_FAILURE;
	}

	// ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	//   o) Allocate and initialize input arrays: Q, K, V
	//   o) Allocate Memory to store the results: Out array
	//   o) Create Buffers in Global Memory to store data
	// ================================================================
	int (*Q)[HEAD_DIM];
	int (*K)[HEAD_DIM];
	int (*V)[HEAD_DIM];
	int (*Out)[HEAD_DIM];

	#ifdef ALL_MESSAGES
	cout << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 4.1: Allocate and Initialize Input Arrays
	// ------------------------------------------------------------------
	/*void *ptr = nullptr;

	// Allocate Q: [SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating and initializing data for Q ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Q array" << endl << endl;
		return EXIT_FAILURE;
	}
	Q = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int j = 0; j < HEAD_DIM; j++) {
			Q[i][j] = rand() % 10 - 5;
		}
	}
	cout << "Done (" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	// Allocate K: [SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating and initializing data for K ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for K array" << endl << endl;
		return EXIT_FAILURE;
	}
	K = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int j = 0; j < HEAD_DIM; j++) {
			K[i][j] = rand() % 10 - 5;
		}
	}
	cout << "Done (" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	// Allocate V: [SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating and initializing data for V ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for V array" << endl << endl;
		return EXIT_FAILURE;
	}
	V = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int j = 0; j < HEAD_DIM; j++) {
			V[i][j] = rand() % 10 - 5;
		}
	}
	cout << "Done (" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	// Allocate Out: [SEQ_LEN][HEAD_DIM]
	cout << "HOST-Info: Allocating memory for Out  ... ";
	if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
		cout << endl << "HOST-Error: Out of Memory during memory allocation for Out array" << endl << endl;
		return EXIT_FAILURE;
	}
	Out = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
	cout << "Done (" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

	cout << endl;*/
	// ------------------------------------------------------------------
    // Step 4.1: Allocate and Initialize Input Arrays from Files
    // ------------------------------------------------------------------
    void *ptr = nullptr;
    auto load_from_file = [&](const char* filename, int (*target)[HEAD_DIM]) {
        ifstream infile(filename);
        if (!infile.is_open()) {
            cout << endl << "HOST-Error: Could not open " << filename << endl;
            return false;
        }
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                if (!(infile >> target[i][j])) {
                    cout << endl << "HOST-Error: Data mismatch in " << filename << " at [" << i << "][" << j << "]" << endl;
                    return false;
                }
            }
        }
        infile.close();
        return true;
    };

    // Allocate Q: [SEQ_LEN][HEAD_DIM]
    cout << "HOST-Info: Allocating and loading data for Q ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for Q array" << endl;
        return EXIT_FAILURE;
    }
    Q = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
    if (!load_from_file("input_q.txt", Q)) return EXIT_FAILURE;
    cout << "Done (Loaded from input_q.txt)" << endl;

    // Allocate K: [SEQ_LEN][HEAD_DIM]
    cout << "HOST-Info: Allocating and loading data for K ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for K array" << endl;
        return EXIT_FAILURE;
    }
    K = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
    if (!load_from_file("input_k.txt", K)) return EXIT_FAILURE;
    cout << "Done (Loaded from input_k.txt)" << endl;

    // Allocate V: [SEQ_LEN][HEAD_DIM]
    cout << "HOST-Info: Allocating and loading data for V ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for V array" << endl;
        return EXIT_FAILURE;
    }
    V = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
    if (!load_from_file("input_v.txt", V)) return EXIT_FAILURE;
    cout << "Done (Loaded from input_v.txt)" << endl;

    // Allocate Out: [SEQ_LEN][HEAD_DIM]
    cout << "HOST-Info: Allocating memory for Out ... ";
    if (posix_memalign(&ptr, 4096, SEQ_LEN * HEAD_DIM * sizeof(int))) {
        cout << endl << "HOST-Error: Out of Memory for Out array" << endl;
        return EXIT_FAILURE;
    }
    Out = reinterpret_cast<int(*)[HEAD_DIM]>(ptr);
    cout << "Done (" << SEQ_LEN << "x" << HEAD_DIM << " elements)" << endl;

    cout << endl;

	// ------------------------------------------------------------------
	// Step 4.2: Create Buffers in Global Memory to store data
	//             o) GlobMem_BUF_Q   - stores Q (R)
	//             o) GlobMem_BUF_K   - stores K (R)
	//             o) GlobMem_BUF_V   - stores V (R)
	//             o) GlobMem_BUF_Out - stores Out (W)
	// ------------------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: Allocating buffers in Global Memory to store Input and Output Data ..." << endl;
	#endif
	cl_mem GlobMem_BUF_Q, GlobMem_BUF_K, GlobMem_BUF_V, GlobMem_BUF_Out;

	// Allocate Global Memory for GlobMem_BUF_Q
	// .......................................................
	GlobMem_BUF_Q = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * HEAD_DIM * sizeof(int), Q, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Q" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_K
	// .......................................................
	GlobMem_BUF_K = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * HEAD_DIM * sizeof(int), K, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_K" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_V
	// .......................................................
	GlobMem_BUF_V = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * HEAD_DIM * sizeof(int), V, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_V" << endl << endl;
		return EXIT_FAILURE;
	}

	// Allocate Global Memory for GlobMem_BUF_Out
	// .......................................................
	GlobMem_BUF_Out = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, 
		SEQ_LEN * HEAD_DIM * sizeof(int), Out, &errCode);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to allocate Global Memory for GlobMem_BUF_Out" << endl << endl;
		return EXIT_FAILURE;
	}

	// ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         o) Set Kernel Arguments
	// 				----------------------------------------------------
	// 				 Kernel	  		Argument Nb		Description
	// 				----------------------------------------------------
	//  			 attention_opt	0			GlobMem_BUF_Q
	//				 attention_opt	1			GlobMem_BUF_K
	//				 attention_opt	2			GlobMem_BUF_V
	//				 attention_opt	3			GlobMem_BUF_Out
	// 				----------------------------------------------------
	//         o) Copy Input Data from Host to Global Memory
	//         o) Submit Kernels for Execution
	//         o) Copy Results from Global Memory to Host
	// ============================================================================
	int Nb_Of_Mem_Events = 4;
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

	errCode |= clSetKernelArg(K_attention_opt, 0, sizeof(cl_mem), &GlobMem_BUF_Q);
	errCode |= clSetKernelArg(K_attention_opt, 1, sizeof(cl_mem), &GlobMem_BUF_K);
	errCode |= clSetKernelArg(K_attention_opt, 2, sizeof(cl_mem), &GlobMem_BUF_V);
	errCode |= clSetKernelArg(K_attention_opt, 3, sizeof(cl_mem), &GlobMem_BUF_Out);

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
	
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Q, 0, 0, NULL, &Mem_op_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write Q to GlobMem_BUF_Q" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_K, 0, 0, NULL, &Mem_op_event[1]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write K to GlobMem_BUF_K" << endl << endl;
		return EXIT_FAILURE;
	}

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_V, 0, 0, NULL, &Mem_op_event[2]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to write V to GlobMem_BUF_V" << endl << endl;
		return EXIT_FAILURE;
	}

	// Migrate output buffer without content
	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, NULL, &Mem_op_event[3]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed Migrate GlobMem_BUF_Out without migrating content" << endl << endl;
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
	cout << "HOST-Info: Submitting Kernel K_attention_opt ..." << endl;

	errCode = clEnqueueTask(Command_Queue, K_attention_opt, 0, NULL, &K_exe_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "HOST-Error: Failed to submit K_attention_opt" << endl << endl;
		return EXIT_FAILURE;
	}

	// ---------------------------------------------------------
	// Step 5.4: Submit Copy Results from Global Memory to Host
	// ---------------------------------------------------------
	#ifdef ALL_MESSAGES
	cout << "HOST_Info: Submitting Copy Results data from Global Memory to Host ..." << endl;
	#endif

	cl_event Mem_read_event[1];

	errCode = clEnqueueMigrateMemObjects(Command_Queue, 1, &GlobMem_BUF_Out, CL_MIGRATE_MEM_OBJECT_HOST, 1, &K_exe_event[0], &Mem_read_event[0]);
	if (errCode != CL_SUCCESS) {
		cout << endl << "Host-Error: Failed to submit Copy Results from GlobMem_BUF_Out to Out" << endl << endl;
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
	char Out_File_Name[] = "Out_output.txt";
	
	cout << "HOST_Info: Store Out output results in: " << Out_File_Name << endl;
	fstream Out_File;
	Out_File.open(Out_File_Name, ios::out);
	if (!Out_File.is_open()) {
		cout << endl << "HOST-Error: Failed to open the " << Out_File_Name << " file for write" << endl << endl;
		return EXIT_FAILURE;
	}
	for (int i = 0; i < SEQ_LEN; i++) {
		for (int d = 0; d < HEAD_DIM; d++) {
			Out_File << Out[i][d] << " ";
		}
		Out_File << endl;
	}
	Out_File.close();

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

	string list_of_kernel_names[Nb_Of_Kernels] = {"attention_opt"};
	run_custom_profiling(Nb_Of_Kernels, Nb_Of_Memory_Transfers, K_exe_event, Mem_op_event, list_of_kernel_names);

	// ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
	clReleaseDevice(Target_Device_ID); // Only available in OpenCL >= 1.2

	for (int i = 0; i < Nb_Of_Mem_Events; i++) clReleaseEvent(Mem_op_event[i]);
	for (int i = 0; i < Nb_Of_Exe_Events; i++) clReleaseEvent(K_exe_event[i]);
	for (int i = 0; i < 1; i++) clReleaseEvent(Mem_read_event[i]);

	clReleaseMemObject(GlobMem_BUF_Q);
	clReleaseMemObject(GlobMem_BUF_K);
	clReleaseMemObject(GlobMem_BUF_V);
	clReleaseMemObject(GlobMem_BUF_Out);

	clReleaseKernel(K_attention_opt);

	clReleaseProgram(Program);
	clReleaseCommandQueue(Command_Queue);
	clReleaseContext(Context);

	free(Platform_IDs);
	free(Device_IDs);
	free(Q);
	free(K);
	free(V);
	free(Out);

	cout << endl << "HOST-Info: DONE" << endl << endl;

	return EXIT_SUCCESS;
}