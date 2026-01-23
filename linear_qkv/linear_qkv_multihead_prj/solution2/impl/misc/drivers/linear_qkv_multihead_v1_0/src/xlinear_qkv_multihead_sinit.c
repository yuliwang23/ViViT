// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef __linux__

#include "xstatus.h"
#include "xparameters.h"
#include "xlinear_qkv_multihead.h"

extern XLinear_qkv_multihead_Config XLinear_qkv_multihead_ConfigTable[];

XLinear_qkv_multihead_Config *XLinear_qkv_multihead_LookupConfig(u16 DeviceId) {
	XLinear_qkv_multihead_Config *ConfigPtr = NULL;

	int Index;

	for (Index = 0; Index < XPAR_XLINEAR_QKV_MULTIHEAD_NUM_INSTANCES; Index++) {
		if (XLinear_qkv_multihead_ConfigTable[Index].DeviceId == DeviceId) {
			ConfigPtr = &XLinear_qkv_multihead_ConfigTable[Index];
			break;
		}
	}

	return ConfigPtr;
}

int XLinear_qkv_multihead_Initialize(XLinear_qkv_multihead *InstancePtr, u16 DeviceId) {
	XLinear_qkv_multihead_Config *ConfigPtr;

	Xil_AssertNonvoid(InstancePtr != NULL);

	ConfigPtr = XLinear_qkv_multihead_LookupConfig(DeviceId);
	if (ConfigPtr == NULL) {
		InstancePtr->IsReady = 0;
		return (XST_DEVICE_NOT_FOUND);
	}

	return XLinear_qkv_multihead_CfgInitialize(InstancePtr, ConfigPtr);
}

#endif

