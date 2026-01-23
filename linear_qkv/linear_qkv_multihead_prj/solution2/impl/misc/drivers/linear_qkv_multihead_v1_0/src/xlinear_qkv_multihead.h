// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
#ifndef XLINEAR_QKV_MULTIHEAD_H
#define XLINEAR_QKV_MULTIHEAD_H

#ifdef __cplusplus
extern "C" {
#endif

/***************************** Include Files *********************************/
#ifndef __linux__
#include "xil_types.h"
#include "xil_assert.h"
#include "xstatus.h"
#include "xil_io.h"
#else
#include <stdint.h>
#include <assert.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stddef.h>
#endif
#include "xlinear_qkv_multihead_hw.h"

/**************************** Type Definitions ******************************/
#ifdef __linux__
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
#else
typedef struct {
    u16 DeviceId;
    u64 Control_BaseAddress;
} XLinear_qkv_multihead_Config;
#endif

typedef struct {
    u64 Control_BaseAddress;
    u32 IsReady;
} XLinear_qkv_multihead;

typedef u32 word_type;

/***************** Macros (Inline Functions) Definitions *********************/
#ifndef __linux__
#define XLinear_qkv_multihead_WriteReg(BaseAddress, RegOffset, Data) \
    Xil_Out32((BaseAddress) + (RegOffset), (u32)(Data))
#define XLinear_qkv_multihead_ReadReg(BaseAddress, RegOffset) \
    Xil_In32((BaseAddress) + (RegOffset))
#else
#define XLinear_qkv_multihead_WriteReg(BaseAddress, RegOffset, Data) \
    *(volatile u32*)((BaseAddress) + (RegOffset)) = (u32)(Data)
#define XLinear_qkv_multihead_ReadReg(BaseAddress, RegOffset) \
    *(volatile u32*)((BaseAddress) + (RegOffset))

#define Xil_AssertVoid(expr)    assert(expr)
#define Xil_AssertNonvoid(expr) assert(expr)

#define XST_SUCCESS             0
#define XST_DEVICE_NOT_FOUND    2
#define XST_OPEN_DEVICE_FAILED  3
#define XIL_COMPONENT_IS_READY  1
#endif

/************************** Function Prototypes *****************************/
#ifndef __linux__
int XLinear_qkv_multihead_Initialize(XLinear_qkv_multihead *InstancePtr, u16 DeviceId);
XLinear_qkv_multihead_Config* XLinear_qkv_multihead_LookupConfig(u16 DeviceId);
int XLinear_qkv_multihead_CfgInitialize(XLinear_qkv_multihead *InstancePtr, XLinear_qkv_multihead_Config *ConfigPtr);
#else
int XLinear_qkv_multihead_Initialize(XLinear_qkv_multihead *InstancePtr, const char* InstanceName);
int XLinear_qkv_multihead_Release(XLinear_qkv_multihead *InstancePtr);
#endif

void XLinear_qkv_multihead_Start(XLinear_qkv_multihead *InstancePtr);
u32 XLinear_qkv_multihead_IsDone(XLinear_qkv_multihead *InstancePtr);
u32 XLinear_qkv_multihead_IsIdle(XLinear_qkv_multihead *InstancePtr);
u32 XLinear_qkv_multihead_IsReady(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_EnableAutoRestart(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_DisableAutoRestart(XLinear_qkv_multihead *InstancePtr);

void XLinear_qkv_multihead_Set_X(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_X(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_Wq(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_Wq(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_bq(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_bq(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_Wk(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_Wk(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_bk(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_bk(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_Wv(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_Wv(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_bv(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_bv(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_Q(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_Q(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_K(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_K(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_Set_V(XLinear_qkv_multihead *InstancePtr, u64 Data);
u64 XLinear_qkv_multihead_Get_V(XLinear_qkv_multihead *InstancePtr);

void XLinear_qkv_multihead_InterruptGlobalEnable(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_InterruptGlobalDisable(XLinear_qkv_multihead *InstancePtr);
void XLinear_qkv_multihead_InterruptEnable(XLinear_qkv_multihead *InstancePtr, u32 Mask);
void XLinear_qkv_multihead_InterruptDisable(XLinear_qkv_multihead *InstancePtr, u32 Mask);
void XLinear_qkv_multihead_InterruptClear(XLinear_qkv_multihead *InstancePtr, u32 Mask);
u32 XLinear_qkv_multihead_InterruptGetEnabled(XLinear_qkv_multihead *InstancePtr);
u32 XLinear_qkv_multihead_InterruptGetStatus(XLinear_qkv_multihead *InstancePtr);

#ifdef __cplusplus
}
#endif

#endif
