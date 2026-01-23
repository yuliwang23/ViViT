// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
/***************************** Include Files *********************************/
#include "xlinear_qkv_multihead.h"

/************************** Function Implementation *************************/
#ifndef __linux__
int XLinear_qkv_multihead_CfgInitialize(XLinear_qkv_multihead *InstancePtr, XLinear_qkv_multihead_Config *ConfigPtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(ConfigPtr != NULL);

    InstancePtr->Control_BaseAddress = ConfigPtr->Control_BaseAddress;
    InstancePtr->IsReady = XIL_COMPONENT_IS_READY;

    return XST_SUCCESS;
}
#endif

void XLinear_qkv_multihead_Start(XLinear_qkv_multihead *InstancePtr) {
    u32 Data;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL) & 0x80;
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL, Data | 0x01);
}

u32 XLinear_qkv_multihead_IsDone(XLinear_qkv_multihead *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL);
    return (Data >> 1) & 0x1;
}

u32 XLinear_qkv_multihead_IsIdle(XLinear_qkv_multihead *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL);
    return (Data >> 2) & 0x1;
}

u32 XLinear_qkv_multihead_IsReady(XLinear_qkv_multihead *InstancePtr) {
    u32 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL);
    // check ap_start to see if the pcore is ready for next input
    return !(Data & 0x1);
}

void XLinear_qkv_multihead_EnableAutoRestart(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL, 0x80);
}

void XLinear_qkv_multihead_DisableAutoRestart(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL, 0);
}

void XLinear_qkv_multihead_Set_X(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_X_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_X_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_X(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_X_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_X_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_Wq(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WQ_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WQ_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_Wq(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WQ_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WQ_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_bq(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BQ_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BQ_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_bq(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BQ_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BQ_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_Wk(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WK_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WK_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_Wk(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WK_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WK_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_bk(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BK_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BK_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_bk(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BK_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BK_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_Wv(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WV_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WV_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_Wv(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WV_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WV_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_bv(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BV_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BV_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_bv(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BV_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BV_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_Q(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_Q_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_Q_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_Q(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_Q_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_Q_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_K(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_K_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_K_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_K(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_K_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_K_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_Set_V(XLinear_qkv_multihead *InstancePtr, u64 Data) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_V_DATA, (u32)(Data));
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_V_DATA + 4, (u32)(Data >> 32));
}

u64 XLinear_qkv_multihead_Get_V(XLinear_qkv_multihead *InstancePtr) {
    u64 Data;

    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Data = XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_V_DATA);
    Data += (u64)XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_V_DATA + 4) << 32;
    return Data;
}

void XLinear_qkv_multihead_InterruptGlobalEnable(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_GIE, 1);
}

void XLinear_qkv_multihead_InterruptGlobalDisable(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_GIE, 0);
}

void XLinear_qkv_multihead_InterruptEnable(XLinear_qkv_multihead *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER);
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER, Register | Mask);
}

void XLinear_qkv_multihead_InterruptDisable(XLinear_qkv_multihead *InstancePtr, u32 Mask) {
    u32 Register;

    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    Register =  XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER);
    XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER, Register & (~Mask));
}

void XLinear_qkv_multihead_InterruptClear(XLinear_qkv_multihead *InstancePtr, u32 Mask) {
    Xil_AssertVoid(InstancePtr != NULL);
    Xil_AssertVoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    //XLinear_qkv_multihead_WriteReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_ISR, Mask);
}

u32 XLinear_qkv_multihead_InterruptGetEnabled(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    return XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER);
}

u32 XLinear_qkv_multihead_InterruptGetStatus(XLinear_qkv_multihead *InstancePtr) {
    Xil_AssertNonvoid(InstancePtr != NULL);
    Xil_AssertNonvoid(InstancePtr->IsReady == XIL_COMPONENT_IS_READY);

    // Current Interrupt Clear Behavior is Clear on Read(COR).
    return XLinear_qkv_multihead_ReadReg(InstancePtr->Control_BaseAddress, XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_ISR);
}

