// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Tool Version Limit: 2022.04
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        bit 9  - interrupt (Read)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0 - enable ap_done interrupt (Read/Write)
//        bit 1 - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/COR)
//        bit 0 - ap_done (Read/COR)
//        bit 1 - ap_ready (Read/COR)
//        others - reserved
// 0x10 : Data signal of X
//        bit 31~0 - X[31:0] (Read/Write)
// 0x14 : Data signal of X
//        bit 31~0 - X[63:32] (Read/Write)
// 0x18 : reserved
// 0x1c : Data signal of Wq
//        bit 31~0 - Wq[31:0] (Read/Write)
// 0x20 : Data signal of Wq
//        bit 31~0 - Wq[63:32] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of bq
//        bit 31~0 - bq[31:0] (Read/Write)
// 0x2c : Data signal of bq
//        bit 31~0 - bq[63:32] (Read/Write)
// 0x30 : reserved
// 0x34 : Data signal of Wk
//        bit 31~0 - Wk[31:0] (Read/Write)
// 0x38 : Data signal of Wk
//        bit 31~0 - Wk[63:32] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of bk
//        bit 31~0 - bk[31:0] (Read/Write)
// 0x44 : Data signal of bk
//        bit 31~0 - bk[63:32] (Read/Write)
// 0x48 : reserved
// 0x4c : Data signal of Wv
//        bit 31~0 - Wv[31:0] (Read/Write)
// 0x50 : Data signal of Wv
//        bit 31~0 - Wv[63:32] (Read/Write)
// 0x54 : reserved
// 0x58 : Data signal of bv
//        bit 31~0 - bv[31:0] (Read/Write)
// 0x5c : Data signal of bv
//        bit 31~0 - bv[63:32] (Read/Write)
// 0x60 : reserved
// 0x64 : Data signal of Q
//        bit 31~0 - Q[31:0] (Read/Write)
// 0x68 : Data signal of Q
//        bit 31~0 - Q[63:32] (Read/Write)
// 0x6c : reserved
// 0x70 : Data signal of K
//        bit 31~0 - K[31:0] (Read/Write)
// 0x74 : Data signal of K
//        bit 31~0 - K[63:32] (Read/Write)
// 0x78 : reserved
// 0x7c : Data signal of V
//        bit 31~0 - V[31:0] (Read/Write)
// 0x80 : Data signal of V
//        bit 31~0 - V[63:32] (Read/Write)
// 0x84 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_AP_CTRL 0x00
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_GIE     0x04
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_IER     0x08
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_ISR     0x0c
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_X_DATA  0x10
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_X_DATA  64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WQ_DATA 0x1c
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_WQ_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BQ_DATA 0x28
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_BQ_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WK_DATA 0x34
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_WK_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BK_DATA 0x40
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_BK_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_WV_DATA 0x4c
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_WV_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_BV_DATA 0x58
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_BV_DATA 64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_Q_DATA  0x64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_Q_DATA  64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_K_DATA  0x70
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_K_DATA  64
#define XLINEAR_QKV_MULTIHEAD_CONTROL_ADDR_V_DATA  0x7c
#define XLINEAR_QKV_MULTIHEAD_CONTROL_BITS_V_DATA  64

