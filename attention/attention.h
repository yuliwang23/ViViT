#ifndef ATTENTION_H
#define ATTENTION_H
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <cmath>
typedef ap_uint<256> uint256_dt;
constexpr int SEQ_LEN = 14 * 14 + 1; // 14*14*16
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr float scale = 0.125; // for Base param. (FE_B_HEAD_DIM = 64)
constexpr int VEC_MAX = SEQ_LEN;

void attention_opt(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int Out[SEQ_LEN][HEAD_DIM]);

void softmax_software(float in[VEC_MAX],
            float out[VEC_MAX],
            int length);

void load_Q_HBM2FIFO(
    int Q[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_Q);

void load_K_HBM2FIFO(
    int K[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_K);

void load_V_HBM2FIFO(
    int V[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_V);

void load_Q_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Q,
    int BRAM_Q[HEAD_DIM]);

void load_K_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_K,
    int BRAM_K[HEAD_DIM]);

void load_V_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_V,
    int BRAM_V[HEAD_DIM]);

void compute_score(
    hls::stream<uint256_dt> &FIFO_Q,
    hls::stream<uint256_dt> &FIFO_K,
    hls::stream<float> &FIFO_score);

void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11]);

void compute_max(
    float BRAM_score[SEQ_LEN + 11],
    float &max_score);

void compute_sum_exp(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float &sum_exp);

void compute_softmax_output(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float sum_exp,
    hls::stream<float> &FIFO_softmax_out);

void compute_softmax_one_dim(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11],
    hls::stream<float> &FIFO_softmax_out);

void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_softmax_out);

void compute_attention_output(
    hls::stream<float> &FIFO_softmax_out,
    hls::stream<uint256_dt> &FIFO_V,
    hls::stream<uint256_dt> &FIFO_Out);

void store_output_BRAM2FIFO(
    int BRAM_Out[HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_Out);

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    int Out[SEQ_LEN][HEAD_DIM]);

#endif // ATTENTION_H