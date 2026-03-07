#include "attention.h"
void attention_opt(
    int Q[SEQ_LEN][HEAD_DIM],
    int K[SEQ_LEN][HEAD_DIM],
    int V[SEQ_LEN][HEAD_DIM],
    int Out[SEQ_LEN][HEAD_DIM])
{
#pragma HLS INTERFACE m_axi port=Q offset=slave bundle=attention_gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=K offset=slave bundle=attention_gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=V offset=slave bundle=attention_gmem2 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=Out offset=slave bundle=attention_gmem3 max_widen_bitwidth=256
#pragma HLS INTERFACE s_axilite port=Q bundle=control
#pragma HLS INTERFACE s_axilite port=K bundle=control
#pragma HLS INTERFACE s_axilite port=V bundle=control
#pragma HLS INTERFACE s_axilite port=Out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
hls::stream<uint256_dt> FIFO_Q;
hls::stream<uint256_dt> FIFO_K;
// hls::stream<uint256_dt> FIFO_V;
hls::stream<float> FIFO_score;
#pragma HLS STREAM variable=FIFO_Q depth=256
#pragma HLS STREAM variable=FIFO_K depth=256
// #pragma HLS STREAM variable=FIFO_V depth=256
#pragma HLS STREAM variable=FIFO_score depth=256
}

void load_Q_HBM2FIFO(
    int Q[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_Q)
{
    uint256_dt *Q_256_ptr = (uint256_dt*)Q;
    load_Q_loop:
    for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++) {
    #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = Q_256_ptr[i];
        FIFO_Q.write(temp_chunk);
    }
}

void load_K_HBM2FIFO(
    int K[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_K)
{
    uint256_dt *K_256_ptr = (uint256_dt*)K;
    load_K_loop_seq_len_times:
    for(int time = 0; time < SEQ_LEN; time++) {
        load_K_loop:
        for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++) {
        #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = K_256_ptr[i];
            FIFO_K.write(temp_chunk);
        }
    }
}

void load_Q_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Q,
    int BRAM_Q[HEAD_DIM])
{
    load_Q_local_row:
    for (int i = 0; i < HEAD_DIM / 8; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = FIFO_Q.read();
        unpack_Q_loop:
        for (int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            BRAM_Q[i * 8 + k][j] = temp_chunk.range(end_bit, start_bit);
        }
    }
}

void load_K_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_K,
    int BRAM_K[HEAD_DIM])
{
    load_K_local_row:
    for (int i = 0; i < HEAD_DIM / 8; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = FIFO_K.read();
        unpack_K_loop:
        for (int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            BRAM_K[i * 8 + k][j] = temp_chunk.range(end_bit, start_bit);
        }
    }
}

void compute_score(
    hls::stream<uint256_dt> &FIFO_Q,
    hls::stream<uint256_dt> &FIFO_K,
    hls::stream<float> &FIFO_score
){
#pragma HLS INLINE off
int BRAM_Q[HEAD_DIM];
int BRAM_K[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=BRAM_Q complete dim=1
#pragma HLS ARRAY_PARTITION variable=BRAM_K complete dim=1
    compute_score_loop:
    for (int i = 0; i < SEQ_LEN; i++) {
        load_Q_FIFO2BRAM(FIFO_Q, BRAM_Q);
        for (int j = 0; j < SEQ_LEN; j++) {
            int sum = 0;
            load_K_FIFO2BRAM(FIFO_K, BRAM_K);
            for (int d = 0; d < HEAD_DIM; d+=16) {
                int reg_Q[16];
                int reg_K[16];
                #pragma HLS ARRAY_PARTITION variable=reg_Q complete
                #pragma HLS ARRAY_PARTITION variable=reg_K complete
                for(int r = 0; r < 16; r++) {
                    #pragma HLS UNROLL factor=16
                    reg_Q[r] = BRAM_Q[d + r];
                    reg_K[r] = BRAM_K[d + r];
                }
                for(int r = 0; r < 16; r++) {
                    #pragma HLS UNROLL factor=16
                    sum += reg_Q[r] * reg_K[r];
                }
            }
            sum = sum >> 3;
            float score = (float)sum;
            FIFO_score.write(score);
        }
    }
}

