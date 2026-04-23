#include "attention.h"

extern "C"
{
    void attention_opt(
        int Q[SEQ_LEN][HEAD_DIM],
        int K[SEQ_LEN][HEAD_DIM],
        int V[SEQ_LEN][HEAD_DIM],
        int Out[SEQ_LEN][HEAD_DIM])
    {
#pragma HLS INTERFACE m_axi port = Q offset = slave bundle = gmem0 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = K offset = slave bundle = gmem1 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = V offset = slave bundle = gmem2 max_widen_bitwidth = 256
#pragma HLS INTERFACE m_axi port = Out offset = slave bundle = gmem3 max_widen_bitwidth = 256
#pragma HLS INTERFACE s_axilite port = Q bundle = control
#pragma HLS INTERFACE s_axilite port = K bundle = control
#pragma HLS INTERFACE s_axilite port = V bundle = control
#pragma HLS INTERFACE s_axilite port = Out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
        hls::stream<uint256_dt> FIFO_Q;
        hls::stream<uint256_dt> FIFO_K;
        hls::stream<uint256_dt> FIFO_V;
        hls::stream<float> FIFO_score;
        hls::stream<float> FIFO_softmax_out;
        hls::stream<uint256_dt> FIFO_Out;
#pragma HLS STREAM variable = FIFO_Q depth = 256
#pragma HLS STREAM variable = FIFO_K depth = 256
#pragma HLS STREAM variable = FIFO_V depth = 256
#pragma HLS STREAM variable = FIFO_score depth = 256
#pragma HLS STREAM variable = FIFO_softmax_out depth = 256
#pragma HLS STREAM variable = FIFO_Out depth = 256

#pragma HLS DATAFLOW

        load_Q_HBM2FIFO(Q, FIFO_Q);
        load_K_HBM2FIFO(K, FIFO_K);
        load_V_HBM2FIFO(V, FIFO_V);
        compute_score(FIFO_Q, FIFO_K, FIFO_score);
        compute_softmax(FIFO_score, FIFO_softmax_out);
        compute_attention_output(FIFO_softmax_out, FIFO_V, FIFO_Out);
        store_result_FIFO2HBM(FIFO_Out, Out);
    }
}

void load_Q_HBM2FIFO(
    int Q[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_Q)
{
#pragma HLS INLINE off
    uint256_dt *Q_256_ptr = (uint256_dt *)Q;
load_Q_loop:
    for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk = Q_256_ptr[i];
        FIFO_Q.write(temp_chunk);
    }
}

void load_K_HBM2FIFO(
    int K[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_K)
{
#pragma HLS INLINE off
    uint256_dt *K_256_ptr = (uint256_dt *)K;
load_K_loop_seq_len_times:
    for (int time = 0; time < SEQ_LEN; time++)
    {
    load_K_loop:
        for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++)
        {
#pragma HLS PIPELINE II = 1
            uint256_dt temp_chunk = K_256_ptr[i];
            FIFO_K.write(temp_chunk);
        }
    }
}

void load_Q_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_Q,
    int BRAM_Q[HEAD_DIM])
{
#pragma HLS INLINE off
load_Q_local_row:
    for (int i = 0; i < HEAD_DIM; i += 8)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk = FIFO_Q.read();
    unpack_Q_loop:
        for (int k = 0; k < 8; k++)
        {
#pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            BRAM_Q[i + k] = temp_chunk.range(end_bit, start_bit);
        }
    }
}

void load_K_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_K,
    int BRAM_K[HEAD_DIM])
{
#pragma HLS INLINE off
load_K_local_row:
    for (int i = 0; i < HEAD_DIM; i += 8)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk = FIFO_K.read();
    unpack_K_loop:
        for (int k = 0; k < 8; k++)
        {
#pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            BRAM_K[i + k] = temp_chunk.range(end_bit, start_bit);
        }
    }
}

void compute_score(
    hls::stream<uint256_dt> &FIFO_Q,
    hls::stream<uint256_dt> &FIFO_K,
    hls::stream<float> &FIFO_score)
{
#pragma HLS INLINE off
    int BRAM_Q[HEAD_DIM];
    int BRAM_K[HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable = BRAM_Q type = cyclic dim = 1 factor = 8
#pragma HLS ARRAY_PARTITION variable = BRAM_K type = cyclic dim = 1 factor = 8
compute_score_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
        load_Q_FIFO2BRAM(FIFO_Q, BRAM_Q);
        for (int j = 0; j < SEQ_LEN; j++)
        {
            int sum = 0;
            load_K_FIFO2BRAM(FIFO_K, BRAM_K);
            for (int d = 0; d < HEAD_DIM; d += 16)
            {
#pragma HLS PIPELINE II = 1
                int reg_Q[16];
                int reg_K[16];
#pragma HLS ARRAY_PARTITION variable = reg_Q complete
#pragma HLS ARRAY_PARTITION variable = reg_K complete
                for (int r = 0; r < 16; r++)
                {
#pragma HLS UNROLL factor = 16
                    reg_Q[r] = BRAM_Q[d + r];
                    reg_K[r] = BRAM_K[d + r];
                }
                for (int r = 0; r < 16; r++)
                {
#pragma HLS UNROLL factor = 16
                    sum += reg_Q[r] * reg_K[r];
                }
            }
            sum = sum >> 3;
            float score = (float)sum;
            FIFO_score.write(score);
        }
    }
}

void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11])
{
#pragma HLS INLINE off
load_score_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
#pragma HLS PIPELINE II = 1
        float temp_score = FIFO_score.read();
        BRAM_score[i] = temp_score;
    }
}

void compute_max(
    float BRAM_score[SEQ_LEN + 11],
    float &max_score)
{
#pragma HLS INLINE off
    max_score = -1e38f;
compute_max_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
#pragma HLS PIPELINE II = 1
        if (BRAM_score[i] > max_score)
        {
            max_score = BRAM_score[i];
        }
    }
    for (int i = SEQ_LEN; i < SEQ_LEN + 11; i++)
    {
#pragma HLS PIPELINE II = 1
        BRAM_score[i] = max_score;
    }
}

void compute_sum_exp(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float &sum_exp)
{
#pragma HLS INLINE off
    float sum_temp[8];
#pragma HLS ARRAY_PARTITION variable = sum_temp complete
    sum_exp = 0.0f;
init_values:
    for (int j = 0; j < 8; j++)
    {
#pragma HLS PIPELINE II = 1
        sum_temp[j] = 0.0f;
    }
compute_sum_exp_loop:
    for (int i = 0; i < SEQ_LEN + 11; i += 8)
    {
#pragma HLS PIPELINE II = 1
        for (int j = 0; j < 8; j++)
        {
#pragma HLS UNROLL factor = 8
            float exp_val = hls::exp(BRAM_score[i + j] - max_score);
            sum_temp[j] += exp_val;
        }
    }
    for (int j = 0; j < 8; j++)
    {
#pragma HLS PIPELINE II = 1
        sum_exp += sum_temp[j];
    }
    sum_exp -= 11.0f;
}

void compute_softmax_output(
    float BRAM_score[SEQ_LEN + 11],
    float max_score,
    float sum_exp,
    hls::stream<float> &FIFO_softmax_out)
{
#pragma HLS INLINE off
    float softmax_val[SEQ_LEN];
compute_softmax_output_loop:
    for (int i = 0; i < SEQ_LEN + 11; i += 8)
    {
#pragma HLS PIPELINE II = 1
        float exp_vals[8];
#pragma HLS ARRAY_PARTITION variable = exp_vals complete
        for (int j = 0; j < 8; j++)
        {
#pragma HLS UNROLL factor = 8
            if (i + j < SEQ_LEN)
            {
                exp_vals[j] = hls::exp(BRAM_score[i + j] - max_score);
                softmax_val[i + j] = exp_vals[j] / sum_exp;
            }
        }
    }
compute_softmax_output_write_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
#pragma HLS PIPELINE II = 1
        FIFO_softmax_out.write(softmax_val[i]);
    }
}

void compute_softmax_one_dim(
    hls::stream<float> &FIFO_score,
    float BRAM_score[SEQ_LEN + 11],
    hls::stream<float> &FIFO_softmax_out)
{
#pragma HLS INLINE off
    float max_score;
    float sum_exp;

    load_score_FIFO2BRAM(FIFO_score, BRAM_score);
    compute_max(BRAM_score, max_score);
    compute_sum_exp(BRAM_score, max_score, sum_exp);
    compute_softmax_output(BRAM_score, max_score, sum_exp, FIFO_softmax_out);
}

void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_softmax_out)
{
#pragma HLS INLINE off
    float BRAM_score[SEQ_LEN + 11];
#pragma HLS ARRAY_PARTITION variable = BRAM_score type = cyclic dim = 1 factor = 8
compute_softmax_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
        compute_softmax_one_dim(FIFO_score, BRAM_score, FIFO_softmax_out);
    }
}

void load_V_HBM2FIFO(
    int V[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_V)
{
#pragma HLS INLINE off
    uint256_dt *V_256_ptr = (uint256_dt *)V;
load_V_loop_seq_len_times:
    for (int time = 0; time < SEQ_LEN; time++)
    {
    load_V_loop:
        for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++)
        {
#pragma HLS PIPELINE II = 1
            uint256_dt temp_chunk = V_256_ptr[i];
            FIFO_V.write(temp_chunk);
        }
    }
}

void load_V_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_V,
    int BRAM_V[HEAD_DIM])
{
#pragma HLS INLINE off
load_V_local_row:
    for (int i = 0; i < HEAD_DIM; i += 8)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk = FIFO_V.read();
    unpack_V_loop:
        for (int k = 0; k < 8; k++)
        {
#pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            BRAM_V[i + k] = temp_chunk.range(end_bit, start_bit);
        }
    }
}

void compute_attention_output(
    hls::stream<float> &FIFO_softmax_out,
    hls::stream<uint256_dt> &FIFO_V,
    hls::stream<uint256_dt> &FIFO_Out)
{
#pragma HLS INLINE off
    int BRAM_V[HEAD_DIM];
    int BRAM_Out[HEAD_DIM];
    float BRAM_acc_temp[HEAD_DIM];
    float softmax_val;
#pragma HLS ARRAY_PARTITION variable = BRAM_V type = cyclic dim = 1 factor = 16
#pragma HLS ARRAY_PARTITION variable = BRAM_Out type = cyclic dim = 1 factor = 16
#pragma HLS ARRAY_PARTITION variable = BRAM_acc_temp type = cyclic dim = 1 factor = 16
#pragma HLS BIND_STORAGE variable = BRAM_V type = RAM_1P impl = LUTRAM
#pragma HLS BIND_STORAGE variable = BRAM_Out type = RAM_1P impl = LUTRAM
#pragma HLS BIND_STORAGE variable = BRAM_acc_temp type = RAM_2P impl = LUTRAM
init_acc_loop:
    for (int i = 0; i < HEAD_DIM; i += 16)
    {
#pragma HLS PIPELINE II = 1
        for (int r = 0; r < 16; r++)
        {
#pragma HLS UNROLL factor = 16
            BRAM_acc_temp[i + r] = 0.0f;
        }
    }
compute_attention_output_loop:
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int j = 0; j < SEQ_LEN; j++)
        {
            load_V_FIFO2BRAM(FIFO_V, BRAM_V);
            softmax_val = FIFO_softmax_out.read();
            for (int d = 0; d < HEAD_DIM; d += 16)
            {
#pragma HLS PIPELINE II = 1
                int reg_V[16];
#pragma HLS ARRAY_PARTITION variable = reg_V complete
                for (int r = 0; r < 16; r++)
                {
#pragma HLS UNROLL factor = 16
                    reg_V[r] = BRAM_V[d + r];
                }
                for (int r = 0; r < 16; r++)
                {
#pragma HLS UNROLL factor = 16
                    BRAM_acc_temp[d + r] += softmax_val * reg_V[r];
                }
            }
        }
        for (int d = 0; d < HEAD_DIM; d += 16)
        {
#pragma HLS PIPELINE II = 1
            for (int r = 0; r < 16; r++)
            {
#pragma HLS UNROLL factor = 16
                BRAM_Out[d + r] = (int)BRAM_acc_temp[d + r];
                BRAM_acc_temp[d + r] = 0.0f;
            }
        }
        store_output_BRAM2FIFO(BRAM_Out, FIFO_Out);
    }
}

void store_output_BRAM2FIFO(
    int BRAM_Out[HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_Out)
{
#pragma HLS INLINE off
store_output_loop:
    for (int i = 0; i < HEAD_DIM; i += 8)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk;
        for (int k = 0; k < 8; k++)
        {
#pragma HLS UNROLL
            temp_chunk.range((k + 1) * 32 - 1, k * 32) = BRAM_Out[i + k];
        }
        FIFO_Out.write(temp_chunk);
    }
}

void store_result_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_Out,
    int Out[SEQ_LEN][HEAD_DIM])
{
#pragma HLS INLINE off
    uint256_dt *Out_256_ptr = (uint256_dt *)Out;
store_result_loop:
    for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++)
    {
#pragma HLS PIPELINE II = 1
        uint256_dt temp_chunk = FIFO_Out.read();
        Out_256_ptr[i] = temp_chunk;
    }
}
