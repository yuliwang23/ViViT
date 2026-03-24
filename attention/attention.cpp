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
hls::stream<uint256_dt> FIFO_V;
hls::stream<float> FIFO_score;
hls::stream<float> FIFO_P;
hls::stream<float> FIFO_l;
hls::stream<float> FIFO_m;
hls::stream<uint256_dt> FIFO_O_in;
hls::stream<uint256_dt> FIFO_O_out;
#pragma HLS STREAM variable=FIFO_Q depth=256
#pragma HLS STREAM variable=FIFO_K depth=256
#pragma HLS STREAM variable=FIFO_V depth=256
#pragma HLS STREAM variable=FIFO_score depth=256
#pragma HLS STREAM variable=FIFO_P depth=256
#pragma HLS STREAM variable=FIFO_l depth=256
#pragma HLS STREAM variable=FIFO_m depth=256
#pragma HLS STREAM variable=FIFO_O_in depth=256
#pragma HLS STREAM variable=FIFO_O_out depth=256
int BRAM_Q[tile_size][HEAD_DIM];
int BRAM_K[tile_size][HEAD_DIM];
int BRAM_V[tile_size][HEAD_DIM];
float BRAM_O[tile_size][HEAD_DIM];
#pragma HLS ARRAY_PARTITION variable=BRAM_Q type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_K type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_V type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_V type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_O type=cyclic dim=2 factor=8
#pragma HLS BIND_STORAGE variable=BRAM_Q type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_K type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_V type=RAM_2P impl=lutram
#pragma HLS BIND_STORAGE variable=BRAM_O type=RAM_2P impl=bram
float BRAM_m[SEQ_LEN + 11];
float BRAM_l[SEQ_LEN + 11];

#pragma HLS DATAFLOW
//data_in
{
    for(int i = 0; i < 13; i++){
        load_QKV_HBM2FIFO(Q, FIFO_Q);
        load_QKV_HBM2FIFO(Out, FIFO_O_in);
    }
}
load_QKV_HBM2FIFO(K, FIFO_K);
load_QKV_HBM2FIFO(V, FIFO_V);
//compute
{
    for(int i = 0; i < 13; i++){
        load_QKV_FIFO2BRAM(FIFO_K, BRAM_K);
        load_QKV_FIFO2BRAM(FIFO_V, BRAM_V);
        for(int j = 0; j < 13; j++) {
            load_QKV_FIFO2BRAM(FIFO_Q, BRAM_Q);
            load_O_FIFO2BRAM(FIFO_O_in,BRAM_O);
            {
                #pragma HLS DATAFLOW
                compute_score(BRAM_Q, BRAM_K, FIFO_score);
                compute_softmax(FIFO_score, FIFO_P, FIFO_l, FIFO_m,(j == 12));
                compute_output(FIFO_m, FIFO_l, FIFO_P, FIFO_O_out, BRAM_m, BRAM_l,BRAM_O, BRAM_V,j);
            }
        }
    }
}
//data_out
{
    for(int i = 0;i < 13; i++){
        store_O_FIFO2HBM(FIFO_O_out,Out);
    }
}
}

void load_QKV_HBM2FIFO( //Q will be call with 13 times
    int QKV[SEQ_LEN][HEAD_DIM],
    hls::stream<uint256_dt> &FIFO_QKV)
{
    #pragma HLS INLINE off
    uint256_dt *QKV_256_ptr = (uint256_dt*)QKV;
    load_QKV_loop:
    for (int i = 0; i < (SEQ_LEN * HEAD_DIM) / 8; i++) {
    #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = QKV_256_ptr[i];
        FIFO_QKV.write(temp_chunk);
    }
    load_padding_loop:
    for(int padding = 0; padding < 11; padding++){
    #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = 0;
        FIFO_QKV.write(temp_chunk);
    }
}

void load_QKV_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_QKV,
    int BRAM_QKV[tile_size][HEAD_DIM])
{
    #pragma HLS INLINE off
    load_QKV_local_row:
    for (int i = 0; i < tile_size; i+=1) {
        for(int j = 0; j < HEAD_DIM; j+=8) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_QKV.read();
            unpack_QKV_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 32;
                int end_bit = start_bit + 31;
                BRAM_QKV[i][j + k] = temp_chunk.range(end_bit, start_bit);//only have a port to write
            }
        }
    }
}

void load_O_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_O,
    float BRAM_O[tile_size][HEAD_DIM])
{
    #pragma HLS INLINE off
    load_O_local_row:
    for (int i = 0; i < tile_size; i+=1) {
        for(int j = 0; j < HEAD_DIM; j+=8) {
            #pragma HLS PIPELINE II=1
            uint256_dt temp_chunk = FIFO_O.read();
            unpack_O_loop:
            for (int k = 0; k < 8; k++) {
                #pragma HLS UNROLL
                int start_bit = k * 32;
                int end_bit = start_bit + 31;
                BRAM_O[i][j + k] = (float)temp_chunk.range(end_bit, start_bit);
            }
        }
    }
}

void compute_score(
    int BRAM_Q[tile_size][HEAD_DIM],
    int BRAM_K[tile_size][HEAD_DIM],
    hls::stream<float> &FIFO_score
){
    #pragma HLS INLINE off
    for(int i = 0; i < tile_size; i++) {
        for (int j = 0; j < tile_size; j++) {
            int sum = 0;
            for (int d = 0; d < HEAD_DIM; d+=8) {
                for(int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL factor=8
                    sum += BRAM_Q[i][d + k] * BRAM_K[j][d + k];
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
    float BRAM_score[tile_size][tile_size])
{
    #pragma HLS INLINE off
    load_score_loop:
    for (int i = 0; i < tile_size; i++) {
        for(int j = 0; j < tile_size; j++) {
            #pragma HLS PIPELINE II=1
            float temp_score = FIFO_score.read();
            BRAM_score[i][j] = temp_score;
        }
    }
}

void compute_softmax(
    hls::stream<float> &FIFO_score,
    hls::stream<float> &FIFO_P,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_m,
    bool last_tile){
    #pragma HLS INLINE off
    float BRAM_score[tile_size][tile_size];
    float BRAM_P[tile_size][tile_size];//if exp calculate at one cycle may not need to use BRAM_P just send P into buffer
    #pragma HLS ARRAY_PARTITION variable=BRAM_score type=cyclic dim=2 factor=4
    #pragma HLS ARRAY_PARTITION variable=BRAM_P type=cyclic dim=2 factor=4
    #pragma HLS BIND_STORAGE variable=BRAM_score type=RAM_2P impl=LUTRAM
    #pragma HLS BIND_STORAGE variable=BRAM_P type=RAM_2P impl=LUTRAM
    load_score_FIFO2BRAM(FIFO_score, BRAM_score);
    compute_softmax_loop:
    for (int i = 0; i < tile_size; i++) {
        float m_tmp = -1e30f;
        float l_tmp = 0;
        compute_max_loop:
        for(int j = 0; j < tile_size; j++) {
            #pragma HLS PIPELINE II=1
            if(last_tile && j >=5){
                BRAM_score[i][j] = m_tmp;
            }else{
                float s = BRAM_score[i][j];
                m_tmp = hls::max(m_tmp, s);
            }
        }
        FIFO_m.write(m_tmp);
        compute_sum_exp_loop:
        for(int j = 0; j < tile_size; j+=8) {
            #pragma HLS PIPELINE II=1
            float p[8];
            for(int k = 0; k < 8; k++) {
                #pragma HLS UNROLL factor=8
                p[k] = hls::exp(BRAM_score[i][j + k] - m_tmp);
                BRAM_P[i][j + k] = p[k];
            }
            float p_1sum[4];
            float p_2sum[2];
            for(int k = 0; k < 4; k++) {
                #pragma HLS UNROLL factor=4
                p_1sum[k] = p[2*k] + p[2*k + 1];
            }
            for(int k = 0; k < 2; k++) {
                #pragma HLS UNROLL factor=2
                p_2sum[k] = p_1sum[2*k] + p_1sum[2*k + 1];
            }
            l_tmp += p_2sum[0] + p_2sum[1];
        }
        if(last_tile){
            l_tmp -= 11.0f;
        }
        FIFO_l.write(l_tmp);
        // compute_P_loop:
        // for(int j = 0; j < tile_size; j+=8) {
        //     #pragma HLS PIPELINE II=1
        //     for(int k = 0;k < 8; k++) {
        //         #pragma HLS UNROLL factor=8
        //         BRAM_P[i][j + k] /= l_tmp;
        //     }
        // }
        send_P_loop:
        for(int j = 0; j < tile_size; j++) {
            #pragma HLS PIPELINE II=1
            FIFO_P.write(BRAM_P[i][j]);
        }
    }
}

void compute_output(
    hls::stream<float> &FIFO_m,
    hls::stream<float> &FIFO_l,
    hls::stream<float> &FIFO_P,
    hls::stream<uint256_dt> &FIFO_O_out,
    float BRAM_m[SEQ_LEN + 11],
    float BRAM_l[SEQ_LEN + 11],
    float BRAM_O[tile_size][HEAD_DIM],
    int BRAM_V[tile_size][HEAD_DIM],
    int tile_idx
){
    #pragma HLS INLINE off
    for(int i = 0; i < tile_size; i++) {
        float m_ij = FIFO_m.read();
        float l_ij = FIFO_l.read();
        float m_i = BRAM_m[i + tile_idx*16];
        float l_i = BRAM_l[i + tile_idx*16];
        float new_m_i = hls::max(m_i, m_ij);
        float exp_diff_mi = hls::exp(m_i - new_m_i);
        float exp_diff_mij = hls::exp(m_ij - new_m_i);
        float new_l_i = exp_diff_mi * l_i + exp_diff_mij * l_ij;
        float p[16];
        read_P_matrix:
        for(int j = 0;j < tile_size;j+=1){
            #pragma HLS PIPELINE II=1
            p[j] = FIFO_P.read();
        }
        for(int j = 0;j < HEAD_DIM;j++){
            float O_new = 0;
            for(int k = 0;k < tile_size;k+=8){
                #pragma HLS PIPELINE II=1
                float reg_O[8];
                for(int w = 0;w < 8;w++){
                    #pragma HLS UNROLL factor=8
                    reg_O[w] = p[k+w]*BRAM_V[k+w][j];
                }
                float O_1sum[4];
                float O_2sum[2];
                for(int w = 0; w < 4; w++) {
                    #pragma HLS UNROLL factor=4
                    O_1sum[w] = reg_O[2*w] + reg_O[2*w + 1];
                }
                for(int w = 0; w < 2; w++) {
                    #pragma HLS UNROLL factor=2
                    O_2sum[w] = O_1sum[2*w] + O_1sum[2*w + 1];
                }
                O_new += O_2sum[0] + O_2sum[1];
            }
            BRAM_O[i][j] = (int)((BRAM_O[i][j]*l_i*exp_diff_mi + O_new*exp_diff_mij)/new_l_i);
        }
        BRAM_m[i] = new_m_i;
        BRAM_l[i] = new_l_i;
    }
    
    uint256_dt O_256;
    for(int i = 0; i < tile_size; i++) {
        for(int j = 0; j < HEAD_DIM; j+=8) {
            #pragma HLS PIPELINE II=1
            for(int k = 0; k < 8; k++) {
                #pragma HLS UNROLL factor=8
                O_256.range((j + k) * 32 + 31, (j + k) * 32) = (int)BRAM_O[i][j + k];
            }
            FIFO_O_out.write(O_256);
        }
    }
}

void store_O_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_O_out,
    int O[SEQ_LEN][HEAD_DIM]
){
    uint256_dt *Global_256_ptr = (uint256_dt*)O;
    int total_chunks = (SEQ_LEN * HEAD_DIM) / 8;

    store_loop:
    for (int i = 0; i < total_chunks; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt temp = FIFO_O_out.read();
        Global_256_ptr[i] = temp;
    }
}