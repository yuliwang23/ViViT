#include "attention.h"

extern "C" void attention_opt(
    float Q[SEQ_LEN][HEAD_DIM],
    float K[SEQ_LEN][HEAD_DIM],
    float V[SEQ_LEN][HEAD_DIM],
    float Out[SEQ_LEN][HEAD_DIM])
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

hls::stream<uint256_dt> FIFO_Q_chunk("FIFO_Q");
hls::stream<uint256_dt> FIFO_K_chunk("FIFO_K");
hls::stream<uint256_dt> FIFO_V_chunk("FIFO_V");
hls::stream<float> FIFO_score("FIFO_score");
hls::stream<float> FIFO_P("FIFO_P");
hls::stream<float> FIFO_m("FIFO_m");
hls::stream<float> FIFO_l("FIFO_l");
hls::stream<uint256_dt> FIFO_O_out("FIFO_O_out");

#pragma HLS STREAM variable=FIFO_Q_chunk depth=256
#pragma HLS STREAM variable=FIFO_K_chunk depth=256
#pragma HLS STREAM variable=FIFO_V_chunk depth=256
#pragma HLS STREAM variable=FIFO_score depth=256
#pragma HLS STREAM variable=FIFO_P depth=256
#pragma HLS STREAM variable=FIFO_l depth=256
#pragma HLS STREAM variable=FIFO_m depth=256
#pragma HLS STREAM variable=FIFO_O_out depth=256

float BRAM_Q[tile_size][HEAD_DIM];
float BRAM_K[tile_size][HEAD_DIM];
float BRAM_V[tile_size][HEAD_DIM];
float BRAM_O[tile_size][HEAD_DIM];
float BRAM_m[tile_size];
float BRAM_l[tile_size];

#pragma HLS ARRAY_PARTITION variable=BRAM_Q type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_K type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_V type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_V type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=BRAM_O type=cyclic dim=2 factor=8
#pragma HLS BIND_STORAGE variable=BRAM_Q type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_K type=RAM_2P impl=bram
#pragma HLS BIND_STORAGE variable=BRAM_V type=RAM_2P impl=lutram
#pragma HLS BIND_STORAGE variable=BRAM_O type=RAM_2P impl=bram

    Outer_Q_Loop:
    for (int i = 0; i < num_tiles; i++) {
        bool last_tile_Q = (i == num_tiles - 1);
        init_states(BRAM_m, BRAM_l, BRAM_O);
        uint256_dt *Q_ptr = (uint256_dt*)&Q[i * tile_size];
        load_tile_HBM2FIFO(Q_ptr, FIFO_Q_chunk, last_tile_Q); 
        load_QKV_FIFO2BRAM(FIFO_Q_chunk, BRAM_Q);
        Inner_KV_Loop:
        for (int j = 0; j < num_tiles; j++) {
            bool last_tile_KV = (j == num_tiles - 1);
            #pragma HLS DATAFLOW
            uint256_dt *K_ptr = (uint256_dt*)&K[j * tile_size];
            load_tile_HBM2FIFO(K_ptr, FIFO_K_chunk, last_tile_KV);
            load_QKV_FIFO2BRAM(FIFO_K_chunk, BRAM_K);
            uint256_dt *V_ptr = (uint256_dt*)&V[j * tile_size];
            load_tile_HBM2FIFO(V_ptr, FIFO_V_chunk, last_tile_KV);
            load_QKV_FIFO2BRAM(FIFO_V_chunk, BRAM_V);
            compute_score(BRAM_Q, BRAM_K, FIFO_score);
            compute_softmax(FIFO_score, FIFO_P, FIFO_l, FIFO_m, last_tile_KV);
            compute_output(FIFO_m, FIFO_l, FIFO_P, FIFO_O_out, BRAM_m, BRAM_l, BRAM_O, BRAM_V, last_tile_KV);
        }
        uint256_dt *Out_ptr = (uint256_dt*)&Out[i * tile_size];
        store_tile_FIFO2HBM(FIFO_O_out, Out_ptr, last_tile_Q);
    }
}

void load_tile_HBM2FIFO(
    uint256_dt *HBM_ptr, 
    hls::stream<uint256_dt> &FIFO_QKV,
    bool last_tile
){
    #pragma HLS INLINE off
    int valid_tokens = last_tile ? 5 : tile_size; 
    int padding_tokens = tile_size - valid_tokens;
    int valid_chunks = valid_tokens * (HEAD_DIM / 8);
    int padding_chunks = padding_tokens * (HEAD_DIM / 8);

    load_valid_loop:
    for (int i = 0; i < valid_chunks; i++) {
        #pragma HLS PIPELINE II=1
        FIFO_QKV.write(HBM_ptr[i]);
    }
    load_padding_loop:
    for(int i = 0; i < padding_chunks; i++){
        #pragma HLS PIPELINE II=1
        FIFO_QKV.write(0);
    }
}

void load_QKV_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_QKV,
    float BRAM_QKV[tile_size][HEAD_DIM])
{
    #pragma HLS INLINE off
    union {
        uint32_t i;
        float f;
    } converter;

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
                converter.i = (uint32_t)temp_chunk.range(end_bit, start_bit);
                BRAM_QKV[i][j + k] = converter.f;
            }
        }
    }
}

void compute_score(
    float BRAM_Q[tile_size][HEAD_DIM],
    float BRAM_K[tile_size][HEAD_DIM],
    hls::stream<float> &FIFO_score
){
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=BRAM_Q type=cyclic dim=2 factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_K type=cyclic dim=2 factor=8
    compute_i_loop:
    for(int i = 0; i < tile_size; i++) {
        compute_j_loop:
        for (int j = 0; j < tile_size; j++) {
            float sum = 0.0f;
            compute_d_loop:
            for (int d = 0; d < HEAD_DIM; d+=8) {
                #pragma HLS PIPELINE II=1
                float local_sum = 0.0f;
                for(int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL factor=8
                    local_sum += BRAM_Q[i][d + k] * BRAM_K[j][d + k];
                }
                sum += local_sum;
            }
            float score = sum * 0.125f;
            FIFO_score.write(score);
        }
    }
}

void load_score_FIFO2BRAM(
    hls::stream<float> &FIFO_score,
    float BRAM_score[tile_size][tile_size])
{
    #pragma HLS INLINE off
    load_score_i_loop:
    for (int i = 0; i < tile_size; i++) {
        load_score_j_loop:
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
    float BRAM_P[tile_size][tile_size];
    #pragma HLS ARRAY_PARTITION variable=BRAM_score type=cyclic dim=2 factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_P type=cyclic dim=2 factor=8
    #pragma HLS BIND_STORAGE variable=BRAM_score type=RAM_2P impl=bram
    #pragma HLS BIND_STORAGE variable=BRAM_P type=RAM_2P impl=bram
    load_score_FIFO2BRAM(FIFO_score, BRAM_score);
    compute_softmax_loop:
    for (int i = 0; i < tile_size; i++) {
        float m_tmp = -1e30f;
        float l_tmp = 0.0f;
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
            float local_l_tmp_sum = 0.0f;
            for(int k = 0; k < 8; k++) {
                #pragma HLS UNROLL factor=8
                p[k] = hls::exp(BRAM_score[i][j + k] - m_tmp);
                BRAM_P[i][j + k] = p[k];
                local_l_tmp_sum += p[k];
            }
            l_tmp += local_l_tmp_sum;
        }
        if(last_tile){
            l_tmp -= 11.0f;
        }
        FIFO_l.write(l_tmp);
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
    float BRAM_m[tile_size],
    float BRAM_l[tile_size],
    float BRAM_O[tile_size][HEAD_DIM],
    float BRAM_V[tile_size][HEAD_DIM],
    bool last_tile
){
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=BRAM_V type=cyclic dim=1 factor=8
    #pragma HLS ARRAY_PARTITION variable=BRAM_O type=cyclic dim=2 factor=8

    for(int i = 0; i < tile_size; i++) {
        float m_ij = FIFO_m.read();
        float l_ij = FIFO_l.read();
        float m_i = BRAM_m[i];
        float l_i = BRAM_l[i];
        float new_m_i = hls::max(m_i, m_ij);
        float exp_diff_mi = hls::exp(m_i - new_m_i);
        float exp_diff_mij = hls::exp(m_ij - new_m_i);
        float new_l_i = exp_diff_mi * l_i + exp_diff_mij * l_ij;
        float p[tile_size];
        #pragma HLS ARRAY_PARTITION variable=p type=cyclic dim=1 factor=8
        read_P_matrix:
        for(int j = 0;j < tile_size;j+=1){
            #pragma HLS PIPELINE II=1
            p[j] = FIFO_P.read();
        }
        compute_output:
        for(int j = 0;j < HEAD_DIM;j++){
            float O_new = 0.0f;
            for(int k = 0;k < tile_size;k+=8){
                #pragma HLS PIPELINE II=1
                float local_O_new = 0.0f;
                for(int w = 0;w < 8;w++){
                    #pragma HLS UNROLL factor=8
                    local_O_new += p[k+w]*BRAM_V[k+w][j];
                }
                O_new += local_O_new;
            }
            BRAM_O[i][j] = (BRAM_O[i][j]*exp_diff_mi + O_new*exp_diff_mij);
        }
        BRAM_m[i] = new_m_i;
        BRAM_l[i] = new_l_i;
    }

    if(last_tile) {
        uint256_dt O_256;
        union {
            float f;
            uint32_t i;
        } converter;

        pack_output_i_loop:
        for(int i = 0; i < tile_size; i++) {
            float final_l_i = BRAM_l[i]; 
            pack_output_j_loop:
            for(int j = 0; j < HEAD_DIM; j += 8) {
                #pragma HLS PIPELINE II=1
                pack_output_k_loop:
                for(int k = 0; k < 8; k++) {
                    #pragma HLS UNROLL factor=8
                    int start_bit = k * 32;
                    int end_bit = start_bit + 31;
                    float normalized_O = BRAM_O[i][j + k] / final_l_i;
                    converter.f = normalized_O;
                    O_256.range(end_bit, start_bit) = converter.i; 
                }
                FIFO_O_out.write(O_256);
            }
        }
    }
}

void store_tile_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_O_out,
    uint256_dt *HBM_ptr,
    bool last_tile_Q 
){
    #pragma HLS INLINE off
    int valid_tokens = last_tile_Q ? 5 : tile_size; 
    int valid_chunks = valid_tokens * (HEAD_DIM / 8);
    int total_chunks_in_fifo = tile_size * (HEAD_DIM / 8);

    store_loop:
    for (int i = 0; i < total_chunks_in_fifo; i++) {
        #pragma HLS PIPELINE II=1
        uint256_dt temp = FIFO_O_out.read();
        if (i < valid_chunks) {
            HBM_ptr[i] = temp;
        }
    }
}

void init_states(
    float BRAM_m[tile_size],
    float BRAM_l[tile_size],
    float BRAM_O[tile_size][HEAD_DIM]
){
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=BRAM_O type=cyclic dim=2 factor=8

    init_loop:
    for(int i = 0; i < tile_size; i++) {
        BRAM_m[i] = -1e30f; // -infinity for max
        BRAM_l[i] = 0.0f;   // 0 for sum
        for(int j = 0; j < HEAD_DIM; j+=8) {
            #pragma HLS PIPELINE II=1
            for(int k = 0; k < 8; k++) {
                #pragma HLS UNROLL factor=8
                BRAM_O[i][j + k] = 0.0f;
            }
        }
    }
}