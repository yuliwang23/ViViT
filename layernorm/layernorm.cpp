#include "layernorm.h"

void layernorm(float x[SEQ_LEN][EMBED_DIM],
               float gamma[EMBED_DIM],
               float beta[EMBED_DIM],
               int8_dt out[SEQ_LEN][EMBED_DIM]){
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=gamma offset=slave bundle=gmem0 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=beta offset=slave bundle=gmem1 max_widen_bitwidth=256
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 max_widen_bitwidth=256

#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=gamma bundle=control
#pragma HLS INTERFACE s_axilite port=beta bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control

static float SRAM_X[EMBED_DIM];
static float SRAM_gamma[EMBED_DIM];
static float SRAM_beta[EMBED_DIM];
static int8_dt SRAM_out[EMBED_DIM];
hls::stream<uint256_dt> FIFO_X;
hls::stream<uint256_dt> FIFO_out;
#pragma HLS STREAM variable=FIFO_X depth=256
#pragma HLS STREAM variable=FIFO_out depth=256

#pragma HLS ARRAY_PARTITION variable=SRAM_X type=cyclic factor=16
#pragma HLS ARRAY_PARTITION variable=SRAM_gamma type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_beta type=cyclic factor=8
#pragma HLS ARRAY_PARTITION variable=SRAM_out type=cyclic factor=16

#pragma HLS BIND_STORAGE variable=SRAM_X type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=SRAM_gamma type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=SRAM_beta type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=SRAM_out type=ram_2p impl=bram
    load_weights_HBM2SRAM(gamma, beta, SRAM_gamma, SRAM_beta);
    for(int i = 0; i < SEQ_LEN; i++){
        #pragma HLS DATAFLOW
        load_X_HBM2FIFO(x[i], FIFO_X);
        {
        load_X_FIFO2BRAM(FIFO_X, SRAM_X);
        compute_mean_variance(SRAM_X, SRAM_gamma, SRAM_beta, SRAM_out);
        store_out_BRAM2FIFO(SRAM_out, FIFO_out);
        }
        store_out_FIFO2HBM(FIFO_out, out[i]);
    }
}

void load_X_HBM2FIFO(
    float X[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_X)
{
    // Use reinterpret_cast for safe bitwise pointer casting
    uint256_dt *X_256_ptr = reinterpret_cast<uint256_dt*>(X);
    
    load_X_loop:
    for (int i = 0; i < EMBED_DIM / 8; i++) {
    #pragma HLS PIPELINE II=1
        uint256_dt temp_chunk = X_256_ptr[i];
        FIFO_X.write(temp_chunk);
    }
}

void load_X_FIFO2BRAM(
    hls::stream<uint256_dt> &FIFO_X,
    float BRAM_X[EMBED_DIM])
{
   load_X_col:
   for (int j = 0; j < EMBED_DIM/8; j++){
   #pragma HLS PIPELINE II=1
      
      // Local Register Buffer
      float reg_buffer[8];
      #pragma HLS ARRAY_PARTITION variable=reg_buffer type=complete
      
      // Read & Unpack
      uint256_dt temp_chunk = FIFO_X.read();
      
      unpack_loop:
      for(int k = 0; k < 8; k++) {
         #pragma HLS UNROLL 
         int start_bit = k * 32;
         int end_bit = start_bit + 31;
         
         // Extract the raw 32 bits from the 256-bit chunk
         uint32_t raw_bits = temp_chunk.range(end_bit, start_bit);
         
         // Use a union to perform safe Type Punning
         union {
             uint32_t i_val;
             float f_val;
         } converter;
         
         // Assign bits to integer side, read out from float side
         converter.i_val = raw_bits;
         reg_buffer[k] = converter.f_val; 
      }
      
      write_bram_loop:
      for(int k = 0; k < 8; k++) {
         #pragma HLS UNROLL
         BRAM_X[j*8+k] = reg_buffer[k];
      }
   }
}

void compute_mean_variance(
    float BRAM_X[EMBED_DIM],
    float gamma[EMBED_DIM],
    float beta[EMBED_DIM],
    int8_dt out[EMBED_DIM]
){
    float local_global_mean = 0.0f; 
    float M2AB = 0.0f;
    float norm_buffer[EMBED_DIM];
    #pragma HLS ARRAY_PARTITION variable=norm_buffer type=cyclic factor=16
    
    #pragma HLS INLINE off
    compute_top_loop:
    for(int j = 0, i = 0; j < EMBED_DIM; j+=16, i++){     
        #pragma HLS PIPELINE II=1 
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        compute_local_sum:
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            float val = BRAM_X[j+k];
            local_sum += val;
            local_sum_sq += val * val;
        }
        float X_b_mean = local_sum * 0.0625f; 
        float M_2_B = local_sum_sq - (local_sum * local_sum * 0.0625f);
        float mean_diff = X_b_mean - local_global_mean;
        local_global_mean += MEAN_WT.data[i] * mean_diff;
        M2AB += M2_WT.data[i] * mean_diff * mean_diff + M_2_B;
    }
    
    float var = M2AB * EMBED_DIM_INV; 
    float inv_std_out = 1.0f / std::sqrt(var + EPSILON); 
    float max_val = -FLT_MAX;
    
    compute_norm_and_find_max_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE II=1
        float local_max = -FLT_MAX;
        
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            int idx = j + k;
            float norm_val = (BRAM_X[idx] - local_global_mean) * inv_std_out * gamma[idx] + beta[idx];
            
            norm_buffer[idx] = norm_val;
            
            float abs_val = std::abs(norm_val);
            local_max = std::max(local_max, abs_val);
        }
        max_val = std::max(max_val, local_max);
    }
    
    float scale = 127.0f / max_val;
    quant_loop:
    for(int j = 0; j < EMBED_DIM; j+=16){
        #pragma HLS PIPELINE II=1
        for(int k = 0; k < 16; k++){
            #pragma HLS UNROLL
            int idx = j + k;
            float scaled_val = norm_buffer[idx] * scale; 
            
            out[idx] = static_cast<int8_dt>(scaled_val); 
        }
    }
}

void store_out_BRAM2FIFO(
    int8_dt out_BRAM[EMBED_DIM],
    hls::stream<uint256_dt> &FIFO_out
){
    #pragma HLS INLINE off
    stream_out_loop:
    for (int j = 0; j < EMBED_DIM; j += 32) { 
        #pragma HLS PIPELINE II=1
        uint256_dt out_chunk;
        pack_loop:
        for (int r = 0; r < 32; r++) {
            #pragma HLS UNROLL
            int start_bit = r << 3; 
            int end_bit = start_bit + 7;
            out_chunk.range(end_bit, start_bit) = out_BRAM[j+r];
        }
        FIFO_out.write(out_chunk);
    }
}

void store_out_FIFO2HBM(
    hls::stream<uint256_dt> &FIFO_out,
    int8_dt Output_Global[EMBED_DIM] 
) {
    #pragma HLS INLINE off
    uint256_dt *Global_256_ptr = reinterpret_cast<uint256_dt*>(Output_Global);
    int total_chunks = EMBED_DIM / 32;

    store_hbm_loop:
    for (int i = 0; i < total_chunks; i++) {
        #pragma HLS PIPELINE II=1
        
        uint256_dt temp = FIFO_out.read();
        Global_256_ptr[i] = temp;
    }
}

void load_weights_HBM2SRAM(
    float HBM_gamma[EMBED_DIM],
    float HBM_beta[EMBED_DIM],
    float SRAM_gamma[EMBED_DIM],
    float SRAM_beta[EMBED_DIM]
) {
    #pragma HLS INLINE off
    uint256_dt *gamma_256_ptr = reinterpret_cast<uint256_dt*>(HBM_gamma);
    uint256_dt *beta_256_ptr  = reinterpret_cast<uint256_dt*>(HBM_beta);
    load_weights_loop:
    for (int i = 0; i < EMBED_DIM / 8; i++) {
        #pragma HLS PIPELINE II=1
        
        uint256_dt temp_gamma_chunk = gamma_256_ptr[i];
        uint256_dt temp_beta_chunk  = beta_256_ptr[i];
        
        unpack_write_loop:
        for(int k = 0; k < 8; k++) {
            #pragma HLS UNROLL 
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            
            uint32_t raw_gamma_bits = temp_gamma_chunk.range(end_bit, start_bit);
            union {
                uint32_t i_val;
                float f_val;
            } conv_gamma;
            conv_gamma.i_val = raw_gamma_bits;
            SRAM_gamma[i*8 + k] = conv_gamma.f_val;
            
            uint32_t raw_beta_bits = temp_beta_chunk.range(end_bit, start_bit);
            union {
                uint32_t i_val;
                float f_val;
            } conv_beta;
            conv_beta.i_val = raw_beta_bits;
            SRAM_beta[i*8 + k] = conv_beta.f_val;
        }
    }
}
