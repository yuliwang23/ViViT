#include "softmax.h"
#include "exp.h"
// Sub-function Declarations
void load_HBM2BRAM(uint256_dt* in, float BRAM_input[VEC_MAX]);
void calculate_max_and_sum(float BRAM_input[VEC_MAX], float &max_val, float &sum_exp);
void normalize_BRAM2HBM(float BRAM_input[VEC_MAX], float max_val, float sum_exp, uint256_dt* out);
extern "C" {
void softmax(uint256_dt* in, uint256_dt* out)
{
    // AXI Interfaces
    #pragma HLS INTERFACE m_axi port=in offset=slave bundle=gmem0 
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 
    #pragma HLS INTERFACE s_axilite port=in bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    // Local BRAM Buffer
    float BRAM_input[VEC_MAX];
    // Partition BRAM cyclically by 8 to allow 8 simultaneous reads/writes per clock cycle
    #pragma HLS ARRAY_PARTITION variable=BRAM_input dim=1 type=cyclic factor=8

    float max_val = -1e38f;
    float sum_exp = 0.0f;

    // Sequential Execution Flow (No DATAFLOW due to Softmax hard barrier)
    
    // 1. Burst read from HBM and unpack to BRAM
    load_HBM2BRAM(in, BRAM_input);
    
    // 2. Compute max and sum
    calculate_max_and_sum(BRAM_input, max_val, sum_exp);
    
    // 3. Normalize, pack, and burst write to HBM
    normalize_BRAM2HBM(BRAM_input, max_val, sum_exp, out);
}
}
void load_HBM2BRAM(uint256_dt* in, float BRAM_input[VEC_MAX])
{
    load_BRAM_loop:
    for (int i = 0; i < VEC_MAX / 8; i++) {
        #pragma HLS PIPELINE II=1
        
        uint256_dt temp_chunk = in[i]; // Burst read 256-bit
        
        unpack_loop:
        for (int k = 0; k < 8; k++) {
            #pragma HLS UNROLL
            int start_bit = k * 32;
            int end_bit = start_bit + 31;
            
            float_cast_t converter;
            // Extract 32 bits and safely cast to float
            converter.i = temp_chunk.range(end_bit, start_bit); 
            
            // Write to cyclically partitioned BRAM (8 parallel writes)
            BRAM_input[i * 8 + k] = converter.f; 
        }
    }
}
void calculate_max_and_sum(float BRAM_input[VEC_MAX], float &max_val, float &sum_exp)
{
    calc_loop:
    for (int i = 0; i < VEC_MAX; i++)
    {
        #pragma HLS PIPELINE
        float x = BRAM_input[i];
        
        if (x > max_val)
        {
            float diff = max_val - x;
            sum_exp = sum_exp * exp_top(diff) + 1.0f;
            max_val = x;
        }
        else
        {
            float diff = x - max_val;
            sum_exp += exp_top(diff);
        }
    }
}

void normalize_BRAM2HBM(float BRAM_input[VEC_MAX], float max_val, float sum_exp, uint256_dt* out)
{
    // Hardware Optimization: Calculate reciprocal outside the loop
    float inv_sum = 1.0f / sum_exp;

    norm_pack_loop:
    for (int i = 0; i < VEC_MAX / 8 + 1; i++)
    {
        #pragma HLS PIPELINE II=1
        uint256_dt out_chunk;
        
        pack_loop:
		if(i!=VEC_MAX/8){
			for (int k = 0; k < 8; k++) {
				#pragma HLS UNROLL

				// Read from cyclically partitioned BRAM (8 parallel reads)
				float x = BRAM_input[i * 8 + k];
				float diff = x - max_val;
				float normalized = exp_top(diff) * inv_sum;

				float_cast_t converter;
				converter.f = normalized;

				// Pack 32 bits into the 256-bit chunk
				int start_bit = k * 32;
				int end_bit = start_bit + 31;
				out_chunk.range(end_bit, start_bit) = converter.i;
			}
		}else{
		for (int k = 0; k < 5; k++) {
			#pragma HLS UNROLL

			// Read from cyclically partitioned BRAM (8 parallel reads)
			float x = BRAM_input[i * 8 + k];
			float diff = x - max_val;
			float normalized = exp_top(diff) * inv_sum;

			float_cast_t converter;
			converter.f = normalized;

			// Pack 32 bits into the 256-bit chunk
			int start_bit = k * 32;
			int end_bit = start_bit + 31;
			out_chunk.range(end_bit, start_bit) = converter.i;
			}
		}
        
        // Burst write 256-bit back to HBM
        out[i] = out_chunk; 
    }
}
