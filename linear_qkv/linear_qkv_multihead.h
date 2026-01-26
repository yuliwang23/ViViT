#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ap_int.h>

constexpr int SEQ_LEN = 14*14 + 1; // 14*14*16 --> per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int tile_size_dim = 64;
constexpr int tile_size_sql = 64;
constexpr int tile_sql_time = SEQ_LEN / tile_size_sql;
constexpr int tile_sql_remain = SEQ_LEN % tile_size_sql;
typedef ap_uint<512> uint512_dt;
 
void linear_qkv_multihead(
    int X[SEQ_LEN][EMBED_DIM],
    int Wq[EMBED_DIM][EMBED_DIM],
    int bq[EMBED_DIM],
    int Wk[EMBED_DIM][EMBED_DIM],
    int bk[EMBED_DIM],
    int Wv[EMBED_DIM][EMBED_DIM],
    int bv[EMBED_DIM],
    int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM]);




#include <cmath>



