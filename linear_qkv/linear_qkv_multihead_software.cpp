#include <cmath>

constexpr int SEQ_LEN = 14*14 + 1; // 14*14*16 --> per frame // 14*15
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
// this version is no optimize
void linear_qkv_multihead_opt(
    int X[SEQ_LEN][EMBED_DIM],
    int Wq[EMBED_DIM][EMBED_DIM],
    int bq[EMBED_DIM],
    int Wk[EMBED_DIM][EMBED_DIM],
    int bk[EMBED_DIM],
    int Wv[EMBED_DIM][EMBED_DIM],
    int bv[EMBED_DIM],
    int Q[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int K[NUM_HEADS][SEQ_LEN][HEAD_DIM],
    int V[NUM_HEADS][SEQ_LEN][HEAD_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < NUM_HEADS; h++)
        {
            for (int d = 0; d < HEAD_DIM; d++)
            {
                float q = 0, k = 0, v = 0;
                int col = h * HEAD_DIM + d;
                for (int e = 0; e < EMBED_DIM; e++)
                {
                    q += X[i][e] * Wq[e][col] + bq[e];
                    k += X[i][e] * Wk[e][col] + bk[e];
                    v += X[i][e] * Wv[e][col] + bv[e];
                }
                Q[h][i][d] = q;
                K[h][i][d] = k;
                V[h][i][d] = v;
            }
        }
    }
}
