

static void linear_quant_host_int_input(
    int8_t in[SEQ_LEN][EMBED_DIM],
    float scale_input[SEQ_LEN],
    int8_t weight[EMBED_DIM][EMBED_DIM],
    float bias[EMBED_DIM],
    float scale_weight[EMBED_DIM],
    int out[SEQ_LEN][EMBED_DIM])
{
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int o = 0; o < EMBED_DIM; o++)
        {
            float acc = bias[o];
            const float input_scale = scale_input[i] * scale_weight[o];
            for (int d = 0; d < EMBED_DIM; d++)
            {
                acc += static_cast<float>(in[i][d]) * static_cast<float>(weight[d][o]) * input_scale;
            }
            out[i][o] = static_cast<int>(roundf(acc));
        }
    }
}

// TODO: not implement here, not used?
// replace with quant. version
static void mlp_host(int in[SEQ_LEN][EMBED_DIM],
                     int W1[EMBED_DIM][MLP_DIM], int b1[MLP_DIM],
                     int W2[MLP_DIM][EMBED_DIM], int b2[EMBED_DIM],
                     int out[SEQ_LEN][EMBED_DIM])
{
    vector<int> hidden(MLP_DIM);
    for (int i = 0; i < SEQ_LEN; i++)
    {
        for (int h = 0; h < MLP_DIM; h++)
        {
            long long acc = b1[h];
            for (int d = 0; d < EMBED_DIM; d++)
                acc += static_cast<long long>(in[i][d]) * W1[d][h];
            hidden[h] = gelu_int(static_cast<int>(acc >> 8));
        }
        for (int d = 0; d < EMBED_DIM; d++)
        {
            long long acc = b2[d];
            for (int h = 0; h < MLP_DIM; h++)
                acc += static_cast<long long>(hidden[h]) * W2[h][d];
            out[i][d] = static_cast<int>(acc >> 8);
        }
    }
}