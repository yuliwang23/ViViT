#ifndef HOST_H
#define HOST_H

constexpr int SEQ_LEN = 14 * 14 + 1; // 14x14 patches + 1 CLS token
constexpr int EMBED_DIM = 768;
constexpr int NUM_HEADS = 12;
constexpr int FRAME = 16;
constexpr int HEAD_DIM = EMBED_DIM / NUM_HEADS;
constexpr int MLP_DIM = 3072;
constexpr int NUM_ENCODER_BLOCKS = 12;

constexpr int TEMP_LEN = 17;
static_assert(SEQ_LEN == 197, "spatial encoder kernels expect 196 patches plus one CLS token");
static_assert(HEAD_DIM == 64, "spatial encoder attention kernels expect 64 channels per head");
static_assert(EMBED_DIM % NUM_HEADS == 0, "embedding dimension must split evenly across heads");
static_assert(MLP_DIM == 4 * EMBED_DIM, "spatial encoder MLP expansion must be 4x embedding dimension");

#endif