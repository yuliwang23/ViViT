#ifndef HOST_H
#define HOST_H

#include "../vivit_config.h"

constexpr int SEQ_LEN = VIVIT_SEQ_LEN;
constexpr int EMBED_DIM = VIVIT_EMBED_DIM;
constexpr int NUM_HEADS = VIVIT_NUM_HEADS;
constexpr int FRAME = VIVIT_FRAME;
constexpr int HEAD_DIM = VIVIT_HEAD_DIM;
constexpr int MLP_DIM = VIVIT_MLP_DIM;
constexpr int NUM_ENCODER_BLOCKS = VIVIT_NUM_ENCODER_BLOCKS;

constexpr int TEMP_LEN = VIVIT_TEMP_LEN;
static_assert(EMBED_DIM % NUM_HEADS == 0, "embedding dimension must split evenly across heads");
static_assert(MLP_DIM == 4 * EMBED_DIM, "spatial encoder MLP expansion must be 4x embedding dimension");

#endif
