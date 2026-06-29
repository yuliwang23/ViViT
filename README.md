# Decrease_params Branch Usage

This branch integrates the spatial encoder into two large kernels:

1. `multihead_attention`: LayerNorm0 + QKV projection + attention + output projection + residual.
2. `mlp`: LayerNorm1 + MLP + residual.

The spatial encoder host now calls these two kernels instead of the old small-kernel chain:
`linear_qkv_multihead -> attention_opt -> projection_out -> residual -> mlp_q -> residual`.

## Vitis Kernel Sources

Import these files into the Vitis kernel/source side:

- `vivit_config.h`
- `multihead_attention/multihead_attention.h`
- `multihead_attention/multihead_attention.cpp`
- `multilayer_perceptron/mlp.h`
- `multilayer_perceptron/mlp.cpp`

Do not use the old spatial encoder small kernels for this flow unless you intentionally switch back to the legacy path. The old small kernels include:

- `linear_with_layernorm/linear_qkv_multihead.cpp`
- `attention/attention.cpp`
- `projection_out/projection.cpp`
- `residual/residual.cpp`
- old `mlp_q` sources

The host expects the xclbin to export these kernel names exactly:

- `multihead_attention`
- `mlp`

## Vitis Host Sources

Import these files into the Vitis host/application side:

- `spatial_encoder/host_Q_head_debugV8.cpp`
- `spatial_encoder/host.h`
- `spatial_encoder/help_functions.h`
- `spatial_encoder/help_functions.cpp`
- `spatial_encoder/utility.h`
- `spatial_encoder/utility.cpp`
- `vivit_config.h`

`host_Q_head_debugV8.cpp` is the active host for this branch. It creates only `multihead_attention` and `mlp`, then copies the MLP output back into the frame input buffer for the next encoder block.

## Switching Parameter Size

All configurable dimensions are centralized in `vivit_config.h`.

Default decrease/small setting:

```cpp
#define VIVIT_SMALL_CONFIG 1
```

Original ViViT-sized setting:

```cpp
#define VIVIT_SMALL_CONFIG 0
```

You can switch in either of these ways:

1. Edit `vivit_config.h` directly and set `VIVIT_SMALL_CONFIG` to `1` or `0`.
2. Keep the file unchanged and pass a compile definition from Vitis, for example `-DVIVIT_SMALL_CONFIG=0` for original size.

When `VIVIT_SMALL_CONFIG=1`, the branch uses the decrease parameters:

- `SEQ_LEN = 4`
- `EMBED_DIM = 64`
- `NUM_HEADS = 2`
- `FRAME = 2`
- `MLP_DIM = 256`
- `NUM_ENCODER_BLOCKS = 1`

When `VIVIT_SMALL_CONFIG=0`, the branch uses the original parameters:

- `SEQ_LEN = 14 * 14 + 1`
- `EMBED_DIM = 768`
- `NUM_HEADS = 12`
- `FRAME = 16`
- `MLP_DIM = 3072`
- `NUM_ENCODER_BLOCKS = 12`

After changing this setting, rebuild both the kernel xclbin and the host application. The host and kernels must be compiled with the same `VIVIT_SMALL_CONFIG` value.

## Link Configuration Note

`spatial_encoder/link.cfg` may still contain the old small-kernel connectivity. For this branch, update the Vitis link configuration to instantiate the new kernels instead:

```ini
[connectivity]
nk=multihead_attention:1
nk=mlp:1
```

Then map the AXI ports for `multihead_attention_1` and `mlp_1` according to the generated kernel interface names in Vitis. The old `linear_qkv_multihead`, `attention_opt`, `projection_out`, `residual`, and `mlp_q` connectivity entries should not be used with `host_Q_head_debugV8.cpp`.

## Runtime Flow

For each spatial encoder block and frame, the host does:

1. Load the block parameters from `params/all_params/SpatialTransformer/encoderblock_<n>/`.
2. Pack Q/K/V/Wout and MLP weights into the layout expected by the integrated kernels.
3. Run `multihead_attention` and write the attention residual output.
4. Run `mlp` and write the final block output.
5. Copy the final block output back to the frame input buffer for the next encoder block.

The final spatial encoder output is written to `spatial_encoder_output.txt`.
