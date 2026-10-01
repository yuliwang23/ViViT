# A Quantization-Aware Frame-Pipelined FPGA Accelerator for Video Transformers

FPGA Spatial Encoder acceleration for ViViT Factorised Encoder, with CPU temporal processing and a 400-class output head. Submitted to ASP-DAC '27. Citation/DOI: TODO.

## Selected implementation

The final six-instance host and link configuration come from main `5504ce4` (`paper-submission-main`). Hardware kernels come from Decrease_params `eff3f5e` (`paper-submission-kernels`). The main four-kernel v0 snapshot is preserved in archive. See [MIGRATION.md](MIGRATION.md) for source revisions and the path map.

| Paper term | Kernel symbol | Source |
|---|---|---|
| MSA-Attn | `mha_attention` | [hw/mha_attention/mha_attention.cpp](hw/mha_attention/mha_attention.cpp) |
| MSA-Proj | `mha_output_projection` | [hw/mha_output_projection/mha_output_projection.cpp](hw/mha_output_projection/mha_output_projection.cpp) |
| FFN1 | `FFN1` | [hw/ffn1/ffn1.cpp](hw/ffn1/ffn1.cpp) |
| FFN2 | `FFN2` | [hw/ffn2/ffn2.cpp](hw/ffn2/ffn2.cpp) |

[host/main.cpp](host/main.cpp) creates six CUs of each kernel type and schedules four stages per frame using OpenCL events. The spatial input is 16 x 197 x 768; CPU temporal processing uses 17 x 768 with 12 encoder blocks. [host/temporal_layer.cpp](host/temporal_layer.cpp) implements CPU temporal operations with OpenMP; the host worker applies the final 400-class projection.

MSA-Attn processes heads through QKV projection, QK/Softmax and SV, with Q/K in BRAM and transposed V storage. Token-sized FP32 normalization buffers accumulate max-abs values before INT8 quantization. MSA-Attn/Proj exchange an FP32 attention map and per-token max; FFN1/FFN2 exchange an FP32 GELU map and per-token max. These boundaries complete reductions across heads and output tiles. LayerNorm, Softmax and GELU nonlinear operations use FP32 in the kernels; FPGA GELU uses a tanh approximation.

Each pipeline's four stages share one dedicated HBM bank. [configs/u280_6_instances.cfg](configs/u280_6_instances.cfg) places pipelines 1–2 on SLR0, 3–4 on SLR1 and 5–6 on SLR2, using banks 0, 4, 8, 12, 16 and 20 respectively. The platform string is `xilinx_u280_gen3x16_xdma_1_202211_1`.

## Repository layout

```text
hw/          selected HLS kernels; HLS source and headers
host/        final host, CPU temporal encoder and support code
configs/     six-instance U280 link/SLR/HBM configuration
tb/          four standalone single-kernel OpenCL harnesses
params/      runtime parameters and tensor fixtures
archive/     main kernel v0, historical hosts and earlier experiments
MIGRATION.md source revisions, complete path map and exact content edits
```

## Inputs and software scope

The host consumes precomputed encoder-input tensors. Raw-video preprocessing, tubelet/spatial embedding, and full FP32 / Static INT8 / Dynamic INT8 software baselines are maintained outside this repository. This checkout does not provide a raw-video inference pipeline.

`ViViT_Output` contains 400-dimensional FP32 logits produced by the Scenic/Flax/JAX ViViT software model. These are neither ground-truth labels nor Dynamic INT8 SW outputs. The program variable `label` holds software/golden logits.

The host pairs reference logits and encoder-block inputs by sample ID, reads the reference into `float label[400]`, and compares its argmax with that of `vivit_out[400]`. A match increments `success_cnt`. The printed success rate is the argmax agreement rate between the FPGA-accelerated application's output and the FP32 reference logits. For mismatches only, the host also computes MAE as `mean(abs(label[i] - vivit_out[i]))` across 400 logits.

Reference logits and encoder-block inputs are exported by an external JAX model. Export script location/link: TODO. Model checkpoint source: TODO.

This agreement rate must not be used as the Kinetics-400 Top-1 accuracy in paper Table 1. Top-1 requires evaluation against ground-truth labels; the evaluation program and data are not included in this repository. Their location and evaluation procedure: TODO.

## Parameters

`params/` contains 575 files: 537 runtime parameter files and 38 other tensor fixtures. The supplied size estimate is approximately 1.9 GB; this checkout totals 1,100,334,775 bytes (approximately 1.10 GB). The scope of the size difference remains TODO. Acquisition/hosting method: TODO.

## Build and run

Tool versions: Vitis 2023.2, Vivado 2023.2, and XRT 2023.2. Source dependencies include Vitis HLS headers, OpenCL/Xilinx host headers, POSIX facilities, and OpenMP. Compiler version and installation instructions: TODO.

The following Vitis project directories are external and are not included in this repository:

| External project | Corresponding repository sources |
|---|---|
| `spatial_v3_kernels` | `hw/` |
| `spatial_v3_system_hw_link` | `configs/` |
| `spatial_v3` | `host/` |

Obtaining the external Vitis projects: TODO. External project source references must correspond to the repository paths above.

```sh
make -C spatial_v3_kernels/Hardware
make -C spatial_v3_system_hw_link/Hardware
make -C spatial_v3/Hardware
```

Author-provided run command, executed in the external Vitis project environment:

```sh
./spatial_v2/Hardware/host_6CU \
Xilinx \
xilinx_u280_gen3x16_xdma_base_1 \
../../spatial_v3_system_hw_link/Hardware/binary_container_1.xclbin
```

The three arguments are the OpenCL platform vendor (matched against `CL_PLATFORM_VENDOR`), the device/platform name (specifically matched against `CL_DEVICE_NAME`), and the xclbin file path loaded by the host, respectively.

The run command uses `xilinx_u280_gen3x16_xdma_base_1`; the link configuration uses `xilinx_u280_gen3x16_xdma_1_202211_1`. These are distinct supplied identifiers. The build uses `spatial_v3`, while the supplied executable path uses `spatial_v2`. Compatibility and execution in the external environment have not been verified here.

## Known Issues

- `mha_attention.h` has an inconsistent include guard and declares an older `multihead_attention` top instead of the implementation's `mha_attention` interface.
- `ffn1.h` references undefined `MLP_DIM_package` in FFN2 helper declarations; some FFN1 helper declarations conflict with static definitions.
- `ffn2.cpp` calls helpers before their declarations/definitions; the header only declares its top kernel.
- `host/help_functions.h` uses unqualified `string`; standalone harnesses include it before their namespace directive.
- Host model and batch-export paths remain environment-specific. Raw-video preprocessing and full software baselines are external.
- CPU quantization/accumulation details differ from FPGA implementation details. Bitwise equivalence is not established.
- Static CU/argument/port/SLR/HBM and local-include checks do not prove compilation, C-sim, co-sim, hardware operation, accuracy or routed frequency.

For header issue locations, see [MIGRATION.md](MIGRATION.md) for specific locations. HLS/OpenCL system headers are external prerequisites, not repository-local include targets.

## Reported results

The following results are **as reported in the paper** and have not yet been reproduced from this repository. See [Inputs and software scope](#inputs-and-software-scope) for the distinction between the host's success rate and Top-1 accuracy.

| Kinetics-400 evaluation | Top-1 |
|---|---|
| FP32 | 71.33% |
| Static INT8 | 68.72% |
| Dynamic INT8 SW | 71.19% |
| Dynamic INT8 FPGA | 70.19% |

Six pipeline instances, 234.3 MHz, and 19.16 frame/s ? as reported in the paper.

| U280 resource | Utilization |
|---|---|
| LUT | 50.48% |
| FF | 33.54% |
| DSP | 68.31% |
| BRAM | 68.61% |
| URAM | 54.38% |

## Paper results and archive

| Paper item | Located implementation/configuration | Reproduction status |
|---|---|---|
| Table 1 | selected host/kernels; CPU temporal and output head | TODO: ground-truth evaluation and external full-model baselines |
| Table 2 | selected six-instance host and U280 placement/HBM config | TODO: other experiment configurations and measured logs |
| Table 3 | TODO | comparison sources/scripts not supplied |
| Fig. 4 | selected placement constraints | TODO: original post-route image/checkpoint |

Original measurement logs and reproduction artifacts: TODO.

[archive/README.md](archive/README.md) explains retained versions. Historical Tcl, tests, configurations and data refer to their original project structure; they are not current build instructions.

Repository-wide license: TODO. Preserve existing source notices. Contact: repository owner contact information ? TODO.
