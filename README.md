# A Quantization-Aware Frame-Pipelined FPGA Accelerator for Video Transformers

FPGA Spatial Encoder acceleration for ViViT Factorised Encoder, with CPU temporal processing and a 400-class output head. Submitted to ASP-DAC '27. Citation/DOI: TODO.

## Selected implementation

The final six-instance host and link configuration come from main `5504ce4` (`paper-submission-main`). Hardware kernels come from Decrease_params `eff3f5e` (`paper-submission-kernels`). The main four-kernel v0 snapshot is preserved in archive. See [MIGRATION.md](MIGRATION.md) for full revisions, paths and line-by-line cleanup edits.

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
hw/          selected HLS kernels; exported symbols unchanged
host/        final host, CPU temporal encoder and support code
configs/     six-instance U280 link/SLR/HBM configuration
tb/          four standalone single-kernel OpenCL harnesses
params/      original parameters and tensor fixtures, retained in place
archive/     main kernel v0, historical hosts and earlier experiments
MIGRATION.md source revisions, complete path map and exact content edits
```

The local untracked `spatial_encoder/host_Q_head_debugV10_tmp.cpp`, if present, is user-managed and not part of the selected application. No empty sw_model/, scripts/ or results/ directories are created.

## Inputs and software scope

The host consumes precomputed encoder-input tensors. Raw-video preprocessing, tubelet/spatial embedding, and full FP32 / Static INT8 / Dynamic INT8 software baselines are maintained outside this repository. This checkout does not provide a raw-video inference pipeline.

The batch host pairs exported encoder inputs with 400-value `ViViT_Output` vectors and compares their argmax with the inferred class. The meaning/provenance of `ViViT_Output` is **TODO**; the printed success rate must not be presented as ground-truth Kinetics-400 Top-1 without that information. Dataset/export instructions and model provenance: TODO.

## Build and run

Required source-level dependencies include Vitis HLS headers, OpenCL/Xilinx host headers, POSIX facilities and OpenMP. Exact Vitis/Vivado/XRT/compiler versions and installation instructions: TODO. The platform identifier above is taken from the selected config, not inferred as a tool version.

The author supplied these build commands:

```sh
make -C spatial_v3_kernels/Hardware
make -C spatial_v3_system_hw_link/Hardware
make -C spatial_v3/Hardware
```

**These Vitis project directories are not included in this repository.** The commands were not executed during cleanup. Their external project source references must be updated using MIGRATION.md before rebuilding the reorganized sources; those external files have not been changed here.

Run: **TODO — actual invocation of the final host_6CU application.** The existing host's absolute parameter/batch-export paths and optional CLI input behavior need an environment-specific invocation. No unverified build, simulation or run command is supplied here.

## Known Issues

- Imported `mha_attention.h` has an inconsistent include guard and declares an older `multihead_attention` top instead of the implementation's `mha_attention` interface.
- Imported `ffn1.h` references undefined `MLP_DIM_package` in FFN2 helper declarations; some FFN1 helper declarations conflict with static definitions.
- `ffn2.cpp` calls helpers before their declarations/definitions; the header only declares its top kernel.
- `host/help_functions.h` uses unqualified `string`; standalone harnesses include it before their namespace directive. This pre-existing declaration/include-order issue is not repaired.
- Host model and batch-export paths remain environment-specific. Raw-video preprocessing and full software baselines are external. `ViViT_Output` semantics remain TODO.
- CPU quantization/accumulation details differ from FPGA implementation details. Cleanup does not alter these or claim bitwise equivalence.
- Static CU/argument/port/SLR/HBM and local-include checks do not prove compilation, C-sim, co-sim, hardware operation, accuracy or routed frequency. No FPGA artifacts were regenerated.

All header issues are preserved per author instruction; see [MIGRATION.md](MIGRATION.md) for specific locations. HLS/OpenCL system headers are external prerequisites, not repository-local include targets.

## Paper results and archive

| Paper item | Located implementation/configuration | Reproduction status |
|---|---|---|
| Table 1 | selected host/kernels; CPU temporal and output head | TODO: ground-truth evaluation and external full-model baselines |
| Table 2 | selected six-instance host and U280 placement/HBM config | TODO: other experiment configurations and measured logs |
| Table 3 | TODO | comparison sources/scripts not supplied |
| Fig. 4 | selected placement constraints | TODO: original post-route image/checkpoint |

No performance/accuracy/resource figures are inferred from the sources. Original paper measurements and artifacts: TODO.

[archive/README.md](archive/README.md) explains retained versions. Historical Tcl, tests, configurations and data are kept byte-for-byte and still refer to their original project structure; they are not current build instructions. The ignored projection test executable remains untracked.

Repository-wide license: TODO. Preserve existing source notices. Contact channel: TODO.
