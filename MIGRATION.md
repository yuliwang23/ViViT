# Migration record

## Authorized source revisions

| Component | Source | Pinned commit | Local tag |
|---|---|---|---|
| Final host, host support, six-instance config and parameters | main | `5504ce464df9a42c1fedbae7dfa9a29903df2278` | `paper-submission-main` |
| Active kernels and their standalone hardware harnesses | Decrease_params | `eff3f5e783948c48877d99e6a124ceedf3976a26` | `paper-submission-kernels` |

Both tags are local; no branch was reset or rewritten and nothing was pushed. Work is on `refactor/cleanup-main`. Windows resolves the requested `Decrease_params` ref to the existing local `decrease_params`; its commit equals fetched `origin/Decrease_params` and the kernel tag. The old branches remain unchanged.

Before importing, compare implementation signatures, argument order/count, kernel/CU names, link ports and banks against the final host/config. This check passed: MSA-Attn 14 arguments, MSA-Proj 7, FFN1 8, FFN2 7; 24 named CUs and 216 mapped ports. Header declarations are separately documented below and are not treated as repaired by this check.

## Commit sequence and preservation

1. Add this migration record.
2. Preserve all 12 main-v0 kernel/harness files using `git mv four_kernel_version archive/legacy_versions/four_kernel_v0/four_kernel_version`, then import with `git checkout Decrease_params -- four_kernel_version`. Commit the preservation and import together, separately from later moves. Git may display v0 preservation as additions because the original paths are immediately reintroduced by checkout; the move command and full v0 snapshot are retained.
3. Move ACTIVE hardware, host, configuration and supporting harness files using `git mv`; require 100% rename for every existing file.
4. Archive historical sources/tests/configs/assets using `git mv`; add explanatory archive READMEs. Every moved existing file must be a 100% rename; README additions are the authorized documentation exception.
5. Correct only necessary active include/editor paths, record every changed line here, and update the root README with scope, supplied external build commands and Known Issues.

After each commit inspect `git diff -M --stat HEAD^ HEAD` and rename status. The author clarified that new documentation, source import and necessary path corrections are allowed content changes; an unexpected content change during a move must stop the task.

The untracked `spatial_encoder/host_Q_head_debugV10_tmp.cpp` is untouched and unstaged. The ignored `projection_out/projection_test.exe` follows its directory into archive without being added to Git. `params/` stays in place. No empty sw_model/, scripts/ or results/ directory is introduced. Historical Tcl and tests retain original contents and paths; they are not repaired.

## Existing header and build issues (not repaired)

These observations were rechecked against the imported kernel revision `eff3f5e783948c48877d99e6a124ceedf3976a26`. Relocation and include-spelling changes do not repair them.

- `hw/mha_attention/mha_attention.h:1`: guard checks `MHA_ATTENTION_H` but defines `MMA_ATTENTION_H`; line 255 declares the old `multihead_attention` interface instead of the implementation's `mha_attention` interface.
- `hw/ffn1/ffn1.h:251`: FFN2 helper declarations reference `MLP_DIM_package` without defining it. Several FFN1 helpers have non-static header declarations but static implementation definitions.
- `hw/ffn2/ffn2.cpp:95`: helpers are called before their definitions; `ffn2.h` declares the top but not these helpers.
- `host/help_functions.h` uses unqualified `string` in `run_custom_profiling`; the single-kernel harnesses include it before their `using namespace std`. No declaration or include-order repair is included.
- Hardware compilation, simulation and execution have not been demonstrated by this cleanup. HLS/OpenCL/system headers require the external toolchain.

The active host's existing absolute model/batch-input paths are preserved because `params/` is not relocated and the external dataset is absent. The reviewed README records the supplied invocation and identifies ViViT_Output as external FP32 Scenic/Flax/JAX logits. The old conflicted link config is archived, not repaired.

## Content/path edits

Only the changes below were made to existing imported/selected source and editor files after relocation. Line numbers are unchanged because every edit replaces exactly one line. Documentation additions/updates are listed separately.

| File | Line | Old | New |
|---|---|---|---|
| `hw/ffn1/ffn1.cpp` | 1 | `#include "FFN1.h"` | `#include "ffn1.h"` |
| `hw/ffn2/ffn2.cpp` | 1 | `#include "FFN2.h"` | `#include "ffn2.h"` |
| `tb/mha_attention/host.cpp` | 20 | `#include "help_functions.h"` | `#include "../../host/help_functions.h"` |
| `tb/mha_attention/host.cpp` | 21 | `#include "mha_attention.h"` | `#include "../../hw/mha_attention/mha_attention.h"` |
| `tb/mha_output_projection/host.cpp` | 20 | `#include "help_functions.h"` | `#include "../../host/help_functions.h"` |
| `tb/mha_output_projection/host.cpp` | 21 | `#include "mha_output_projection.h"` | `#include "../../hw/mha_output_projection/mha_output_projection.h"` |
| `tb/ffn1/host.cpp` | 20 | `#include "help_functions.h"` | `#include "../../host/help_functions.h"` |
| `tb/ffn1/host.cpp` | 21 | `#include "FFN1.h"` | `#include "../../hw/ffn1/ffn1.h"` |
| `tb/ffn2/host.cpp` | 20 | `#include "help_functions.h"` | `#include "../../host/help_functions.h"` |
| `tb/ffn2/host.cpp` | 21 | `#include "FFN2.h"` | `#include "../../hw/ffn2/ffn2.h"` |
| `.vscode/launch.json` | 11 | `"cwd": "<original-local-checkout>/spatial_encoder",` | `"cwd": "${workspaceFolder}/host",` |
| `.vscode/launch.json` | 12 | `"program": "<original-local-checkout>/spatial_encoder/build/Debug/outDebug",` | `"program": "${workspaceFolder}/host/build/Debug/outDebug",` |

No host runtime input/parameter paths, exported kernel symbols, CU names, argument order, SLR assignments or HBM indices were changed. FFN source/header filenames alone were lowercased. The root README was filled from its blank main version; archive READMEs and this migration record are newly authored documentation.

## Static verification

Completed source/host/config checks: 24 CUs, 216 mapped ports, argument counts 14/7/8/7, matching bank arrays and two pipelines per SLR. All 22 active quoted includes resolve with exact path spelling. No case-insensitive path collisions.

Final file accounting: 841 original filesystem files + 12 imported/preserved kernel snapshot files + 1 MIGRATION.md + 44 new archive READMEs = 898 files (excluding .git). Root README existed and was updated, not counted as a new file. All 575 params files remain in place and byte-identical. All original archived files are byte-identical, including main-v0 kernel sources. The ignored executable follows its original module without becoming tracked; the local untracked host's bytes/path are unchanged. Old branch tips and both source tags are unchanged.

Archived quoted includes with no local resolution: 60. They are reported as historical limitations, not repaired or treated as active dependencies. External HLS/OpenCL/system headers are not supplied by this repository. No compiler, C-simulation, co-simulation, synthesis, FPGA run or accuracy evaluation was executed.

12 necessary include/editor-path occurrences were changed; the table above enumerates each. Known header issues remain unchanged. Run and ViViT_Output semantics remain TODO.

## Complete source-to-destination map

Source revision disambiguates the two copies of four_kernel_version. KEEP/HOLD rows are intentional. All existing files, including the ignored executable and local untracked host, are covered. The two kernel snapshots produce 853 preserved/imported file destinations before new documentation.

| Source revision | Old path | New path | Action |
|---|---|---|---|
| `5504ce464df9` | `.gitignore` | `.gitignore` | KEEP |
| `5504ce464df9` | `.vscode/c_cpp_properties.json` | `.vscode/c_cpp_properties.json` | KEEP |
| `5504ce464df9` | `.vscode/launch.json` | `.vscode/launch.json` | KEEP |
| `5504ce464df9` | `.vscode/settings.json` | `.vscode/settings.json` | KEEP |
| `5504ce464df9` | `README.md` | `README.md` | KEEP |
| `5504ce464df9` | `attention/2head/Host_test_2head.cpp` | `archive/experimental/attention/2head/Host_test_2head.cpp` | git mv |
| `5504ce464df9` | `attention/2head/link_2head.cfg` | `archive/experimental/attention/2head/link_2head.cfg` | git mv |
| `5504ce464df9` | `attention/2head/xrt_2head_hw_emu.ini` | `archive/experimental/attention/2head/xrt_2head_hw_emu.ini` | git mv |
| `5504ce464df9` | `attention/attention.cpp` | `archive/experimental/attention/attention.cpp` | git mv |
| `5504ce464df9` | `attention/attention.h` | `archive/experimental/attention/attention.h` | git mv |
| `5504ce464df9` | `attention/flash_attention_host.cpp` | `archive/experimental/attention/flash_attention_host.cpp` | git mv |
| `5504ce464df9` | `attention/host.cpp` | `archive/experimental/attention/host.cpp` | git mv |
| `5504ce464df9` | `attention/run_hls.tcl` | `archive/experimental/attention/run_hls.tcl` | git mv |
| `5504ce464df9` | `exp_function/exp.cpp` | `archive/experimental/exp_function/exp.cpp` | git mv |
| `5504ce464df9` | `exp_function/exp.h` | `archive/experimental/exp_function/exp.h` | git mv |
| `5504ce464df9` | `exp_function/run_hls.tcl` | `archive/experimental/exp_function/run_hls.tcl` | git mv |
| `5504ce464df9` | `four_kernel_version/FFN1/FFN1.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN1/FFN1.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN1/FFN1.cpp` | `hw/ffn1/ffn1.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/FFN1/FFN1.h` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN1/FFN1.h` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN1/FFN1.h` | `hw/ffn1/ffn1.h` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/FFN1/host.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN1/host.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN1/host.cpp` | `tb/ffn1/host.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/FFN2/FFN2.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN2/FFN2.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN2/FFN2.cpp` | `hw/ffn2/ffn2.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/FFN2/FFN2.h` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN2/FFN2.h` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN2/FFN2.h` | `hw/ffn2/ffn2.h` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/FFN2/host.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/FFN2/host.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/FFN2/host.cpp` | `tb/ffn2/host.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_attention/host.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_attention/host.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_attention/host.cpp` | `tb/mha_attention/host.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_attention/mha_attention.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_attention/mha_attention.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_attention/mha_attention.cpp` | `hw/mha_attention/mha_attention.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_attention/mha_attention.h` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_attention/mha_attention.h` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_attention/mha_attention.h` | `hw/mha_attention/mha_attention.h` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_output_projection/host.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_output_projection/host.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_output_projection/host.cpp` | `tb/mha_output_projection/host.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_output_projection/mha_output_projection.cpp` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_output_projection/mha_output_projection.cpp` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_output_projection/mha_output_projection.cpp` | `hw/mha_output_projection/mha_output_projection.cpp` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `four_kernel_version/mha_output_projection/mha_output_projection.h` | `archive/legacy_versions/four_kernel_v0/four_kernel_version/mha_output_projection/mha_output_projection.h` | preserve main v0 with git mv before checkout; import commit |
| `eff3f5e78394` | `four_kernel_version/mha_output_projection/mha_output_projection.h` | `hw/mha_output_projection/mha_output_projection.h` | git checkout Decrease_params, then git mv; active/imported harness |
| `5504ce464df9` | `layernorm/hbm.cfg` | `archive/experimental/layernorm/hbm.cfg` | git mv |
| `5504ce464df9` | `layernorm/host.cpp` | `archive/experimental/layernorm/host.cpp` | git mv |
| `5504ce464df9` | `layernorm/host_hbm.cpp` | `archive/experimental/layernorm/host_hbm.cpp` | git mv |
| `5504ce464df9` | `layernorm/layernorm.cpp` | `archive/experimental/layernorm/layernorm.cpp` | git mv |
| `5504ce464df9` | `layernorm/layernorm.h` | `archive/experimental/layernorm/layernorm.h` | git mv |
| `5504ce464df9` | `layernorm/layernorm_test_scaler.cpp` | `archive/experimental/layernorm/layernorm_test_scaler.cpp` | git mv |
| `5504ce464df9` | `layernorm/output_out.txt` | `archive/experimental/layernorm/output_out.txt` | git mv |
| `5504ce464df9` | `layernorm/output_scaler.txt` | `archive/experimental/layernorm/output_scaler.txt` | git mv |
| `5504ce464df9` | `layernorm/run_hls.tcl` | `archive/experimental/layernorm/run_hls.tcl` | git mv |
| `5504ce464df9` | `linear_qkv/help_functions.cpp` | `archive/experimental/linear_qkv/help_functions.cpp` | git mv |
| `5504ce464df9` | `linear_qkv/help_functions.h` | `archive/experimental/linear_qkv/help_functions.h` | git mv |
| `5504ce464df9` | `linear_qkv/host.cpp` | `archive/experimental/linear_qkv/host.cpp` | git mv |
| `5504ce464df9` | `linear_qkv/linear_qkv_multihead.cpp` | `archive/experimental/linear_qkv/linear_qkv_multihead.cpp` | git mv |
| `5504ce464df9` | `linear_qkv/linear_qkv_multihead.h` | `archive/experimental/linear_qkv/linear_qkv_multihead.h` | git mv |
| `5504ce464df9` | `linear_qkv/run_hls.tcl` | `archive/experimental/linear_qkv/run_hls.tcl` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wk.txt` | `archive/experimental/linear_qkv_quantize/Wk.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wk_transpose.txt` | `archive/experimental/linear_qkv_quantize/Wk_transpose.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wq.txt` | `archive/experimental/linear_qkv_quantize/Wq.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wq_transpose.txt` | `archive/experimental/linear_qkv_quantize/Wq_transpose.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wv.txt` | `archive/experimental/linear_qkv_quantize/Wv.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/Wv_transpose.txt` | `archive/experimental/linear_qkv_quantize/Wv_transpose.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/X.txt` | `archive/experimental/linear_qkv_quantize/X.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/bk.txt` | `archive/experimental/linear_qkv_quantize/bk.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/bq.txt` | `archive/experimental/linear_qkv_quantize/bq.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/bv.txt` | `archive/experimental/linear_qkv_quantize/bv.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/hbm.cfg` | `archive/experimental/linear_qkv_quantize/hbm.cfg` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/host.cpp` | `archive/experimental/linear_qkv_quantize/host.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/host_hbm.cpp` | `archive/experimental/linear_qkv_quantize/host_hbm.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_quan.cpp` | `archive/experimental/linear_qkv_quantize/linear_qkv_multihead_quan.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_quan.h` | `archive/experimental/linear_qkv_quantize/linear_qkv_multihead_quan.h` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_quan_test.cpp` | `archive/experimental/linear_qkv_quantize/linear_qkv_multihead_quan_test.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_quan_wrong.cpp` | `archive/legacy_versions/linear_qkv_quantize/linear_qkv_multihead_quan_wrong.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_quan_wrong.h` | `archive/legacy_versions/linear_qkv_quantize/linear_qkv_multihead_quan_wrong.h` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_multihead_random_test.cpp` | `archive/experimental/linear_qkv_quantize/linear_qkv_multihead_random_test.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/linear_qkv_quan_generate.cpp` | `archive/experimental/linear_qkv_quantize/linear_qkv_quan_generate.cpp` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/scale_input.txt` | `archive/experimental/linear_qkv_quantize/scale_input.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/scale_wk.txt` | `archive/experimental/linear_qkv_quantize/scale_wk.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/scale_wq.txt` | `archive/experimental/linear_qkv_quantize/scale_wq.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/scale_wv.txt` | `archive/experimental/linear_qkv_quantize/scale_wv.txt` | git mv |
| `5504ce464df9` | `linear_qkv_quantize/transpose.cpp` | `archive/experimental/linear_qkv_quantize/transpose.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/host.cpp` | `archive/experimental/linear_with_layernorm/host.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wk.txt` | `archive/experimental/linear_with_layernorm/input_data/Wk.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wk_transposed.txt` | `archive/experimental/linear_with_layernorm/input_data/Wk_transposed.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wq.txt` | `archive/experimental/linear_with_layernorm/input_data/Wq.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wq_transposed.txt` | `archive/experimental/linear_with_layernorm/input_data/Wq_transposed.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wv.txt` | `archive/experimental/linear_with_layernorm/input_data/Wv.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/Wv_transposed.txt` | `archive/experimental/linear_with_layernorm/input_data/Wv_transposed.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/beta.txt` | `archive/experimental/linear_with_layernorm/input_data/beta.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/bk.txt` | `archive/experimental/linear_with_layernorm/input_data/bk.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/bq.txt` | `archive/experimental/linear_with_layernorm/input_data/bq.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/bv.txt` | `archive/experimental/linear_with_layernorm/input_data/bv.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/gamma.txt` | `archive/experimental/linear_with_layernorm/input_data/gamma.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/input_x.txt` | `archive/experimental/linear_with_layernorm/input_data/input_x.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/scale_wk.txt` | `archive/experimental/linear_with_layernorm/input_data/scale_wk.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/scale_wq.txt` | `archive/experimental/linear_with_layernorm/input_data/scale_wq.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/input_data/scale_wv.txt` | `archive/experimental/linear_with_layernorm/input_data/scale_wv.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/linear_qkv_multihead.cpp` | `archive/experimental/linear_with_layernorm/linear_qkv_multihead.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/linear_qkv_multihead.h` | `archive/experimental/linear_with_layernorm/linear_qkv_multihead.h` | git mv |
| `5504ce464df9` | `linear_with_layernorm/linear_qkv_multihead_test.cpp` | `archive/experimental/linear_with_layernorm/linear_qkv_multihead_test.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/K_output.txt` | `archive/experimental/linear_with_layernorm/output_check/K_output.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/K_output_golden.txt` | `archive/experimental/linear_with_layernorm/output_check/K_output_golden.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/Q_output.txt` | `archive/experimental/linear_with_layernorm/output_check/Q_output.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/Q_output_golden.txt` | `archive/experimental/linear_with_layernorm/output_check/Q_output_golden.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/V_output.txt` | `archive/experimental/linear_with_layernorm/output_check/V_output.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/V_output_golden.txt` | `archive/experimental/linear_with_layernorm/output_check/V_output_golden.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/check_result.cpp` | `archive/experimental/linear_with_layernorm/output_check/check_result.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/golden_qkv_gen.cpp` | `archive/experimental/linear_with_layernorm/output_check/golden_qkv_gen.cpp` | git mv |
| `5504ce464df9` | `linear_with_layernorm/output_check/verification_log.txt` | `archive/experimental/linear_with_layernorm/output_check/verification_log.txt` | git mv |
| `5504ce464df9` | `linear_with_layernorm/run_hls.tcl` | `archive/experimental/linear_with_layernorm/run_hls.tcl` | git mv |
| `5504ce464df9` | `mlp/connectivity.cfg` | `archive/experimental/mlp/connectivity.cfg` | git mv |
| `5504ce464df9` | `mlp/fast_math.h` | `archive/experimental/mlp/fast_math.h` | git mv |
| `5504ce464df9` | `mlp/gelu_lut.h` | `archive/experimental/mlp/gelu_lut.h` | git mv |
| `5504ce464df9` | `mlp/host.cpp` | `archive/experimental/mlp/host.cpp` | git mv |
| `5504ce464df9` | `mlp/input_data/W1_q.txt` | `archive/experimental/mlp/input_data/W1_q.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/W2_q.txt` | `archive/experimental/mlp/input_data/W2_q.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/b1.txt` | `archive/experimental/mlp/input_data/b1.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/b2.txt` | `archive/experimental/mlp/input_data/b2.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/beta.txt` | `archive/experimental/mlp/input_data/beta.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/gamma.txt` | `archive/experimental/mlp/input_data/gamma.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/gen_input.cpp` | `archive/experimental/mlp/input_data/gen_input.cpp` | git mv |
| `5504ce464df9` | `mlp/input_data/golden_gen.cpp` | `archive/experimental/mlp/input_data/golden_gen.cpp` | git mv |
| `5504ce464df9` | `mlp/input_data/golden_output.txt` | `archive/experimental/mlp/input_data/golden_output.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/input.txt` | `archive/experimental/mlp/input_data/input.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/scale_w1.txt` | `archive/experimental/mlp/input_data/scale_w1.txt` | git mv |
| `5504ce464df9` | `mlp/input_data/scale_w2.txt` | `archive/experimental/mlp/input_data/scale_w2.txt` | git mv |
| `5504ce464df9` | `mlp/mlp.cpp` | `archive/experimental/mlp/mlp.cpp` | git mv |
| `5504ce464df9` | `mlp/mlp.h` | `archive/experimental/mlp/mlp.h` | git mv |
| `5504ce464df9` | `mlp/mlp_table.cpp` | `archive/experimental/mlp/mlp_table.cpp` | git mv |
| `5504ce464df9` | `mlp/mlp_table.h` | `archive/experimental/mlp/mlp_table.h` | git mv |
| `5504ce464df9` | `mlp/mlp_testbench.cpp` | `archive/experimental/mlp/mlp_testbench.cpp` | git mv |
| `5504ce464df9` | `mlp/mlp_testbench_files.cpp` | `archive/experimental/mlp/mlp_testbench_files.cpp` | git mv |
| `5504ce464df9` | `mlp/output_check/compare_outputs.cpp` | `archive/experimental/mlp/output_check/compare_outputs.cpp` | git mv |
| `5504ce464df9` | `mlp/output_check/golden_output.txt` | `archive/experimental/mlp/output_check/golden_output.txt` | git mv |
| `5504ce464df9` | `mlp/output_check/output.txt` | `archive/experimental/mlp/output_check/output.txt` | git mv |
| `5504ce464df9` | `mlp/output_check/output_sigmoid.txt` | `archive/experimental/mlp/output_check/output_sigmoid.txt` | git mv |
| `5504ce464df9` | `mlp/software/mlp_software.cpp` | `archive/experimental/mlp/software/mlp_software.cpp` | git mv |
| `5504ce464df9` | `mlp/software/mlp_software.h` | `archive/experimental/mlp/software/mlp_software.h` | git mv |
| `5504ce464df9` | `mlp/software/tensor_ops_software.cpp` | `archive/experimental/mlp/software/tensor_ops_software.cpp` | git mv |
| `5504ce464df9` | `mlp/software/tensor_ops_software.h` | `archive/experimental/mlp/software/tensor_ops_software.h` | git mv |
| `5504ce464df9` | `multihead_attention/host.cpp` | `archive/experimental/multihead_attention/host.cpp` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/Wk_transposed.txt` | `archive/experimental/multihead_attention/input_data/Wk_transposed.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/Wo_transposed.txt` | `archive/experimental/multihead_attention/input_data/Wo_transposed.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/Wq_transposed.txt` | `archive/experimental/multihead_attention/input_data/Wq_transposed.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/Wv_transposed.txt` | `archive/experimental/multihead_attention/input_data/Wv_transposed.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/beta.txt` | `archive/experimental/multihead_attention/input_data/beta.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/bk.txt` | `archive/experimental/multihead_attention/input_data/bk.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/bo.txt` | `archive/experimental/multihead_attention/input_data/bo.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/bq.txt` | `archive/experimental/multihead_attention/input_data/bq.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/bv.txt` | `archive/experimental/multihead_attention/input_data/bv.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/gamma.txt` | `archive/experimental/multihead_attention/input_data/gamma.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/input_x.txt` | `archive/experimental/multihead_attention/input_data/input_x.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/scale_wk.txt` | `archive/experimental/multihead_attention/input_data/scale_wk.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/scale_wo.txt` | `archive/experimental/multihead_attention/input_data/scale_wo.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/scale_wq.txt` | `archive/experimental/multihead_attention/input_data/scale_wq.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/scale_wv.txt` | `archive/experimental/multihead_attention/input_data/scale_wv.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/wk.txt` | `archive/experimental/multihead_attention/input_data/wk.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/wo.txt` | `archive/experimental/multihead_attention/input_data/wo.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/wq.txt` | `archive/experimental/multihead_attention/input_data/wq.txt` | git mv |
| `5504ce464df9` | `multihead_attention/input_data/wv.txt` | `archive/experimental/multihead_attention/input_data/wv.txt` | git mv |
| `5504ce464df9` | `multihead_attention/multihead_attention.cpp` | `archive/experimental/multihead_attention/multihead_attention.cpp` | git mv |
| `5504ce464df9` | `multihead_attention/multihead_attention.h` | `archive/experimental/multihead_attention/multihead_attention.h` | git mv |
| `5504ce464df9` | `multihead_attention/multihead_attention_test.cpp` | `archive/experimental/multihead_attention/multihead_attention_test.cpp` | git mv |
| `5504ce464df9` | `multihead_attention/output_data/Out_output.txt` | `archive/experimental/multihead_attention/output_data/Out_output.txt` | git mv |
| `5504ce464df9` | `multihead_attention/output_data/compare_results.py` | `archive/experimental/multihead_attention/output_data/compare_results.py` | git mv |
| `5504ce464df9` | `multihead_attention/output_data/gloden_MLP_in.txt` | `archive/experimental/multihead_attention/output_data/gloden_MLP_in.txt` | git mv |
| `5504ce464df9` | `multihead_attention/output_data/gloden_output.txt` | `archive/experimental/multihead_attention/output_data/gloden_output.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_input.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_input.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_output.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_output.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_input.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_input.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_output.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_output.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__encoder_output.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__encoder_output.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt` | git mv |
| `5504ce464df9` | `multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_2.txt` | `archive/experimental/multihead_attention/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_2.txt` | git mv |
| `5504ce464df9` | `multihead_attention/run_hls.tcl` | `archive/experimental/multihead_attention/run_hls.tcl` | git mv |
| `5504ce464df9` | `multilayer_perceptron/host.cpp` | `archive/experimental/multilayer_perceptron/host.cpp` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/W1.txt` | `archive/experimental/multilayer_perceptron/input_data/W1.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/W1_transposed.txt` | `archive/experimental/multilayer_perceptron/input_data/W1_transposed.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/W2.txt` | `archive/experimental/multilayer_perceptron/input_data/W2.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/W2_transposed.txt` | `archive/experimental/multilayer_perceptron/input_data/W2_transposed.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/b1.txt` | `archive/experimental/multilayer_perceptron/input_data/b1.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/b2.txt` | `archive/experimental/multilayer_perceptron/input_data/b2.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/beta.txt` | `archive/experimental/multilayer_perceptron/input_data/beta.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/gamma.txt` | `archive/experimental/multilayer_perceptron/input_data/gamma.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/input_x.txt` | `archive/experimental/multilayer_perceptron/input_data/input_x.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/scale_w1.txt` | `archive/experimental/multilayer_perceptron/input_data/scale_w1.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/input_data/scale_w2.txt` | `archive/experimental/multilayer_perceptron/input_data/scale_w2.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/mlp.cpp` | `archive/experimental/multilayer_perceptron/mlp.cpp` | git mv |
| `5504ce464df9` | `multilayer_perceptron/mlp.h` | `archive/experimental/multilayer_perceptron/mlp.h` | git mv |
| `5504ce464df9` | `multilayer_perceptron/mlp_test.cpp` | `archive/experimental/multilayer_perceptron/mlp_test.cpp` | git mv |
| `5504ce464df9` | `multilayer_perceptron/output_data/compare_results.py` | `archive/experimental/multilayer_perceptron/output_data/compare_results.py` | git mv |
| `5504ce464df9` | `multilayer_perceptron/output_data/gloden_output.txt` | `archive/experimental/multilayer_perceptron/output_data/gloden_output.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_input.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_input.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_output.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MHA_output.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_input.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_input.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_output.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__MLP_output.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__encoder_output.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__encoder_output.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_1.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_2.txt` | `archive/experimental/multilayer_perceptron/pattern/sample2_intermediate_SpatialTransformer__encoderblock_0__layer_norm_input_2.txt` | git mv |
| `5504ce464df9` | `multilayer_perceptron/run_hls.tcl` | `archive/experimental/multilayer_perceptron/run_hls.tcl` | git mv |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoder_norm/bias.txt` | `params/all_params/SpatialTransformer/encoder_norm/bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoder_norm/scale.txt` | `params/all_params/SpatialTransformer/encoder_norm/scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_0_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_1_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/SpatialTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/SpatialTransformer/posembed_input/pos_embedding.txt` | `params/all_params/SpatialTransformer/posembed_input/pos_embedding.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoder_norm/bias.txt` | `params/all_params/TemporalTransformer/encoder_norm/bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoder_norm/scale.txt` | `params/all_params/TemporalTransformer/encoder_norm/scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_0/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_1/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_10/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_11/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_2/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_3/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_4/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_5/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_6/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_7/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_8/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_0_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_0_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_1_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/LayerNorm_1_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_0_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MlpBlock_0_Dense_1_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_key_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_out_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_query_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_bias.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | `params/all_params/TemporalTransformer/encoderblock_9/MultiHeadDotProductAttention_0_value_kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/all_params/TemporalTransformer/posembed_input/pos_embedding.txt` | `params/all_params/TemporalTransformer/posembed_input/pos_embedding.txt` | KEEP |
| `5504ce464df9` | `params/all_params/cls_TemporalTransformer.txt` | `params/all_params/cls_TemporalTransformer.txt` | KEEP |
| `5504ce464df9` | `params/all_params/embedding/bias.txt` | `params/all_params/embedding/bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/embedding/kernel.txt` | `params/all_params/embedding/kernel.txt` | KEEP |
| `5504ce464df9` | `params/all_params/output_projection/bias.txt` | `params/all_params/output_projection/bias.txt` | KEEP |
| `5504ce464df9` | `params/all_params/output_projection/kernel_qvalue.txt` | `params/all_params/output_projection/kernel_qvalue.txt` | KEEP |
| `5504ce464df9` | `params/all_params/output_projection/kernel_scale.txt` | `params/all_params/output_projection/kernel_scale.txt` | KEEP |
| `5504ce464df9` | `params/counting_num` | `params/counting_num` | KEEP |
| `5504ce464df9` | `params/counting_num.cpp` | `params/counting_num.cpp` | KEEP |
| `5504ce464df9` | `params/intermediate/README.md` | `params/intermediate/README.md` | KEEP |
| `5504ce464df9` | `params/intermediate/encoder_block.txt` | `params/intermediate/encoder_block.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample10_output.txt` | `params/intermediate/sample10_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample1_output.txt` | `params/intermediate/sample1_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample2_output.txt` | `params/intermediate/sample2_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample3_output.txt` | `params/intermediate/sample3_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample4_output.txt` | `params/intermediate/sample4_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample5_output.txt` | `params/intermediate/sample5_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample6_output.txt` | `params/intermediate/sample6_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample7_output.txt` | `params/intermediate/sample7_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample8_output.txt` | `params/intermediate/sample8_output.txt` | KEEP |
| `5504ce464df9` | `params/intermediate/sample9_output.txt` | `params/intermediate/sample9_output.txt` | KEEP |
| `5504ce464df9` | `params/params_info.txt` | `params/params_info.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_0_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_1_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_1_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_2_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_2_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_3_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_3_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_4_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_4_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_5_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_5_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_6_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_6_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_7_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_7_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_8_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_8_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_9_in.txt` | `params/vivit_video_input_AfterConvAndEmbedding/sample_video_9_in.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_0_out.txt` | `params/vivit_video_output/sample_video_0_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_1_out.txt` | `params/vivit_video_output/sample_video_1_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_2_out.txt` | `params/vivit_video_output/sample_video_2_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_3_out.txt` | `params/vivit_video_output/sample_video_3_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_4_out.txt` | `params/vivit_video_output/sample_video_4_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_5_out.txt` | `params/vivit_video_output/sample_video_5_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_6_out.txt` | `params/vivit_video_output/sample_video_6_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_7_out.txt` | `params/vivit_video_output/sample_video_7_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_8_out.txt` | `params/vivit_video_output/sample_video_8_out.txt` | KEEP |
| `5504ce464df9` | `params/vivit_video_output/sample_video_9_out.txt` | `params/vivit_video_output/sample_video_9_out.txt` | KEEP |
| `5504ce464df9` | `projection_out/help_functions.cpp` | `archive/experimental/projection_out/help_functions.cpp` | git mv |
| `5504ce464df9` | `projection_out/help_functions.h` | `archive/experimental/projection_out/help_functions.h` | git mv |
| `5504ce464df9` | `projection_out/link.cfg` | `archive/experimental/projection_out/link.cfg` | git mv |
| `5504ce464df9` | `projection_out/projection.cpp` | `archive/experimental/projection_out/projection.cpp` | git mv |
| `5504ce464df9` | `projection_out/projection.h` | `archive/experimental/projection_out/projection.h` | git mv |
| `5504ce464df9` | `projection_out/projection_host.cpp` | `archive/experimental/projection_out/projection_host.cpp` | git mv |
| `5504ce464df9` | `projection_out/projection_test.cpp` | `archive/experimental/projection_out/projection_test.cpp` | git mv |
| `ignored` | `projection_out/projection_test.exe` | `archive/experimental/projection_out/projection_test.exe` | untracked binary move; remain ignored |
| `5504ce464df9` | `residual/host_envs/help_functions.cpp` | `archive/legacy_versions/residual/host_envs/help_functions.cpp` | git mv |
| `5504ce464df9` | `residual/host_envs/help_functions.h` | `archive/legacy_versions/residual/host_envs/help_functions.h` | git mv |
| `5504ce464df9` | `residual/host_envs/host.cpp` | `archive/legacy_versions/residual/host_envs/host.cpp` | git mv |
| `5504ce464df9` | `residual/host_envs/host.h` | `archive/legacy_versions/residual/host_envs/host.h` | git mv |
| `5504ce464df9` | `residual/host_envs/host2.cpp` | `archive/legacy_versions/residual/host_envs/host2.cpp` | git mv |
| `5504ce464df9` | `residual/host_envs/layer.cpp` | `archive/legacy_versions/residual/host_envs/layer.cpp` | git mv |
| `5504ce464df9` | `residual/host_envs/layer.h` | `archive/legacy_versions/residual/host_envs/layer.h` | git mv |
| `5504ce464df9` | `residual/host_envs/utility.cpp` | `archive/legacy_versions/residual/host_envs/utility.cpp` | git mv |
| `5504ce464df9` | `residual/host_envs/utility.h` | `archive/legacy_versions/residual/host_envs/utility.h` | git mv |
| `5504ce464df9` | `residual/residual.cpp` | `archive/experimental/residual/residual.cpp` | git mv |
| `5504ce464df9` | `residual/residual.h` | `archive/experimental/residual/residual.h` | git mv |
| `5504ce464df9` | `residual/residual_2.cpp` | `archive/experimental/residual/residual_2.cpp` | git mv |
| `5504ce464df9` | `residual/residual_2.h` | `archive/experimental/residual/residual_2.h` | git mv |
| `5504ce464df9` | `residual/residual_tb.cpp` | `archive/experimental/residual/residual_tb.cpp` | git mv |
| `5504ce464df9` | `softmax/exp.cpp` | `archive/experimental/softmax/exp.cpp` | git mv |
| `5504ce464df9` | `softmax/exp.h` | `archive/experimental/softmax/exp.h` | git mv |
| `5504ce464df9` | `softmax/run_hls.tcl` | `archive/experimental/softmax/run_hls.tcl` | git mv |
| `5504ce464df9` | `softmax/softmax.cpp` | `archive/experimental/softmax/softmax.cpp` | git mv |
| `5504ce464df9` | `softmax/softmax.h` | `archive/experimental/softmax/softmax.h` | git mv |
| `5504ce464df9` | `spatial_encoder/binary_container_1-link.cfg` | `configs/u280_6_instances.cfg` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/deprecated.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/deprecated.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/help_functions_deprecated.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/help_functions_deprecated.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/help_functions_deprecated.h` | `archive/legacy_versions/spatial_encoder/deprecated/help_functions_deprecated.h` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_Q_head_debugV2.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_Q_head_debugV2.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_Q_head_debugV3.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_Q_head_debugV3.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_Q_head_debugV4_ori.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_Q_head_debugV4_ori.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_optimized_v5.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_optimized_v5.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_quant.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_quant.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_quant_parallel_head.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_quant_parallel_head.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/deprecated/host_quant_parallel_head_debugging.cpp` | `archive/legacy_versions/spatial_encoder/deprecated/host_quant_parallel_head_debugging.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/help_functions.cpp` | `host/help_functions.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/help_functions.h` | `host/help_functions.h` | git mv |
| `5504ce464df9` | `spatial_encoder/host.h` | `host/host.h` | git mv |
| `5504ce464df9` | `spatial_encoder/host_6CU.cpp` | `host/main.cpp` | git mv |
| `untracked` | `spatial_encoder/host_Q_head_debugV10_tmp.cpp` | `spatial_encoder/host_Q_head_debugV10_tmp.cpp` | KEEP user local work |
| `5504ce464df9` | `spatial_encoder/host_Q_head_debugV6.cpp` | `archive/legacy_versions/spatial_encoder/host_Q_head_debugV6.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/host_Q_head_debugV7.cpp` | `archive/legacy_versions/spatial_encoder/host_Q_head_debugV7.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/host_quant_debug_version.cpp` | `archive/legacy_versions/spatial_encoder/host_quant_debug_version.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/layer.cpp` | `archive/legacy_versions/spatial_encoder/layer.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/layer.h` | `archive/legacy_versions/spatial_encoder/layer.h` | git mv |
| `5504ce464df9` | `spatial_encoder/link.cfg` | `archive/legacy_versions/spatial_encoder/link.cfg` | git mv |
| `5504ce464df9` | `spatial_encoder/old_version/host_Q_head_debugV4.cpp` | `archive/legacy_versions/spatial_encoder/old_version/host_Q_head_debugV4.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/old_version/host_Q_head_debugV5.cpp` | `archive/legacy_versions/spatial_encoder/old_version/host_Q_head_debugV5.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/temporal_layer.cpp` | `host/temporal_layer.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/temporal_layer.h` | `host/temporal_layer.h` | git mv |
| `5504ce464df9` | `spatial_encoder/utility.cpp` | `host/utility.cpp` | git mv |
| `5504ce464df9` | `spatial_encoder/utility.h` | `host/utility.h` | git mv |
| `5504ce464df9` | `testbench/attention/attention_generate_data.cpp` | `archive/experimental/testbench/attention/attention_generate_data.cpp` | git mv |
| `5504ce464df9` | `testbench/attention/attention_test.cpp` | `archive/experimental/testbench/attention/attention_test.cpp` | git mv |
| `5504ce464df9` | `testbench/attention/attention_v0_test.cpp` | `archive/experimental/testbench/attention/attention_v0_test.cpp` | git mv |
| `5504ce464df9` | `testbench/exp_test.cpp` | `archive/experimental/testbench/exp_test.cpp` | git mv |
| `5504ce464df9` | `testbench/flash_attention_test.cpp` | `archive/experimental/testbench/flash_attention_test.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/generate.cpp` | `archive/experimental/testbench/layernorm/generate.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/layernorm_test.cpp` | `archive/experimental/testbench/layernorm/layernorm_test.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/generate_input.cpp` | `archive/experimental/testbench/layernorm/sw/generate_input.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/host.cpp` | `archive/experimental/testbench/layernorm/sw/host.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/input_beta.txt` | `archive/experimental/testbench/layernorm/sw/input_beta.txt` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/input_gamma.txt` | `archive/experimental/testbench/layernorm/sw/input_gamma.txt` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/input_x.txt` | `archive/experimental/testbench/layernorm/sw/input_x.txt` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/layernorm_output.txt` | `archive/experimental/testbench/layernorm/sw/layernorm_output.txt` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/layernorm_test.cpp` | `archive/experimental/testbench/layernorm/sw/layernorm_test.cpp` | git mv |
| `5504ce464df9` | `testbench/layernorm/sw/layernorm_test_txt.cpp` | `archive/experimental/testbench/layernorm/sw/layernorm_test_txt.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/generate_data.cpp` | `archive/experimental/testbench/linear_qkv_multihead/generate_data.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/linear_qkv_multihead_software.cpp` | `archive/experimental/testbench/linear_qkv_multihead/linear_qkv_multihead_software.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/linear_qkv_multihead_software_test_txt.cpp` | `archive/experimental/testbench/linear_qkv_multihead/linear_qkv_multihead_software_test_txt.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/linear_qkv_multihead_test.cpp` | `archive/experimental/testbench/linear_qkv_multihead/linear_qkv_multihead_test.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/linear_qkv_test_host.cpp` | `archive/experimental/testbench/linear_qkv_multihead/linear_qkv_test_host.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/temp.cpp` | `archive/experimental/testbench/linear_qkv_multihead/temp.cpp` | git mv |
| `5504ce464df9` | `testbench/linear_qkv_multihead/temp_stream.cpp` | `archive/experimental/testbench/linear_qkv_multihead/temp_stream.cpp` | git mv |
| `5504ce464df9` | `testbench/softmax_test.cpp` | `archive/experimental/testbench/softmax_test.cpp` | git mv |

## README review follow-up

README now records the supplied paper results, tool versions, external build/run commands and FP32 reference semantics. Source inspection confirms sample-ID pairing and argument usage; MAE is computed only for argmax mismatches. No executable source was changed. Historical editor-path values above use a redacted checkout placeholder to avoid publishing local identities or absolute paths.
