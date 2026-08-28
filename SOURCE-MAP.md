# Source map

## W4A16 integration base

Source project: `qwen38-quantize`  
Commit: `1f41c76d068de51c4612f8e86c0f1fcc9ce79352`

Owned imports:

- common runtime overlays: `gpu_worker.py`, `multiproc_executor.py`,
  `qwen_gdn_linear_attn.py`, `shared_experts.py`;
- external PLE overlays: `ple_layer.py`, `ple_offload/worker.py`,
  `ple_offload/ple_external_source.py`;
- external PLE validator and manifest;
- frozen W4A16 text/Vision request helpers and Phase 2 evidence.

`ple_layer.py` includes the proven external-scale repair. GPU ranks must load
the BF16 scalar `0.00019931793212890625` from the external checkpoint with
manifest validation. Missing scale must fail closed.

## FP8 QSA KV owner

Source project: `qwen38-kv-quantize`  
Commit: `01fc6179c91bbe7bba7d73e2798b43deb6979bf5`

Owned imports:

- `overlays/qwen38/qsa.py`;
- `overlays/qwen38/ops_qsa.py`;
- `overlays/qwen38/scales.json`;
- `overlays/qwen38/BASE.sha256`;
- QSA unit, kernel, quality and benchmark tests.

## Runtime and checkpoints

- Image digest:
  `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
- vendor vLLM: `0.1.dev20073+g8e685d198`;
- W4A16 revision: `9236d703b25f25eb5c17e9640204f84fa1ce0c6e`;
- external NVFP4/PLE revision:
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
