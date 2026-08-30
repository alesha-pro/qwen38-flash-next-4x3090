# Source map

## Cyankiwi AWQ 4x3090 v1 integration

Source project: `qwen38-flash-next-cyankiwi-awq`
Commit: `7aaa2b8e1eae853c36605bb366c1a581869bdbd5`

Owned imports:

- common runtime overlays: `gpu_worker.py`, `multiproc_executor.py`,
  `qwen_gdn_linear_attn.py`, `shared_experts.py`;
- external PLE overlays: `ple_layer.py`, `ple_offload/worker.py`,
  `ple_offload/ple_external_source.py`;
- external PLE validator and manifest;
- AWQ thin-checkpoint builder, FP8 companion-projection builder, mandatory
  BF16 shared-expert-gate repair and BF16 GDN `in_proj_a` / `in_proj_b`
  restoration;
- split-projection model and hyperconnection overlays required by that
  derived checkpoint.

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
- cyankiwi AWQ revision: `01324cfa2c3f46948781fad30641ac360014e008`;
- external NVFP4/PLE revision:
  `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
