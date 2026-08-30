# Qwen3.8 Flash Next AWQ INT4 4x3090 v1

One-command vLLM recipe for Qwen3.8 Flash Next 122B / 51B active on four
RTX 3090s. The validated profile keeps Vision, prefix caching, external FP8
PLE, calibrated FP8 QSA KV, CUDA Graphs, and a configured 262K context window.

[![GPU](https://img.shields.io/badge/GPU-4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K_configured-ffb000)](#validated-v1-profile)
[![Vision](https://img.shields.io/badge/Vision-validated-6f42c1)](#validated-v1-profile)
[![KV cache](https://img.shields.io/badge/QSA_KV-FP8_E4M3-0969da)](#stack)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

## Quick start

```bash
git clone https://github.com/alesha-pro/qwen38-flash-next-4x3090.git
cd qwen38-flash-next-4x3090
./run_qwen_next.sh
```

The first run downloads only the required AWQ shards, derives the local
`Qwen3.8-Flash-Next-AWQ-INT4-4x3090-v1` checkpoint, downloads selected FP8
PLE shards, validates the composition, starts vLLM, waits for `/health`, then
runs text and Vision smoke requests. The OpenAI-compatible API is
`http://127.0.0.1:8018/v1`.

```bash
./run_qwen_next.sh --check-only
./run_qwen_next.sh --download-only
./stop_qwen_next.sh
```

## Validated v1 profile

The final v1 selection was tested on 2026-08-30 at 220 W per GPU. It is a
quality gate, not a claim that every workload is solved by a short smoke test.

| Gate | Result |
|---|---|
| Allocated QSA KV capacity | 276,145 tokens |
| Vision battery | 7 / 7 pass, plus 2 / 2 screenshot-understanding checks |
| Image-bearing long-context needle | coherent responses at 249,810 and 251,301 served prompt tokens |
| Prefix-cache multi-turn | 13 / 13 retained features; zero cross-request contamination |
| Browser-rendered Three.js task | two independent Pagoda Garden runs, human-reviewed pass |
| Matched quality suite | 16 / 20 pass, identical shared passes to the preserved rollback build |

The one known tool-recovery failure existed in both builds. A long-stability
case was borderline in one run and passed in a prior rerun, so it is not used
as a quality-win claim.

### Matched startup and depth sanity

The final build was compared with the preceding rollback build using fresh
boots, fixed nonces, `reasoning_effort=low`, `temperature=0`, and the same
262K serving configuration. In the seven comparable R2 cells, final v1 never
had a worse TTFT; this is a regression check, not a statistical speed claim.

| Input depth | Cold TTFT, rollback → v1 | Warm TTFT, rollback → v1 |
|---:|---:|---:|
| 4,096 | 9.278 s → 6.731 s | 5.010 s → 3.322 s |
| 32,768 | 18.849 s → 18.637 s | 1.870 s → 1.754 s |
| 131,072 | 75.299 s → 74.579 s | 2.182 s → 2.161 s |
| 250,000 | 152.777 s → 151.798 s | rollback emitted no visible content; v1 2.369 s |

The old `thin-v2-fix` measurements in
[`benchmarks/2026-08-29-awq-v2fix/`](benchmarks/2026-08-29-awq-v2fix/) are
kept as historical rollback data. Do not present them as a v1 throughput run.

## Checkpoint

The runtime model is derived from
[`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
at `01324cfa2c3f46948781fad30641ac360014e008`:

1. keep upstream AWQ shards 2 and 3;
2. omit shard 1, which is PLE-only, and use selected external FP8 PLE from
   [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
3. omit shard 4, which is MTP-only, because this deep-context profile keeps
   MTP disabled;
4. FP8-quantize selected BF16 companion projections;
5. restore 48 `shared_expert_gate` tensors to BF16. The runtime module is
   BF16 and would otherwise read FP8 bytes as corrupt gate values;
6. restore 72 Gated Delta Net `in_proj_a` / `in_proj_b` tensors to their
   original BF16 values. They add about 8.4 MiB per RTX 3090 and are part of
   the recurrent state-update path.

The builder is fail-closed: it requires exactly 72 GDN restores and removes
their matching FP8 scale tensors from both shards and the checkpoint index. It
writes `AWQ-4X3090-V1-MANIFEST.json` and `GDN-BF16-RESTORE-MANIFEST.json`
beside the local checkpoint. No custom model upload is needed.

## Stack

| Component | Effective setting |
|---|---|
| Main weights | cyankiwi AWQ INT4, local `Qwen3.8-Flash-Next-AWQ-INT4-4x3090-v1` |
| External PLE | selected FP8 E4M3 shards from RadixArk NVFP4 |
| Parallelism | tensor parallel 4 + expert parallel 4 |
| KV cache | calibrated FP8 E4M3 on the QSA path |
| Vision and prefix cache | enabled |
| Tool calls | Qwen3 XML and reasoning parsers |
| CUDA graphs | `{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}` |
| Eager mode / MTP | disabled / disabled |

The launcher verifies the pinned vLLM image, vendor QSA hashes, external-PLE
manifest, checkpoint index and overlays before mounting anything. See
[PATCH-MAP.md](PATCH-MAP.md) and [SOURCE-MAP.md](SOURCE-MAP.md).

## Hardware

| Resource | Minimum | Recommended |
|---|---:|---:|
| GPU | 4x RTX 3090 | 4x RTX 3090 |
| Aggregate VRAM | 96 GB | 96 GB |
| System RAM | 64 GB, theoretical | 128 GB |
| Free SSD for first build | 250 GB | 300 GB |

The measured host has 125 GB RAM. 64 GB is a theoretical floor, not a comfort
recommendation: the external PLE worker previously used about 49 GiB RSS.

## Configuration

```bash
QWEN38_MODELS_ROOT=/mnt/ssd/models \
MAX_MODEL_LEN=262144 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=1024 \
KV_CACHE_DTYPE=fp8_e4m3 \
./run_qwen_next.sh
```

| Variable | Default | Purpose |
|---|---|---|
| `QWEN38_MODELS_ROOT` | `~/.cache/qwen38-flash-next-4x3090/models` | checkpoint storage |
| `HOST` / `PORT` | `127.0.0.1` / `8018` | API bind address |
| `MAX_MODEL_LEN` | `262144` | configured context |
| `MAX_NUM_SEQS` | `1` | validated 256K plus Vision profile |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | scheduler token budget |
| `GPU_MEMORY_UTILIZATION` | `0.98` | vLLM memory fraction |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | QSA KV cache dtype |

## Reproducibility boundary

This is an Ampere-specific research runtime with pinned vLLM overlays. FP8 KV
for this QSA path is not an upstream vLLM feature. The starter profile trades
concurrency for 256K plus Vision headroom.
