# Qwen3.8 Flash Next on 4x RTX 3090

One-command vLLM recipe for Qwen3.8 Flash Next 122B / 51B active with Vision,
external FP8 PLE, calibrated FP8 QSA KV and a configured 262K context window.

[![GPU](https://img.shields.io/badge/GPU-4x_RTX_3090-76B900?logo=nvidia&logoColor=white)](#hardware)
[![Context](https://img.shields.io/badge/context-262K-ffb000)](#measured-256k-vision-speed)
[![Vision](https://img.shields.io/badge/Vision-on-6f42c1)](#stack)
[![KV cache](https://img.shields.io/badge/QSA_KV-FP8_E4M3-0969da)](#stack)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)

## Quick start

```bash
git clone https://github.com/alesha-pro/qwen38-flash-next-4x3090.git
cd qwen38-flash-next-4x3090
./run_qwen_next.sh
```

The first run downloads only the required AWQ shards, derives the local
`thin-v2-fix` checkpoint, downloads selected FP8 PLE shards, validates the
composition, starts vLLM, waits for `/health`, then runs text and Vision smoke
requests. The OpenAI-compatible API is `http://127.0.0.1:8018/v1`.

```bash
./run_qwen_next.sh --check-only
./run_qwen_next.sh --download-only
./stop_qwen_next.sh
```

## Measured 256K Vision speed

Measured on 2026-08-29 at 220 W per GPU. Every request had a real OCR image,
a unique nonce to defeat prefix-cache reuse and `enable_thinking=false`. The
server stayed up throughout: TP4+EP4, external FP8 PLE, calibrated FP8 E4M3
QSA KV, `max_num_seqs=1`, FULL_DECODE_ONLY CUDA Graphs and no eager mode.

| Served input | TTFT / prefill | Approx. prefill rate | Decode |
|---:|---:|---:|---:|
| 32,610 tokens + Vision | 18.01 s | 1,811 tok/s | 63.1 tok/s median (3 runs) |
| 130,911 tokens + Vision | 75.25 s | 1,740 tok/s | 64.9 tok/s |
| 229,212 tokens + Vision | 138.26 s | 1,658 tok/s | 66.6 tok/s |
| 252,732 tokens + Vision | 162.48 s | 1,555 tok/s | 67.9 tok/s |

Dedicated near-max decode: 252,732 input tokens + Vision, `154.12 s` TTFT,
`64.31 tok/s` decode and `157.66 s` end to end. Decode stays in the 63-68
tok/s band from 32K to almost full context; the long-context cost is prefill.

This profile intentionally has `MAX_NUM_SEQS=1` to fit 256K+Vision. At 32K,
c=2 took 38.32 s and c=4 took 76.29 s wall time: FIFO queueing, not aggregate
concurrency. Do not raise it without your own capacity and quality gates.

Raw JSONL and telemetry: [`benchmarks/2026-08-29-awq-v2fix/`](benchmarks/2026-08-29-awq-v2fix/).
W4A16 data under [`benchmarks/2026-08-28/`](benchmarks/2026-08-28/) is a
historical control; do not mix its methodology with this suite.

## Checkpoint

The runtime model is derived from
[`cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4`](https://huggingface.co/cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4)
at `01324cfa2c3f46948781fad30641ac360014e008`:

1. keep upstream AWQ shards 2 and 3;
2. omit shard 1 (PLE-only) and use selected external FP8 PLE from
   [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4);
3. omit shard 4 (MTP-only; MTP is disabled);
4. FP8-quantize safe BF16 companion projections;
5. restore `shared_expert_gate` to BF16. Its runtime module is BF16 and would
   otherwise read FP8 bytes as corrupt gate values.

`run_qwen_next.sh` builds `thin-v2-fix` automatically and writes
`AWQ-V2FIX-MANIFEST.json` beside it. No custom model upload is needed.

## Stack

| Component | Effective setting |
|---|---|
| Main weights | cyankiwi AWQ INT4, local derived thin-v2-fix |
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
| `MAX_NUM_SEQS` | `1` | validated 256K+Vision profile |
| `MAX_NUM_BATCHED_TOKENS` | `1024` | scheduler token budget |
| `GPU_MEMORY_UTILIZATION` | `0.98` | vLLM memory fraction |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | QSA KV cache dtype |

## Reproducibility boundary

This is an Ampere-specific research runtime with pinned vLLM overlays. FP8 KV
for this QSA path is not an upstream vLLM feature. The starter profile trades
concurrency for 256K+Vision headroom.
