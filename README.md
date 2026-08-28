# Qwen3.8-Flash-Next on 4x RTX 3090

Reproducible vLLM recipe for serving the 122B/51B-active
Qwen3.8-Flash-Next model with Vision on four RTX 3090 GPUs.

The tested stack combines:

- `VnimanieAI/Qwen3.8-Flash-Next-W4A16` model weights;
- FP8 E4M3 PLE tensors selected from `RadixArk/Qwen3.8-Flash-Next-NVFP4`;
- calibrated FP8 E4M3 QSA KV cache;
- tensor parallel 4 plus expert parallel 4;
- prefix caching;
- `VLLM_COMPILE` and `FULL_AND_PIECEWISE` CUDA graphs;
- full multimodal/Vision support, without `--language-model-only` or eager mode.

The upstream checkpoints are downloaded at pinned immutable revisions. This
repository contains the runtime composition, vLLM overlays, calibrated QSA
scales, validation manifest and smoke tests. It does not republish upstream
weights.

## Requirements

- Linux x86_64;
- exactly 4x NVIDIA RTX 3090 (96 GB aggregate VRAM);
- 64 GB system RAM as a theoretical, not yet validated minimum;
- 128 GB system RAM recommended for comfortable startup headroom;
- recent NVIDIA driver with working Docker GPU passthrough;
- Docker and NVIDIA Container Toolkit;
- approximately 250 GB of free model storage;
- internet access to Docker Hub and Hugging Face.

## One-command start

```bash
git clone https://github.com/alesha-pro/qwen38-flash-next-4x3090.git
cd qwen38-flash-next-4x3090
./run_qwen_next.sh
```

The first run downloads roughly 237 GB of model data and can take a long
time. Subsequent starts reuse the local files. The launcher waits for the
OpenAI-compatible endpoint and runs one text plus one Vision smoke request.

The 64 GB RAM floor is an estimate, not a completed validation result. The
external FP8 PLE worker used approximately 49.2 GiB RSS in the reference run,
leaving little room for the operating system and runtime processes on a
64 GB host. Use 128 GB when possible.

Default endpoint: `http://127.0.0.1:8018/v1`.

Stop only this stack:

```bash
./stop_qwen_next.sh
```

## Storage and configuration

Copy `.env.example` to `.env` to override defaults. On a server with model
storage mounted at `/mnt/ssd/models`:

```bash
QWEN38_MODELS_ROOT=/mnt/ssd/models ./run_qwen_next.sh
```

Important options:

```bash
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=512
KV_CACHE_DTYPE=fp8_e4m3
PORT=8018
HOST=127.0.0.1
SKIP_SMOKE=0
```

The observed KV capacity printed by vLLM is not itself proof that a request
was served. The validated reference run served a real image-bearing request
with 261,591 input tokens while Vision and compiled CUDA graphs were enabled.

## Existing local weights

No download is performed when both checkpoint directories validate:

```bash
MODEL_DIR=/path/to/Qwen3.8-Flash-Next-W4A16 \
PLE_MODEL_DIR=/path/to/Qwen3.8-Flash-Next-NVFP4 \
./run_qwen_next.sh
```

Run only static checks:

```bash
MODEL_DIR=/path/to/w4a16 \
PLE_MODEL_DIR=/path/to/nvfp4 \
./run_qwen_next.sh --check-only
```

Download without starting GPUs:

```bash
./run_qwen_next.sh --download-only
```

## What is patched

See `PATCH-MAP.md` and `SOURCE-MAP.md`. The launcher verifies the pinned
container image ID and the base hashes of the vendor QSA files before mounting
the overlays. PLE validation is fail-closed and checks the pinned revision,
file sizes, tensor metadata, the BF16 scale tensor and sampled FP8 shard
payload hashes.

## Reproducibility boundary

Reference runtime:

- image: `vllm/vllm-openai:qwen38-flash-next`;
- image ID: `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8`;
- vLLM: `0.1.dev20073+g8e685d198`;
- W4A16 revision: `9236d703b25f25eb5c17e9640204f84fa1ce0c6e`;
- FP8 PLE revision: `7b719225242aacd3dbd3f9407468c2ee9a9d2594`.

This is currently an Ampere-specific research runtime, not an upstream vLLM
feature. Do not silently substitute a newer image: base-file hashes are part
of the compatibility contract.
