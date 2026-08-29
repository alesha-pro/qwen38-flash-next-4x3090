#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
PINNED_IMAGE_ID="${PINNED_IMAGE_ID:-sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8}"
MODEL_DIR="${MODEL_DIR:-/mnt/ssd/models/Qwen3.8-Flash-Next-AWQ-INT4-cyankiwi-thin-v2-fix}"
PLE_MODEL_DIR="${PLE_MODEL_DIR:-/mnt/ssd/models/Qwen3.8-Flash-Next-NVFP4}"
PLE_MANIFEST="${PLE_MANIFEST:-${ROOT}/manifests/ple-external-fp8-radixark.json}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/mnt/ssd/hf_cache}"
COMPILE_CACHE_DIR="${COMPILE_CACHE_DIR:-/mnt/ssd/vllm_compile_cache}"
PORT="${PORT:-8018}"
HOST="${HOST:-127.0.0.1}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-awq-v2fix-262k}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.98}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-}"
if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
    # Single-quoted default: a ${VAR:-...} word containing braces appends a
    # literal '}' when VAR is set (bash terminates the word at the first '}').
    COMPILATION_CONFIG='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'
fi
DIST=/usr/local/lib/python3.12/dist-packages
VLLM_PKG="${DIST}/vllm"

[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || {
    echo "derived cyankiwi AWQ thin-v2-fix checkpoint missing" >&2; exit 2; }
[[ -f "${PLE_MODEL_DIR}/model.safetensors.index.json" ]] || {
    echo "external PLE checkpoint missing" >&2; exit 2; }
[[ -f "${PLE_MANIFEST}" ]] || { echo "PLE manifest missing" >&2; exit 2; }

actual_image="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
[[ "${actual_image}" == "${PINNED_IMAGE_ID}" ]] || {
    echo "image mismatch: expected ${PINNED_IMAGE_ID}, got ${actual_image}" >&2
    exit 2
}
python3 "${ROOT}/scripts/ple_manifest.py" validate \
    "${PLE_MODEL_DIR}" "${PLE_MANIFEST}"
mkdir -p "${COMPILE_CACHE_DIR}/vllm" "${COMPILE_CACHE_DIR}/torchinductor"

for f in \
    overlays/ple_layer.py overlays/ple_offload/worker.py \
    overlays/ple_offload/ple_external_source.py overlays/gpu_worker.py \
    overlays/multiproc_executor.py overlays/qwen_gdn_linear_attn.py \
    overlays/shared_experts.py; do
    [[ -f "${ROOT}/${f}" ]] || { echo "missing ${f}" >&2; exit 2; }
done

EXTRA_ARGS=()
[[ -n "${MAX_NUM_BATCHED_TOKENS}" ]] && \
    EXTRA_ARGS+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
[[ -n "${KV_CACHE_MEMORY_BYTES}" ]] && \
    EXTRA_ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY_BYTES}")

QSA_MOUNTS=()
QSA_ENV=()
SPLIT_PROJ_MOUNTS=()
SPLIT_PROJ_ENV=()
for f in overlays/qwen38_flash_next_model.py overlays/hyperconnection.py; do
    [[ -f "${ROOT}/${f}" ]] || {
        echo "missing split-projection overlay ${f}" >&2; exit 2; }
done
SPLIT_PROJ_ENV+=(
    -e VLLM_GDN_SPLIT_PROJ=1
    -e VLLM_HYPER_SPLIT_PROJ=1
)
SPLIT_PROJ_MOUNTS+=(
    -v "${ROOT}/overlays/qwen38_flash_next_model.py:${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/model.py:ro"
    -v "${ROOT}/overlays/hyperconnection.py:${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/hyperconnection.py:ro"
)
if [[ "${KV_CACHE_DTYPE}" == fp8* ]]; then
    for f in qsa.py ops_qsa.py scales.json BASE.sha256; do
        [[ -f "${ROOT}/overlays/qwen38/${f}" ]] || {
            echo "missing FP8 QSA artifact ${f}" >&2; exit 2; }
    done
    actual_hashes="$(docker run --rm --entrypoint sha256sum "${IMAGE}" \
        "${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/qsa.py" \
        "${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ops/qsa.py")"
    expected_hashes="$(grep -v '^#' "${ROOT}/overlays/qwen38/BASE.sha256")"
    [[ "${actual_hashes}" == "${expected_hashes}" ]] || {
        echo "QSA base hash mismatch" >&2; exit 2; }
    QSA_ENV+=(-e QSA_FP8_SCALES_FILE=/qsa-scales/scales.json)
    QSA_MOUNTS+=(
        -v "${ROOT}/overlays/qwen38/qsa.py:${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/qsa.py:ro"
        -v "${ROOT}/overlays/qwen38/ops_qsa.py:${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro"
        -v "${ROOT}/overlays/qwen38/scales.json:/qsa-scales/scales.json:ro"
    )
    echo "calibrated FP8 QSA overlays enabled" >&2
fi

DOCKER_RUN_ARGS=(--rm)
[[ "${DETACH:-0}" == 1 ]] && DOCKER_RUN_ARGS+=(--detach)

exec docker run "${DOCKER_RUN_ARGS[@]}" \
    --name "${CONTAINER_NAME}" \
    --gpus all \
    --ipc=host \
    --network=host \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_PLE_MODEL_PATH=/ple-model \
    -e VLLM_PLE_MANIFEST=/ple-manifest/manifest.json \
    -e VLLM_PLE_EMBEDDING_DTYPE=float8_e4m3fn \
    -e HF_HOME=/hf-cache \
    -e TORCHINDUCTOR_CACHE_DIR=/compile-cache/torchinductor \
    "${QSA_ENV[@]}" \
    "${SPLIT_PROJ_ENV[@]}" \
    -v "${MODEL_DIR}:/model:ro" \
    -v "${PLE_MODEL_DIR}:/ple-model:ro" \
    -v "${PLE_MANIFEST}:/ple-manifest/manifest.json:ro" \
    -v "${HF_CACHE_DIR}:/hf-cache" \
    -v "${COMPILE_CACHE_DIR}/vllm:/root/.cache/vllm" \
    -v "${COMPILE_CACHE_DIR}/torchinductor:/compile-cache/torchinductor" \
    -v "${ROOT}/overlays/ple_layer.py:${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" \
    -v "${ROOT}/overlays/ple_offload/worker.py:${VLLM_PKG}/v1/ple_offload/worker.py:ro" \
    -v "${ROOT}/overlays/ple_offload/ple_external_source.py:${VLLM_PKG}/v1/ple_offload/ple_external_source.py:ro" \
    -v "${ROOT}/overlays/gpu_worker.py:${VLLM_PKG}/v1/worker/gpu_worker.py:ro" \
    -v "${ROOT}/overlays/multiproc_executor.py:${VLLM_PKG}/v1/executor/multiproc_executor.py:ro" \
    -v "${ROOT}/overlays/qwen_gdn_linear_attn.py:${VLLM_PKG}/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro" \
    -v "${ROOT}/overlays/shared_experts.py:${VLLM_PKG}/model_executor/layers/fused_moe/runner/shared_experts.py:ro" \
    "${SPLIT_PROJ_MOUNTS[@]}" \
    "${QSA_MOUNTS[@]}" \
    "${IMAGE}" \
    /model \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --kv-cache-dtype "${KV_CACHE_DTYPE}" \
    --compilation-config "${COMPILATION_CONFIG}" \
    "${EXTRA_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --no-enable-flashinfer-autotune \
    --host "${HOST}" \
    --port "${PORT}"
