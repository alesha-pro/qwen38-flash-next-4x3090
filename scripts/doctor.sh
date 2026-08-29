#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

: "${MODEL_DIR:?MODEL_DIR is required}"
: "${PLE_MODEL_DIR:?PLE_MODEL_DIR is required}"
PORT="${PORT:-8018}"

for cmd in docker nvidia-smi python3; do
    command -v "${cmd}" >/dev/null || { echo "missing command: ${cmd}" >&2; exit 2; }
done
docker info >/dev/null 2>&1 || { echo "Docker daemon is unavailable" >&2; exit 2; }

mapfile -t gpu_names < <(nvidia-smi --query-gpu=name --format=csv,noheader)
[[ "${#gpu_names[@]}" == 4 ]] || {
    echo "expected exactly 4 GPUs, found ${#gpu_names[@]}" >&2; exit 2; }
for name in "${gpu_names[@]}"; do
    [[ "${name}" == *"RTX 3090"* ]] || {
        echo "unsupported GPU: ${name}; this recipe targets RTX 3090" >&2; exit 2; }
done

if [[ "${ALLOW_BUSY_GPU:-0}" != 1 ]]; then
    busy="$(nvidia-smi --query-compute-apps=pid,process_name \
        --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${busy//[[:space:]]/}" ]] || {
        echo "GPU compute processes are already running; refusing to interfere:" >&2
        echo "${busy}" >&2
        exit 2
    }
fi

if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${PORT}" | grep -q .; then
    echo "TCP port ${PORT} is already in use" >&2
    exit 2
fi

[[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] || {
    echo "primary cyankiwi AWQ thin-v2-fix checkpoint is incomplete: ${MODEL_DIR}" >&2; exit 2; }
[[ -f "${PLE_MODEL_DIR}/model.safetensors.index.json" ]] || {
    echo "external PLE checkpoint is incomplete: ${PLE_MODEL_DIR}" >&2; exit 2; }

python3 "${ROOT}/scripts/verify_primary.py" "${MODEL_DIR}"

echo "DOCTOR PASS: 4x RTX 3090, Docker, checkpoints and port ${PORT}"
