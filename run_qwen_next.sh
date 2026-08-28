#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${ROOT}/.env" ]] && set -a && source "${ROOT}/.env" && set +a

MODELS_ROOT="${QWEN38_MODELS_ROOT:-${HOME}/.cache/qwen38-flash-next-4x3090/models}"
export MODEL_DIR="${MODEL_DIR:-${MODELS_ROOT}/Qwen3.8-Flash-Next-W4A16}"
export PLE_MODEL_DIR="${PLE_MODEL_DIR:-${MODELS_ROOT}/Qwen3.8-Flash-Next-NVFP4}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-${MODELS_ROOT}/.hf-cache}"
export PORT="${PORT:-8018}"
export HOST="${HOST:-127.0.0.1}"
export CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-4x3090}"

mode=start
case "${1:-}" in
    "") ;;
    --check-only) mode=check ;;
    --download-only) mode=download ;;
    -h|--help) sed -n '1,180p' "${ROOT}/README.md"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
esac

needs_download=0
python3 "${ROOT}/scripts/verify_primary.py" "${MODEL_DIR}" \
    >/dev/null 2>&1 || needs_download=1
if ! python3 "${ROOT}/scripts/ple_manifest.py" validate \
        "${PLE_MODEL_DIR}" "${ROOT}/manifests/ple-external-fp8-radixark.json" \
        >/dev/null 2>&1; then
    needs_download=1
fi

if [[ "${needs_download}" == 1 ]]; then
    mkdir -p "${MODELS_ROOT}" "${HF_CACHE_DIR}"
    VENV="${ROOT}/.venv"
    if [[ ! -x "${VENV}/bin/python" ]]; then
        python3 -m venv "${VENV}"
        "${VENV}/bin/python" -m pip install --quiet 'huggingface_hub==1.2.2'
    fi
    "${VENV}/bin/python" "${ROOT}/scripts/download_models.py" \
        --model-dir "${MODEL_DIR}" --ple-dir "${PLE_MODEL_DIR}"
fi

if [[ "${mode}" == download ]]; then
    echo "DOWNLOAD COMPLETE"
    exit 0
fi

bash "${ROOT}/scripts/doctor.sh"
bash "${ROOT}/scripts/preflight.sh"

if [[ "${mode}" == check ]]; then
    echo "CHECK COMPLETE"
    exit 0
fi

export DETACH=1
bash "${ROOT}/scripts/launch-integrated.sh"

if ! bash "${ROOT}/scripts/wait_ready.sh"; then
    docker logs --tail 200 "${CONTAINER_NAME}" >&2 || true
    exit 1
fi

if [[ "${SKIP_SMOKE:-0}" != 1 ]]; then
    python3 "${ROOT}/tests/smoke.py" "${PORT}"
fi

cat <<EOF
READY
OpenAI endpoint: http://${HOST}:${PORT}/v1
Container: ${CONTAINER_NAME}
Logs: docker logs -f ${CONTAINER_NAME}
Stop: ${ROOT}/stop_qwen_next.sh
EOF
