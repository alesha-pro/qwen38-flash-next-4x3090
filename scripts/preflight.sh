#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
PINNED_IMAGE_ID="sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8"
MODEL="${MODEL_DIR:?MODEL_DIR is required}"
PLE_MODEL="${PLE_MODEL_DIR:?PLE_MODEL_DIR is required}"
PLE_MANIFEST="${ROOT}/manifests/ple-external-fp8-radixark.json"

docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}"
actual_image="$(docker image inspect "${IMAGE}" --format '{{.Id}}')"
[[ "${actual_image}" == "${PINNED_IMAGE_ID}" ]] || {
    echo "image mismatch: ${actual_image}" >&2; exit 2; }

VLLM_PKG=/usr/local/lib/python3.12/dist-packages/vllm
actual_qsa_hashes="$(docker run --rm --entrypoint sha256sum "${IMAGE}" \
    "${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/qsa.py" \
    "${VLLM_PKG}/models/qwen3_8_flash_next/nvidia/ops/qsa.py")"
expected_qsa_hashes="$(grep -v '^#' "${ROOT}/overlays/qwen38/BASE.sha256")"
[[ "${actual_qsa_hashes}" == "${expected_qsa_hashes}" ]] || {
    echo "QSA vendor base hash mismatch" >&2; exit 2; }

[[ -f "${MODEL}/model.safetensors.index.json" ]]
[[ -f "${PLE_MODEL}/model.safetensors.index.json" ]]
python3 "${ROOT}/scripts/ple_manifest.py" validate "${PLE_MODEL}" "${PLE_MANIFEST}"

python3 -m py_compile \
    "${ROOT}/overlays/ple_layer.py" \
    "${ROOT}/overlays/ple_offload/worker.py" \
    "${ROOT}/overlays/ple_offload/ple_external_source.py" \
    "${ROOT}/overlays/qwen38/qsa.py" \
    "${ROOT}/overlays/qwen38/ops_qsa.py"
bash -n "${ROOT}/scripts/launch-integrated.sh"
bash -n "${ROOT}/run_qwen_next.sh"
bash -n "${ROOT}/stop_qwen_next.sh"

echo "PREFLIGHT PASS: static composition and pinned inputs verified"
