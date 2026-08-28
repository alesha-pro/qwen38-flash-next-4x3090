#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
[[ -f "${ROOT}/.env" ]] && set -a && source "${ROOT}/.env" && set +a
name="${CONTAINER_NAME:-qwen38-flash-next-4x3090}"

if docker ps --format '{{.Names}}' | grep -Fxq "${name}"; then
    docker stop --time 30 "${name}"
    echo "stopped ${name}"
else
    echo "${name} is not running"
fi
