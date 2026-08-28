#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8018}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen38-flash-next-4x3090}"
TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"
deadline=$((SECONDS + TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
    if ! docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
        echo "container ${CONTAINER_NAME} exited during startup" >&2
        exit 1
    fi
    if python3 - "${HOST}" "${PORT}" <<'PY' >/dev/null 2>&1
import sys, urllib.request
urllib.request.urlopen(f"http://{sys.argv[1]}:{sys.argv[2]}/health", timeout=2).read()
PY
    then
        echo "server health check passed"
        exit 0
    fi
    sleep 5
done

echo "server did not become ready within ${TIMEOUT_SECONDS}s" >&2
exit 1
