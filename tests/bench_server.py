#!/usr/bin/env python3
"""Single-stream decode benchmark against the local server.

Reports TTFT and TPOT from streaming SSE. Boundary: end-to-end smoke
benchmark, temperature=0, fixed prompt; used for BF16-vs-FP8 comparison at
identical settings, not as an absolute performance claim.
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8018
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3

PROMPT = (
    "Write a detailed technical explanation of how paged KV caches work in "
    "LLM inference engines, covering block tables, fragmentation, and "
    "prefix sharing."
)

payload = {
    "model": "/model",
    "prompt": PROMPT,
    "max_tokens": MAX_TOKENS,
    "temperature": 0,
    "stream": True,
    "stream_options": {"include_usage": True},
}

for rep in range(REPS):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first = None
    n_tokens = 0
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("choices") and chunk["choices"][0].get("text"):
                n_tokens += 1
                if t_first is None:
                    t_first = time.monotonic()
    t_end = time.monotonic()
    ttft = (t_first - t0) if t_first else float("nan")
    tpot = (t_end - t_first) / max(n_tokens - 1, 1) if t_first else float("nan")
    print(
        f"rep{rep}: tokens={n_tokens} ttft={ttft * 1e3:.1f}ms "
        f"tpot={tpot * 1e3:.2f}ms ({1.0 / tpot:.1f} tok/s)"
    )
