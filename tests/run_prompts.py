#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Drive calibration or eval prompt sets against the local server.

Usage: run_prompts.py <calib|eval> [port] [max_tokens]
Output: one JSON per line on stdout: {"i": idx, "text": output}.
Deterministic (temperature=0). For calibration runs, outputs are discarded;
the server-side QSA_FP8_CALIBRATE_OUT collector does the real work.
"""
import json
import sys
import urllib.request

sys.path.insert(0, "/mnt/nvme2/projects/qwen38-kv-quantize/tests")
from prompts import CALIBRATION_PROMPTS, EVAL_PROMPTS  # noqa: E402

which = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 8018
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 8
prompts = CALIBRATION_PROMPTS if which == "calib" else EVAL_PROMPTS

for i, prompt in enumerate(prompts):
    payload = {
        "model": "/model",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)["choices"][0]["text"]
    print(json.dumps({"i": i, "text": out}), flush=True)
