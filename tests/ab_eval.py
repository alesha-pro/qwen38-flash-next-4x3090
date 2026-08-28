#!/usr/bin/env python3
"""Deterministic A/B eval arm: EVAL_PROMPTS with temperature 0 + top-5
logprobs on the first generated tokens, for BF16-KV vs FP8-KV comparison.

Usage: ab_eval.py <arm_name> <out_jsonl> [port] [max_tokens]
Writes one JSON per prompt: prompt, text, per-token top-5 logprobs.
"""
import json
import sys
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from prompts import EVAL_PROMPTS  # noqa: E402

arm = sys.argv[1]
out_path = sys.argv[2]
port = int(sys.argv[3]) if len(sys.argv) > 3 else 8018
max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 64

with open(out_path, "w") as f:
    for i, prompt in enumerate(EVAL_PROMPTS):
        payload = {
            "model": "/model",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "logprobs": 5,
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.load(resp)
        choice = data["choices"][0]
        rec = {
            "arm": arm, "i": i, "prompt": prompt,
            "text": choice["text"],
            "finish_reason": choice.get("finish_reason"),
            "logprobs": choice.get("logprobs", {}).get("content", []),
        }
        f.write(json.dumps(rec) + "\n")
        print(f"[{arm} {i:02d}] {choice['text'][:60]!r}", flush=True)
print("AB_ARM_DONE", arm)
