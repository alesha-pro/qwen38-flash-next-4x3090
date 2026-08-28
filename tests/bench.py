#!/usr/bin/env python3
"""Benchmark driver (B2/B3/B4/B5): image-bearing requests with stream timing.

Modes:
  single <depth_tokens> <repeats> [out_tokens]   — B2/B5: concurrency 1
  prefill <depth_tokens> <repeats> [out_tokens]  — B4 arm (same as single,
                                                    kept for clarity)
  concur <depth_tokens> <c> <repeats>            — B3: parallel requests

Every request contains ocr_text.png and a unique session nonce at the START
of the text so prefix-cache reuse across repeats is impossible.

Usage: bench.py <mode> <args...>
Env: BENCH_PORT (default 8018), BENCH_TAG (run label for stdout).
Writes JSONL records to stdout; caller redirects.
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get("BENCH_PORT", "8018"))
TAG = os.environ.get("BENCH_TAG", "bench")
BASE = f"http://127.0.0.1:{PORT}"

_HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PATH = os.environ.get("BENCH_IMG",
                          os.path.join(_HERE, "assets", "ocr_text.png"))

with open(IMG_PATH, "rb") as f:
    B64 = base64.b64encode(f.read()).decode()
FILLER_UNIT = ("The archivist logged another routine morning in the stacks; "
               "the catalog remained stable and no volume required attention. ")

DATA_URI = f"data:image/png;base64,{B64}"


def _tokenize_count(text):
    req = urllib.request.Request(
        f"{BASE}/tokenize",
        data=json.dumps({"model": "/model", "prompt": text}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return len(json.load(r)["tokens"])

def build_prompt(depth_tokens, nonce, calibrate=True):
    reserve = 3500 if depth_tokens > 250000 else 400
    budget = max(64, depth_tokens - reserve)
    per_unit = 20.05  # measured bulk BPE rate for FILLER_UNIT repetitions
    n_units = max(1, int(budget / per_unit))
    question = ("What is the session access code stated in the late text, "
                "and what is the order number printed in the image? "
                "Answer on two lines prefixed 'ANSWER1: '/'ANSWER2: '.")

    def assemble(n):
        prefix = FILLER_UNIT * (n * 55 // 100)
        suffix = FILLER_UNIT * (n - n * 55 // 100)
        needle = f" NOTE: session access code {nonce}. "
        return (f"SESSION {nonce}. " + prefix + needle + suffix
                + "\n\n" + question)

    prompt = assemble(n_units)
    if calibrate and n_units > 1:
        got = _tokenize_count(prompt)
        if abs(got - budget) > budget * 0.02:
            n_units = max(1, int(n_units * budget / got))
            prompt = assemble(n_units)
    return prompt


def one_request(depth_tokens, nonce, out_tokens):
    prompt = build_prompt(depth_tokens, nonce)
    body = {
        "model": "/model",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": DATA_URI}},
            {"type": "text", "text": prompt},
        ]}],
        "temperature": 0,
        "max_tokens": out_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t_req = time.perf_counter()
    t_first = None
    itls = []
    usage = None
    text_parts = []
    finish = None
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if ev.get("usage"):
                usage = ev["usage"]
            ch = ev.get("choices")
            if not ch:
                continue
            delta = ch[0].get("delta", {}) or {}
            tok = delta.get("content")
            if tok:
                t_now = time.perf_counter()
                if t_first is None:
                    t_first = t_now
                else:
                    itls.append(t_now - t_last)
                t_last = t_now
                text_parts.append(tok)
            if ch[0].get("finish_reason"):
                finish = ch[0]["finish_reason"]
    t_end = time.perf_counter()
    n_out = usage.get("completion_tokens") if usage else len(itls) + 1
    n_in = usage.get("prompt_tokens") if usage else None
    ttft = (t_first - t_req) if t_first else None
    decode_s = (t_end - t_first) if t_first else 0.0
    itls_valid = itls[1:] if itls else []  # drop first gap (pre-stream)
    tpot = (sum(itls_valid) / len(itls_valid)) if itls_valid else None
    rec = {
        "tag": TAG, "depth_target": depth_tokens, "nonce": nonce,
        "prompt_tokens": n_in, "completion_tokens": n_out,
        "finish_reason": finish, "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_ms": round(tpot * 1000, 3) if tpot else None,
        "decode_tok_s": round((n_out - 1) / decode_s, 3) if decode_s > 0 and n_out > 1 else None,
        "e2e_s": round(t_end - t_req, 3),
        "itl_ms_all": [round(x * 1000, 3) for x in itls],
        "answer": "".join(text_parts)[-160:],
    }
    return rec


def main():
    mode = sys.argv[1]
    if mode in ("single", "prefill"):
        depth, repeats = int(sys.argv[2]), int(sys.argv[3])
        out_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 256
        for r in range(repeats):
            nonce = f"NV{int(time.time())%100000:05d}-{r:02d}"
            rec = one_request(depth, nonce, out_tokens)
            print(json.dumps(rec), flush=True)
    elif mode == "concur":
        depth, c, repeats = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
        out_tokens = int(sys.argv[5]) if len(sys.argv) > 5 else 64
        for r in range(repeats):
            nonces = [f"NV{int(time.time())%100000:05d}-{r:02d}-{k:02d}" for k in range(c)]
            recs = [None] * c
            def work(k):
                recs[k] = one_request(depth, nonces[k], out_tokens)
            ts = [threading.Thread(target=work, args=(k,)) for k in range(c)]
            t0 = time.perf_counter()
            for t in ts: t.start()
            for t in ts: t.join()
            wall = time.perf_counter() - t0
            agg = {
                "tag": TAG, "mode": "concur", "depth_target": depth,
                "concurrency": c, "repeat": r, "wall_s": round(wall, 3),
                "requests": recs,
                "ok": sum(1 for x in recs if x and x["finish_reason"] in ("stop", "length")),
            }
            print(json.dumps(agg), flush=True)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
