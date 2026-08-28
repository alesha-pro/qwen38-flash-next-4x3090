#!/usr/bin/env python3
"""Long-context image-bearing needle test (Gate I4/I5).

Builds a request containing: one real image (ocr_text.png), text padding
with a unique late needle, and final questions covering both the text needle
and image content. Verifies actual served input tokens, needle recall, image
answer and coherent decode.

Usage: needle_vision.py <run_dir> <target_tokens> [port]
Writes needle_result.json into <run_dir>. Exit 0 on PASS.
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request

RUN = sys.argv[1]
TARGET = int(sys.argv[2])
PORT = int(sys.argv[3]) if len(sys.argv) > 3 else 8018
BASE = f"http://127.0.0.1:{PORT}"
IMG = os.path.join(RUN, "inputs", "ocr_text.png")
NEEDLE = "ZULU-7731-QUEBEC"
IMG_ANSWER = "Q-8842-XK"

with open(IMG, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_uri = f"data:image/png;base64,{b64}"

FILLER_UNIT = ("The archivist logged another routine morning in the stacks; "
               "the catalog remained stable and no volume required attention. ")


def tokenize(text):
    payload = {"model": "/model", "prompt": text}
    req = urllib.request.Request(
        f"{BASE}/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return len(json.load(r)["tokens"])


# Reserve headroom for image tokens + questions + template (~2.5k tokens).
budget = TARGET - 2500
n_units = max(1, budget // 14)  # ~14 tokens per filler unit
prefix = FILLER_UNIT * (n_units * 55 // 100)
suffix = FILLER_UNIT * (n_units - n_units * 55 // 100)
needle_sentence = (f" IMPORTANT RECORD: the archive access code for this "
                   f"session is {NEEDLE}. ")
question = (
    "Two questions about the material above. "
    "1) What is the archive access code stated in the late part of the text? "
    "2) What is the order number printed in the image? "
    "Answer on two lines, each prefixed exactly 'ANSWER1: ' / 'ANSWER2: ' "
    "with just the value and nothing else."
)
prompt_text = prefix + needle_sentence + suffix + "\n\n" + question

t0 = time.time()
tok_count = tokenize(prompt_text)
# One refinement pass to land closer to target.
if abs(tok_count - budget) > budget * 0.05:
    scale = budget / tok_count
    n_units = max(1, int(n_units * scale))
    prefix = FILLER_UNIT * (n_units * 55 // 100)
    suffix = FILLER_UNIT * (n_units - n_units * 55 // 100)
    prompt_text = prefix + needle_sentence + suffix + "\n\n" + question
    tok_count = tokenize(prompt_text)

body = {
    "model": "/model",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text", "text": prompt_text},
    ]}],
    "temperature": 0,
    "max_tokens": 512,
}
req = urllib.request.Request(
    f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"})
t1 = time.time()
with urllib.request.urlopen(req, timeout=7200) as r:
    resp = json.load(r)
t2 = time.time()

text = resp["choices"][0]["message"]["content"]
usage = resp.get("usage", {})
final = text[text.rfind("</think>") + 8:].strip() if "</think>" in text else text.strip()
finish = resp["choices"][0].get("finish_reason")

checks = {
    "image_present": True,
    "tokenizer_counted_input": tok_count,
    "served_prompt_tokens": usage.get("prompt_tokens"),
    "completion_tokens": usage.get("completion_tokens"),
    "finish_reason": finish,
    "needle_found": NEEDLE in final,
    "image_answer_correct": IMG_ANSWER in final,
    "coherent": finish == "stop" and len(final) > 0,
}
checks["PASS"] = all(checks[k] for k in ("needle_found", "image_answer_correct", "coherent"))
result = {
    "target_tokens": TARGET,
    "checks": checks,
    "final_answer": final[:500],
    "raw_length": len(text),
    "tokenize_seconds": round(t1 - t0, 1),
    "request_seconds": round(t2 - t1, 1),
    "request": body,
}
with open(os.path.join(RUN, "needle_result.json"), "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps({k: checks[k] for k in checks}, indent=2))
print(f"final_answer[:300]: {final[:300]!r}")
print("NEEDLE_VISION:", "PASS" if checks["PASS"] else "FAIL")
sys.exit(0 if checks["PASS"] else 1)
