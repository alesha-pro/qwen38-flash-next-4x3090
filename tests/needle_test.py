#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Needle-in-haystack long-context check within the vision pool ceiling.

Builds a ~30K-token document (fits the 46,530-token vision-mode pool), hides
two needles, asks for both. Deterministic (temperature=0).
"""
import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8018


def post(path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return json.load(resp)


def count_tokens(text):
    return len(post("/tokenize", {"model": "/model", "prompt": text})["tokens"])


filler = ("The quarterly logistics report covers shipping volumes, warehouse "
          "rotations, and routine inventory adjustments across regional depots. ")
needle1 = "The override code for vault seven is KX-9917-ZEBRA."
needle2 = "The auditor's private note mentions a discrepancy of exactly 43112 units."

doc = filler * 500 + " " + needle1 + " " + filler * 1000 + " " + needle2 + " " + filler * 500
n = count_tokens(doc)
print(f"document tokens: {n}")
while n > 42000:  # shrink to fit pool with margin
    doc = doc[: int(len(doc) * 0.95)]
    n = count_tokens(doc)
print(f"adjusted tokens: {n}")

q = ("\n\nQuestion: what is the override code for vault seven, and what exact "
     "discrepancy did the auditor's private note mention? Answer in one sentence.")
payload = {"model": "/model", "prompt": doc + q, "max_tokens": 128,
           "temperature": 0}
t0 = __import__("time").monotonic()
resp = post("/v1/completions", payload)
dt = __import__("time").monotonic() - t0
text = resp["choices"][0]["text"]
usage = resp.get("usage", {})
print(f"TTFT+decode: {dt:.1f}s, usage: {usage}")
print(f"answer: {text!r}")
ok1 = "KX-9917-ZEBRA" in text
ok2 = "43112" in text
print(f"[{'OK' if ok1 else 'FAIL'}] needle 1 (override code)")
print(f"[{'OK' if ok2 else 'FAIL'}] needle 2 (discrepancy)")
print("RESULT:", "PASS" if ok1 and ok2 else "FAIL")
sys.exit(0 if ok1 and ok2 else 1)
