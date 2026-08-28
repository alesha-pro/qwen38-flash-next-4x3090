#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Vision smoke test: image understanding + text-after-image + prefix reuse.

Usage: vision_test.py [port]
Sends the generated 2x2 color grid (tests/assets/test-grid.png) via chat
completions and checks the model identifies the quadrant colors; then asks a
text-only follow-up in the same conversation; then repeats the image request
to exercise the multimodal prefix cache.
"""
import base64
import json
import sys
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8018

with open("/mnt/nvme2/projects/qwen38-kv-quantize/tests/assets/test-grid.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
data_uri = f"data:image/png;base64,{b64}"


def chat(messages, max_tokens=128):
    payload = {"model": "/model", "messages": messages, "max_tokens": max_tokens,
               "temperature": 0}
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.load(resp)["choices"][0]["message"]["content"]


failures = 0

q1 = ("This image is a 2x2 grid of four solid color quadrants. "
      "Name the color of each quadrant: top-left, top-right, bottom-left, bottom-right.")
msgs1 = [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": data_uri}},
    {"type": "text", "text": q1},
]}]
a1 = chat(msgs1)
print(f"A1 (image understanding): {a1!r}")
ok1 = all(c in a1.lower() for c in ("red", "blue", "green", "yellow"))
print(f"[{'OK' if ok1 else 'FAIL'}] all four quadrant colors named")
failures += 0 if ok1 else 1

msgs2 = msgs1 + [{"role": "assistant", "content": a1},
                 {"role": "user", "content": "Without looking at the image again: which color was top-left? Answer with one word."}]
a2 = chat(msgs2)
print(f"A2 (text-after-image): {a2!r}")
ok2 = "red" in a2.lower()
print(f"[{'OK' if ok2 else 'FAIL'}] text-after-image recalls top-left=red")
failures += 0 if ok2 else 1

a3 = chat(msgs1)
print(f"A3 (multimodal prefix reuse): {a3!r}")
ok3 = a3 == a1
print(f"[{'OK' if ok3 else 'FAIL'}] repeated image request is deterministic")
failures += 0 if ok3 else 1

print("RESULT:", "PASS" if failures == 0 else f"FAIL ({failures})")
sys.exit(1 if failures else 0)
