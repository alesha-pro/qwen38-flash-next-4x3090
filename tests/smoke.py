#!/usr/bin/env python3
import base64
import json
import sys
import urllib.request
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8018
base = f"http://127.0.0.1:{port}"


def request(content, max_tokens=128):
    payload = {
        "model": "/model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{base}/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as response:
        return json.load(response)["choices"][0]["message"]["content"]


text = request("Answer with exactly the number: 17*23+5")
if "396" not in text:
    raise SystemExit(f"text smoke failed: {text!r}")
print("TEXT SMOKE PASS")

image_path = Path(__file__).parent / "assets" / "ocr_text.png"
encoded = base64.b64encode(image_path.read_bytes()).decode()
vision = request([
    {"type": "text", "text": "Read the order number in this image. Return only it."},
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
], max_tokens=256)
if "Q-8842-XK" not in vision:
    raise SystemExit(f"Vision smoke failed: {vision!r}")
print("VISION SMOKE PASS")
