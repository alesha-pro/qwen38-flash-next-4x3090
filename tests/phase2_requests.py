#!/usr/bin/env python3
"""Phase 2 request driver: text + vision battery against the local server.

Sends deterministic requests (temperature 0) to http://127.0.0.1:8018 and
writes inputs+outputs JSON to the run directory.
"""
import base64
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8018"
RUN = sys.argv[1]
IMG = os.path.join(RUN, "inputs")
OUT = os.path.join(RUN, "requests.jsonl")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def chat(name, content, max_tokens=128):
    body = {
        "model": "/model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.load(r)
    dt = time.time() - t0
    rec = {"name": name, "request": body, "response": resp,
           "wall_seconds": round(dt, 2)}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    print(f"[{name}] {dt:.1f}s prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')}\n{text}\n---",
          flush=True)
    return text


def img_item(name):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64(os.path.join(IMG, name))}"}}


if os.path.exists(OUT):
    os.remove(OUT)

# T1: text coherence
chat("text_coherence",
     "The capital of France is", max_tokens=64)

# T2: deterministic arithmetic + JSON (PLE exercises common ngrams)
chat("text_arith_json",
     "Compute 17*23+5 and answer with a JSON object "
     "{\"result\": <number>} only.", max_tokens=64)

# V1: OCR
chat("vision_ocr",
     [{"type": "text", "text": "Read every line of text in this image "
       "exactly, one per line."},
      img_item("ocr_text.png")], max_tokens=128)

# V2: chart reading
chat("vision_chart",
     [{"type": "text", "text": "This is a bar chart. List each category and "
       "its value, and name the category with the highest bar."},
      img_item("chart_bars.png")], max_tokens=160)

# V3: spatial relations
chat("vision_spatial",
     [{"type": "text", "text": "Describe the position of each shape: the red "
       "circle, the blue square and the green triangle. Which is on the "
       "left, which on the right, which at the top?"},
      img_item("spatial_shapes.png")], max_tokens=160)

# V4: fine detail
chat("vision_detail",
     [{"type": "text", "text": "This image is a numbered 8x8 grid. One cell "
       "has a yellow background. What number is written in the yellow cell?"},
      img_item("detail_grid.png")], max_tokens=96)

# V5: multi-image
chat("vision_multi",
     [{"type": "text", "text": "I show two images. Describe the differences "
       "between the first and the second image."},
      img_item("multi_day.png"),
      img_item("multi_night.png")], max_tokens=192)

print("ALL_REQUESTS_DONE")
