#!/usr/bin/env python3
"""Integration battery: frozen Phase 2 text/Vision checks with verdicts.

Same prompts and images as tests/phase2_requests.py, but with enough
max_tokens for the reasoning model to finish, plus automatic PASS/FAIL
verdicts. Writes requests.jsonl and results.jsonl into the run directory.

Usage: battery.py <run_dir> [port] [max_tokens]
Expects <run_dir>/inputs/ produced by tests/phase2_gen_test_images.py.
"""
import base64
import json
import os
import re
import sys
import time
import urllib.request

RUN = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8018
DEFAULT_MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 768
BASE = f"http://127.0.0.1:{PORT}"
IMG = os.path.join(RUN, "inputs")
REQ_OUT = os.path.join(RUN, "requests.jsonl")
RES_OUT = os.path.join(RUN, "results.jsonl")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def img_item(name):
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64(os.path.join(IMG, name))}"}}


def chat(name, content, max_tokens=None):
    body = {
        "model": "/model",
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.load(r)
    dt = time.time() - t0
    rec = {"name": name, "request": body, "response": resp,
           "wall_seconds": round(dt, 2)}
    with open(REQ_OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    choice = resp["choices"][0]
    text = choice["message"]["content"]
    usage = resp.get("usage", {})
    print(f"[{name}] {dt:.1f}s prompt={usage.get('prompt_tokens')} "
          f"completion={usage.get('completion_tokens')} "
          f"finish={choice.get('finish_reason')}", flush=True)
    return text, resp


def final_answer(text):
    """Strip reasoning: answer is after the last </think>, else full text."""
    idx = text.rfind("</think>")
    return text[idx + len("</think>"):].strip() if idx >= 0 else text.strip()


def verdict(name, ok, answer, extra=""):
    rec = {"name": name, "verdict": "PASS" if ok else "FAIL",
           "final_answer": answer, "note": extra}
    with open(RES_OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  -> {'PASS' if ok else 'FAIL'} {extra}\n  answer: {answer[:300]}",
          flush=True)
    return ok


results = []

# T1: text coherence
text, _ = chat("text_coherence", "The capital of France is", max_tokens=384)
ans = final_answer(text)
results.append(verdict("text_coherence", "Paris" in ans, ans))

# T2: arithmetic 17*23+5 = 396
text, _ = chat("text_arith_json",
               "Compute 17*23+5 and answer with a JSON object "
               "{\"result\": <number>} only.", max_tokens=512)
ans = final_answer(text)
results.append(verdict("text_arith_json", bool(re.search(r'"?result"?\s*:\s*396', ans))
                       or "396" in ans, ans))

# V1: OCR, four exact lines
text, _ = chat("vision_ocr",
               [{"type": "text", "text": "Read every line of text in this image "
                 "exactly, one per line."},
                img_item("ocr_text.png")])
ans = final_answer(text)
expected = ["QWEN38 PLE VISION TEST", "Order number: Q-8842-XK",
            "Total due: 137.42 EUR", "Date: 2026-08-27"]
missing = [ln for ln in expected if ln not in ans]
results.append(verdict("vision_ocr", not missing, ans,
                       f"missing={missing}" if missing else "4/4 lines exact"))

# V2: chart reading
text, _ = chat("vision_chart",
               [{"type": "text", "text": "This is a bar chart. List each category and "
                 "its value, and name the category with the highest bar."},
                img_item("chart_bars.png")])
ans = final_answer(text).lower()
ok = all(f"{c}" in ans for c in ("cats", "dogs", "birds", "fish")) \
    and all(v in ans for v in ("30", "55", "15", "45")) \
    and re.search(r"(highest|largest|most|maximum|max)[^.]*dogs|dogs[^.]*"
                  r"(highest|largest|most|maximum|max|55)", ans)
results.append(verdict("vision_chart", bool(ok), ans))

# V3: spatial relations
text, _ = chat("vision_spatial",
               [{"type": "text", "text": "Describe the position of each shape: the red "
                 "circle, the blue square and the green triangle. Which is on the "
                 "left, which on the right, which at the top?"},
                img_item("spatial_shapes.png")])
ans = final_answer(text).lower()
ok = re.search(r"circle[^.]*left|left[^.]*circle", ans) \
    and re.search(r"square[^.]*right|right[^.]*square", ans) \
    and re.search(r"triangle[^.]*top|top[^.]*triangle", ans)
results.append(verdict("vision_spatial", bool(ok), ans))

# V4: fine detail = 43
text, _ = chat("vision_detail",
               [{"type": "text", "text": "This image is a numbered 8x8 grid. One cell "
                 "has a yellow background. What number is written in the yellow cell?"},
                img_item("detail_grid.png")])
ans = final_answer(text)
results.append(verdict("vision_detail", bool(re.search(r"\b43\b", ans)), ans))

# V5: multi-image day/night
text, _ = chat("vision_multi",
               [{"type": "text", "text": "I show two images. Describe the differences "
                 "between the first and the second image."},
                img_item("multi_day.png"),
                img_item("multi_night.png")], max_tokens=1024)
ans = final_answer(text).lower()
ok = ("day" in ans and "night" in ans) and ("sun" in ans and "moon" in ans)
results.append(verdict("vision_multi", ok, ans))

n_pass = sum(results)
print(f"BATTERY: {n_pass}/{len(results)} PASS")
sys.exit(0 if n_pass == len(results) else 1)
