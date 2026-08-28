#!/usr/bin/env python3
"""Generate deterministic Phase 2 vision test images (CPU only)."""
import os
import sys

from PIL import Image, ImageDraw

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)


def save(img, name):
    img.save(os.path.join(OUT, name))
    print(name, img.size)


# 1. OCR: clear text on white
img = Image.new("RGB", (800, 300), "white")
d = ImageDraw.Draw(img)
d.text((40, 60), "QWEN38 PLE VISION TEST", fill="black")
d.text((40, 120), "Order number: Q-8842-XK", fill="black")
d.text((40, 180), "Total due: 137.42 EUR", fill="black")
d.text((40, 240), "Date: 2026-08-27", fill="black")
save(img, "ocr_text.png")

# 2. Chart: labeled bar chart
img = Image.new("RGB", (800, 600), "white")
d = ImageDraw.Draw(img)
vals = [("cats", 30, "red"), ("dogs", 55, "green"), ("birds", 15, "blue"),
        ("fish", 45, "orange")]
d.text((300, 20), "Pet survey 2026", fill="black")
for i, (label, v, color) in enumerate(vals):
    x = 100 + i * 170
    h = int(v / 55 * 400)
    d.rectangle([x, 500 - h, x + 100, 500], fill=color)
    d.text((x + 20, 510), label, fill="black")
    d.text((x + 30, 500 - h - 25), str(v), fill="black")
d.line([60, 500, 760, 500], fill="black", width=2)
save(img, "chart_bars.png")

# 3. Spatial relations: shapes
img = Image.new("RGB", (700, 400), "white")
d = ImageDraw.Draw(img)
d.ellipse([50, 120, 190, 260], fill="red")            # red circle left
d.rectangle([480, 100, 640, 260], fill="blue")        # blue square right
d.polygon([(350, 40), (280, 160), (420, 160)], fill="green")  # green triangle top center
d.line([0, 350, 700, 350], fill="black", width=3)
save(img, "spatial_shapes.png")

# 4. Fine detail: numbered grid 8x8 with a marked cell
img = Image.new("RGB", (800, 800), "white")
d = ImageDraw.Draw(img)
n = 0
for r in range(8):
    for c in range(8):
        x0, y0 = 10 + c * 98, 10 + r * 98
        fill = "yellow" if (r, c) == (5, 3) else "white"
        d.rectangle([x0, y0, x0 + 96, y0 + 96], outline="black", fill=fill)
        d.text((x0 + 40, y0 + 42), str(n), fill="black")
        n += 1
save(img, "detail_grid.png")

# 5a/5b. Multi-image pair: two simple scenes
img = Image.new("RGB", (500, 300), "skyblue")
d = ImageDraw.Draw(img)
d.ellipse([350, 30, 450, 130], fill="yellow")
d.rectangle([0, 220, 500, 300], fill="green")
d.text((180, 140), "day scene", fill="black")
save(img, "multi_day.png")

img = Image.new("RGB", (500, 300), "navy")
d = ImageDraw.Draw(img)
d.ellipse([350, 30, 430, 110], fill="white")
d.rectangle([0, 220, 500, 300], fill="darkgreen")
d.text((170, 140), "night scene", fill="white")
save(img, "multi_night.png")
print("done ->", OUT)
