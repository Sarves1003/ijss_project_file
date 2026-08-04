#!/usr/bin/env python3
"""
Builds a labeled contact-sheet figure from the real physical myCobot 280
deployment photos (h1-h5, provided by the user) plus the operator dashboard
screenshot, for the manuscript's hardware-implementation section. No image
content is synthesized -- this only crops/resizes/labels the real photos.
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
HWDIR = os.path.join(HERE, "..", "figures", "hardware")

PANELS = [
    ("hw_h1.png", "(a) Camera-and-gripper end effector, conveyor view"),
    ("hw_h2.png", "(b) Full arm, side view (home-adjacent pose)"),
    ("hw_h5.png", "(c) Full arm, extended reach pose"),
    ("hw_h3.png", "(d) Arm and gripper, approach pose"),
    ("hw_h4.png", "(e) Camera/gripper module close-up"),
    ("hw_dashboard.png", "(f) Operator dashboard: ROI zone, live detection"),
]

TARGET_W, TARGET_H = 560, 420
PAD = 14
LABEL_H = 34
COLS, ROWS = 3, 2

font = None
for cand in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
    if os.path.exists(cand):
        font = ImageFont.truetype(cand, 18)
        break
if font is None:
    font = ImageFont.load_default()

cell_w = TARGET_W + 2 * PAD
cell_h = TARGET_H + LABEL_H + 2 * PAD
sheet = Image.new("RGB", (COLS * cell_w, ROWS * cell_h), "white")
draw = ImageDraw.Draw(sheet)

for idx, (fname, label) in enumerate(PANELS):
    img = Image.open(os.path.join(HWDIR, fname)).convert("RGB")
    # Fit into TARGET_W x TARGET_H preserving aspect ratio, centered on white
    scale = min(TARGET_W / img.width, TARGET_H / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), "black")
    canvas.paste(img_resized, ((TARGET_W - new_w) // 2, (TARGET_H - new_h) // 2))

    col, row = idx % COLS, idx // COLS
    x0 = col * cell_w + PAD
    y0 = row * cell_h + PAD
    sheet.paste(canvas, (x0, y0))
    draw.rectangle([x0, y0, x0 + TARGET_W, y0 + TARGET_H], outline="black", width=2)

    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    draw.text((x0 + (TARGET_W - text_w) // 2, y0 + TARGET_H + 6), label, fill="black", font=font)

out_png = os.path.join(HWDIR, "hardware_setup.png")
sheet.save(out_png, dpi=(300, 300))

print(f"Saved {out_png} ({sheet.width}x{sheet.height})")
