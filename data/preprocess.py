import os
import sys
import cv2
import yaml
import numpy as np
from PIL import Image

# ── Load paths from config ─────────────────────────────────────────────────
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "config.yaml"
)
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

IMAGE_DIR  = cfg["paths"]["image_dir"]
MASK_DIR   = cfg["paths"]["mask_dir"]
OUTPUT_TXT = cfg["paths"]["valid_files"]
os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)

MIN_SIZE = 64  # filter threshold

# ── 1. Filter Tiny Images ──────────────────────────────────────────────────
print("=" * 55)
print("1. FILTERING TINY IMAGES (width or height < 64px)")
print("=" * 55)

all_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])

valid_files  = []
dropped_files = []

for f in all_files:
    with Image.open(os.path.join(IMAGE_DIR, f)) as img:
        w, h = img.size
        if w < MIN_SIZE or h < MIN_SIZE:
            dropped_files.append((f, w, h))
        else:
            valid_files.append(f)

print(f"Total images       : {len(all_files)}")
print(f"Dropped (< 64px)   : {len(dropped_files)}")
print(f"Remaining valid    : {len(valid_files)}")
print("\nDropped image list:")
for fname, w, h in dropped_files:
    print(f"  {fname}  →  {w}x{h}")

# ── 2. Verify Mask Binarization ────────────────────────────────────────────
print("\n" + "=" * 55)
print("2. MASK BINARIZATION CHECK")
print("=" * 55)

non_binary_count = 0
non_binary_files = []

for f in valid_files:
    mask = cv2.imread(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"  WARNING: Could not read mask for {f}")
        continue
    unique_vals = np.unique(mask)
    is_binary = all(v in [0, 255] for v in unique_vals)
    if not is_binary:
        non_binary_count += 1
        non_binary_files.append((f, unique_vals.tolist()))

print(f"Valid masks checked       : {len(valid_files)}")
print(f"Perfectly binary masks    : {len(valid_files) - non_binary_count}")
print(f"Masks with intermediate   : {non_binary_count}")

if non_binary_count > 0:
    print("\nSample non-binary masks (first 10):")
    for fname, vals in non_binary_files[:10]:
        print(f"  {fname}  →  unique values: {vals[:10]}{'...' if len(vals) > 10 else ''}")
    print("\n  NOTE: dataset.py will threshold all masks at 127 during loading.")
else:
    print("  All masks are perfectly binary. No thresholding issues.")

# ── 3. Resolution Summary of Valid Files ───────────────────────────────────
print("\n" + "=" * 55)
print("3. VALID FILES RESOLUTION SUMMARY")
print("=" * 55)

widths, heights = [], []
for f in valid_files:
    with Image.open(os.path.join(IMAGE_DIR, f)) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)

print(f"Min  width : {min(widths)}    Max width : {max(widths)}")
print(f"Min  height: {min(heights)}    Max height: {max(heights)}")
print(f"Mean width : {np.mean(widths):.1f}    Mean height: {np.mean(heights):.1f}")

# ── 4. Save valid_files.txt ────────────────────────────────────────────────
with open(OUTPUT_TXT, 'w') as f:
    for fname in valid_files:
        f.write(fname + '\n')

print("\n" + "=" * 55)
print("4. SAVED")
print("=" * 55)
print(f"valid_files.txt saved → {OUTPUT_TXT}")
print(f"Total valid image-mask pairs ready for training: {len(valid_files)}")
print("=" * 55)