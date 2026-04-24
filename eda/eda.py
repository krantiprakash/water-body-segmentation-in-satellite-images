import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
IMAGE_DIR = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Images"
MASK_DIR  = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Masks"
OUT_DIR   = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\eda\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Dataset Overview ────────────────────────────────────────────────────
image_files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')])
mask_files  = sorted([f for f in os.listdir(MASK_DIR)  if f.endswith('.jpg')])

print("=" * 55)
print("1. DATASET OVERVIEW")
print("=" * 55)
print(f"Total images : {len(image_files)}")
print(f"Total masks  : {len(mask_files)}")

missing = [f for f in image_files if f not in set(mask_files)]
print(f"Missing masks: {len(missing)}")

# ── 2. Resolution Analysis ─────────────────────────────────────────────────
widths, heights = [], []
tiny = []

for f in image_files:
    with Image.open(os.path.join(IMAGE_DIR, f)) as img:
        w, h = img.size
        widths.append(w)
        heights.append(h)
        if w < 64 or h < 64:
            tiny.append(f)

print("\n" + "=" * 55)
print("2. RESOLUTION ANALYSIS")
print("=" * 55)
print(f"Min  width : {min(widths)}   Max width : {max(widths)}")
print(f"Min  height: {min(heights)}   Max height: {max(heights)}")
print(f"Mean width : {np.mean(widths):.1f}   Mean height: {np.mean(heights):.1f}")
print(f"Images < 64px (will be filtered): {len(tiny)}")
print(f"Images remaining after filter   : {len(image_files) - len(tiny)}")

# Resolution histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(widths,  bins=50, color='steelblue', edgecolor='white')
axes[0].set_title("Width Distribution")
axes[0].set_xlabel("Width (px)")
axes[0].set_ylabel("Count")
axes[1].hist(heights, bins=50, color='coral', edgecolor='white')
axes[1].set_title("Height Distribution")
axes[1].set_xlabel("Height (px)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "resolution_histogram.png"), dpi=150)
plt.close()
print(f"\nResolution histogram saved → eda/outputs/resolution_histogram.png")

# ── 3. Sample Visualization ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("3. SAMPLE VISUALIZATION")
print("=" * 55)

np.random.seed(42)
samples = np.random.choice(image_files, 6, replace=False)

fig, axes = plt.subplots(6, 2, figsize=(8, 18))
for i, fname in enumerate(samples):
    img  = cv2.cvtColor(cv2.imread(os.path.join(IMAGE_DIR, fname)), cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
    axes[i, 0].imshow(img);  axes[i, 0].set_title(f"Image: {fname}", fontsize=7)
    axes[i, 1].imshow(mask, cmap='gray'); axes[i, 1].set_title("Mask", fontsize=7)
    axes[i, 0].axis('off');  axes[i, 1].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_pairs.png"), dpi=150)
plt.close()
print(f"Sample pairs saved → eda/outputs/sample_pairs.png")

# ── 4. Class Balance ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("4. CLASS BALANCE (water vs non-water pixels)")
print("=" * 55)

water_pcts = []
for f in image_files:
    mask = cv2.imread(os.path.join(MASK_DIR, f), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    pct = (mask > 127).sum() / mask.size * 100
    water_pcts.append(pct)

water_pcts = np.array(water_pcts)
print(f"Mean  water %: {water_pcts.mean():.2f}%")
print(f"Median water %: {np.median(water_pcts):.2f}%")
print(f"Min   water %: {water_pcts.min():.2f}%")
print(f"Max   water %: {water_pcts.max():.2f}%")
print(f"Images with <5%  water: {(water_pcts < 5).sum()}")
print(f"Images with >95% water: {(water_pcts > 95).sum()}")

# Class balance histogram
plt.figure(figsize=(10, 4))
plt.hist(water_pcts, bins=50, color='teal', edgecolor='white')
plt.title("Water Pixel % per Image")
plt.xlabel("Water Pixel %")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_balance.png"), dpi=150)
plt.close()
print(f"Class balance histogram saved → eda/outputs/class_balance.png")

print("\n" + "=" * 55)
print("EDA COMPLETE — check eda/outputs/ for all plots")
print("=" * 55)

# the tiny list has all filenames. We just didn't print them.
# Run this quick one-liner from terminal to see the tiny images and their resolutions.
# python -c "
# import os
# from PIL import Image
# IMAGE_DIR = r'C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Images'
# tiny = [(f, Image.open(os.path.join(IMAGE_DIR, f)).size) for f in sorted(os.listdir(IMAGE_DIR)) if f.endswith('.jpg') and min(Image.open(os.path.join(IMAGE_DIR, f)).size) < 64]
# [print(f[0], f[1]) for f in tiny]
# print('Total:', len(tiny))
# "