import os
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────
image_dir = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Images"
mask_dir  = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Masks"

# ── Load all image files ───────────────────────────────────────────────────
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files  = sorted([f for f in os.listdir(mask_dir)  if f.endswith('.jpg')])

print(f"Total images : {len(image_files)}")
print(f"Total masks  : {len(mask_files)}")

# ── Check missing masks ────────────────────────────────────────────────────
missing_masks = [f for f in image_files if f not in set(mask_files)]
if missing_masks:
    print(f"Images without masks: {missing_masks}")
else:
    print("All images have corresponding masks.")

# ── Build resolution dict ──────────────────────────────────────────────────
image_resolutions = {}
for img_file in image_files:
    with Image.open(os.path.join(image_dir, img_file)) as img:
        image_resolutions[img_file] = img.size  # (width, height)

# ── FULL DATASET (all 2841 images) ────────────────────────────────────────
print("\n" + "=" * 55)
print("FULL DATASET (all 2841 images)")
print("=" * 55)

# Smallest image by min(width, height)
min_img  = min(image_resolutions, key=lambda f: min(image_resolutions[f]))
min_res  = image_resolutions[min_img]

# Largest image by max(width, height)
max_img  = max(image_resolutions, key=lambda f: max(image_resolutions[f]))
max_res  = image_resolutions[max_img]

# Smallest image by area (width * height)
min_area_img = min(image_resolutions, key=lambda f: image_resolutions[f][0] * image_resolutions[f][1])
min_area_res = image_resolutions[min_area_img]

# Largest image by area
max_area_img = max(image_resolutions, key=lambda f: image_resolutions[f][0] * image_resolutions[f][1])
max_area_res = image_resolutions[max_area_img]

print(f"Smallest (min side)  : {min_img} → {min_res[0]}x{min_res[1]}")
print(f"Largest  (max side)  : {max_img} → {max_res[0]}x{max_res[1]}")
print(f"Smallest (by area)   : {min_area_img} → {min_area_res[0]}x{min_area_res[1]}")
print(f"Largest  (by area)   : {max_area_img} → {max_area_res[0]}x{max_area_res[1]}")

# ── FILTERED DATASET (after removing images < 64px) ───────────────────────
print("\n" + "=" * 55)
print("FILTERED DATASET (after removing < 64px images)")
print("=" * 55)

filtered = {f: r for f, r in image_resolutions.items()
            if r[0] >= 64 and r[1] >= 64}
dropped  = {f: r for f, r in image_resolutions.items()
            if r[0] < 64 or r[1] < 64}

print(f"Total after filter   : {len(filtered)}")
print(f"Dropped (< 64px)     : {len(dropped)}")

# Smallest in filtered dataset
min_filt_img = min(filtered, key=lambda f: min(filtered[f]))
min_filt_res = filtered[min_filt_img]

# Largest in filtered dataset
max_filt_img = max(filtered, key=lambda f: max(filtered[f]))
max_filt_res = filtered[max_filt_img]

# Smallest by area in filtered
min_filt_area_img = min(filtered, key=lambda f: filtered[f][0] * filtered[f][1])
min_filt_area_res = filtered[min_filt_area_img]

# Largest by area in filtered
max_filt_area_img = max(filtered, key=lambda f: filtered[f][0] * filtered[f][1])
max_filt_area_res = filtered[max_filt_area_img]

print(f"\nSmallest (min side)  : {min_filt_img} → {min_filt_res[0]}x{min_filt_res[1]}")
print(f"Largest  (max side)  : {max_filt_img} → {max_filt_res[0]}x{max_filt_res[1]}")
print(f"Smallest (by area)   : {min_filt_area_img} → {min_filt_area_res[0]}x{min_filt_area_res[1]}")
print(f"Largest  (by area)   : {max_filt_area_img} → {max_filt_area_res[0]}x{max_filt_area_res[1]}")

# ── DROPPED IMAGES LIST ────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("DROPPED IMAGES (< 64px in any dimension)")
print("=" * 55)
for f, r in sorted(dropped.items(), key=lambda x: min(x[1])):
    print(f"  {f} → {r[0]}x{r[1]}")