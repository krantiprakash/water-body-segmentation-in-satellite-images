import os
from PIL import Image

# Define paths
image_dir = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Images"
mask_dir = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Masks"

# Get list of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.jpg')]

# Count images
num_images = len(image_files)
print(f"Number of images: {num_images}")

# Check for corresponding masks
missing_masks = []
for img_file in image_files:
    if img_file not in mask_files:
        missing_masks.append(img_file)

if missing_masks:
    print(f"Images without corresponding masks: {missing_masks}")
else:
    print("All images have corresponding masks.")

# Check resolutions and match
image_resolutions = {}
mask_resolutions = {}
mismatched_resolutions = []
small_images_count = 0

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    mask_path = os.path.join(mask_dir, img_file)
    try:
        with Image.open(img_path) as img:
            img_res = img.size  # (width, height)
            image_resolutions[img_file] = img_res
            if img_res[0] < 64 or img_res[1] < 64:
                small_images_count += 1
        with Image.open(mask_path) as mask:
            mask_res = mask.size
            mask_resolutions[img_file] = mask_res
        if img_res != mask_res:
            mismatched_resolutions.append((img_file, img_res, mask_res))
    except Exception as e:
        print(f"Error reading {img_file}: {e}")

print(f"Number of images smaller than 64x64: {small_images_count}")

print(f"Total images checked: {len(image_resolutions)}")
print(f"Total masks checked: {len(mask_resolutions)}")

if mismatched_resolutions:
    print("Images with mismatched resolutions (image vs mask):")
    for img, img_res, mask_res in mismatched_resolutions:
        print(f"  {img}: Image {img_res}, Mask {mask_res}")
else:
    print("All images and masks have matching resolutions.")

# Unique image resolutions
unique_image_res = set(image_resolutions.values())
print(f"Unique image resolutions: {sorted(unique_image_res)}")

# Unique mask resolutions
unique_mask_res = set(mask_resolutions.values())
print(f"Unique mask resolutions: {sorted(unique_mask_res)}")