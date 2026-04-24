import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Paths ──────────────────────────────────────────────────────────────────
IMAGE_DIR      = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Images"
MASK_DIR       = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\Water Bodies Dataset\Masks"
VALID_FILES    = r"C:\Users\RadheRadhe\Desktop\Self project\CV\Aereo\data\valid_files.txt"

# ── Config ─────────────────────────────────────────────────────────────────
IMAGE_SIZE  = 256
BATCH_SIZE  = 16
NUM_WORKERS = 2
RANDOM_SEED = 42

# ── Augmentations ──────────────────────────────────────────────────────────
def get_transforms(mode):
    if mode == "train":
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:  # val / test
        return A.Compose([
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

# ── Dataset Class ──────────────────────────────────────────────────────────
class WaterBodyDataset(Dataset):
    def __init__(self, file_list, image_dir, mask_dir, mode="train"):
        self.file_list  = file_list
        self.image_dir  = image_dir
        self.mask_dir   = mask_dir
        self.transforms = get_transforms(mode)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]

        # Load image (BGR → RGB)
        img_path = os.path.join(self.image_dir, fname)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (H, W, 3)

        # Load mask (grayscale)
        mask_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # (H, W)

        # Threshold mask → binary 0 or 1
        mask = (mask > 127).astype(np.float32)  # (H, W)

        # Apply transforms (resize + aug + normalize + to tensor)
        augmented = self.transforms(image=image, mask=mask)
        image = augmented["image"]  # (3, 256, 256) float32
        mask  = augmented["mask"]   # (256, 256) float32

        # Add channel dim to mask → (1, 256, 256)
        mask = mask.unsqueeze(0)

        return image, mask

# ── Split + DataLoaders ────────────────────────────────────────────────────
def get_dataloaders(
    valid_files_path=VALID_FILES,
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    seed=RANDOM_SEED,
    max_samples=None,
):
    # Read valid filenames
    with open(valid_files_path, "r") as f:
        files = [line.strip() for line in f.readlines()]

    # Shuffle with fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    files = list(rng.permutation(files))

    # Debug mode — limit samples
    if max_samples is not None:
        files = files[:max_samples]

    # 80 / 10 / 10 split
    n       = len(files)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    train_files = files[:n_train]
    val_files   = files[n_train : n_train + n_val]
    test_files  = files[n_train + n_val:]

    print("=" * 45)
    print("DATASET SPLIT")
    print("=" * 45)
    print(f"Total valid files : {n}")
    print(f"Train             : {len(train_files)}")
    print(f"Val               : {len(val_files)}")
    print(f"Test              : {len(test_files)}")
    print("=" * 45)

    # Datasets
    train_ds = WaterBodyDataset(train_files, image_dir, mask_dir, mode="train")
    val_ds   = WaterBodyDataset(val_files,   image_dir, mask_dir, mode="val")
    test_ds  = WaterBodyDataset(test_files,  image_dir, mask_dir, mode="test")

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Quick Sanity Check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    # Check one batch
    images, masks = next(iter(train_loader))
    print(f"Image batch shape : {images.shape}")   # (16, 3, 256, 256)
    print(f"Mask  batch shape : {masks.shape}")    # (16, 1, 256, 256)
    print(f"Image dtype       : {images.dtype}")   # float32
    print(f"Mask  dtype       : {masks.dtype}")    # float32
    print(f"Image min/max     : {images.min():.3f} / {images.max():.3f}")
    print(f"Mask  unique vals : {masks.unique()}")  # tensor([0., 1.])
    print("\ndataset.py sanity check passed.")