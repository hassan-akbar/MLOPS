import os
import copy
from collections import Counter

import torch
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from data.dataset_loader import PressureDistributionDataset
from data.preprocessing import get_transforms
from dotenv import load_dotenv

load_dotenv()


# --- Constants ---
def get_dataloaders():
    CLASSES = ["left", "right", "supine", "outofbed", "prone"]
    # Local dataset paths
    ROOT_DIR = "../DATASET"
    TRAIN_PATH = os.path.join(ROOT_DIR, "TRAIN")
    VAL_PATH = os.path.join(ROOT_DIR, "VALIDATION")
    TEST_PATH = os.path.join(ROOT_DIR, "TEST")

    # --- Transforms ---
    train_tfms, val_tfms = get_transforms(img_size=224)

    # --- Load entire dataset ---
    print("🔹 Loading dataset...")
    dataset_rgb = PressureDistributionDataset(
        root_dir=TRAIN_PATH, classes=CLASSES, transform=None
    )

    # Extract numeric labels for stratification
    labels = [label for _, label in dataset_rgb.samples]

    # --- Stratified split (80/20) ---
    train_idx, val_idx = train_test_split(
        list(range(len(dataset_rgb))), test_size=0.2, stratify=labels, random_state=42
    )

    # --- Create copies for separate transforms ---
    train_ds_rgb = copy.copy(dataset_rgb)
    val_ds_rgb = copy.copy(dataset_rgb)

    train_ds_rgb.transform = train_tfms
    val_ds_rgb.transform = val_tfms

    # --- Subsets ---
    train_ds_rgb = Subset(train_ds_rgb, train_idx)
    val_ds_rgb = Subset(val_ds_rgb, val_idx)

    # --- Oversampling for imbalance ---
    train_labels = [dataset_rgb.samples[i][1] for i in train_idx]
    class_counts = torch.bincount(torch.tensor(train_labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[y] for y in train_labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_labels), replacement=True
    )

    # --- Test set (separate validation folder) ---
    test_ds_rgb = PressureDistributionDataset(
        root_dir=VAL_PATH, classes=CLASSES, transform=val_tfms
    )

    # --- DataLoaders ---
    BATCH_SIZE = 32
    NUM_WORKERS = 0

    train_loader_rgb = DataLoader(
        train_ds_rgb,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=NUM_WORKERS,
    )
    val_loader_rgb = DataLoader(
        val_ds_rgb, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader_rgb = DataLoader(
        test_ds_rgb, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    return train_loader_rgb, test_loader_rgb, val_loader_rgb
