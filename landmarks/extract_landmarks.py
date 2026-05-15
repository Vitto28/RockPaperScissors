"""
extract_landmarks.py

Downloads the Kaggle Rock-Paper-Scissors dataset via kagglehub and extracts
MediaPipe hand landmarks from every image, saving them as a CSV ready for
classifier.py.

Expected dataset layout (inside the kagglehub download path):
    Rock-Paper-Scissors/
        train/
            Rock/   Paper/   Scissors/
        test/
            Rock/   Paper/   Scissors/
        validation/
            *.png  (label encoded in filename, e.g. "rock07.png", "TEST_SCISSORS_1.jpg")

Output CSV columns:
    label, x1,y1,z1, x2,y2,z2, ..., x20,y20,z20
    (63 columns total — landmark 0 is the wrist used for normalization and dropped)

Usage:
    python extract_landmarks.py --output landmarks.csv
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import re
import argparse
from pathlib import Path
from typing import Optional

import kagglehub

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--output', default='landmarks.csv',
                    help='Output CSV path')
args = parser.parse_args()

# ── Download dataset ───────────────────────────────────────────────────────────
print("Downloading dataset via kagglehub...")
download_path = Path(kagglehub.dataset_download("sanikamal/rock-paper-scissors-dataset"))
print(f"Downloaded to: {download_path}")

# kagglehub sometimes wraps the content in a version subfolder.
# Find the directory that actually contains train/test/validation.
def find_dataset_root(base: Path) -> Path:
    for candidate in [base] + sorted(base.rglob('train'))[:1]:
        root = candidate if candidate.name != 'train' else candidate.parent
        if (root / 'train').exists() or (root / 'test').exists():
            return root
    raise FileNotFoundError(f"Could not find train/test folders under {base}")

ROOT = find_dataset_root(download_path)
print(f"Dataset root: {ROOT}")

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,   # important: each image is independent
    max_num_hands=1,
    model_complexity=1,        # use full model for offline extraction accuracy
    min_detection_confidence=0.3,  # low threshold — clean studio images should be easy
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# ── Label helpers ──────────────────────────────────────────────────────────────
LABEL_NORM = {
    'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors',
    'scissor': 'scissors',  # some datasets spell it without the s
}

def label_from_folder(folder_name: str) -> Optional[str]:
    """Infer label from a subfolder name like 'Rock', 'Paper', 'Scissors'."""
    return LABEL_NORM.get(folder_name.lower().strip())

def label_from_filename(filename: str) -> Optional[str]:
    """Infer label from a filename using a case-insensitive regex search."""
    name = filename.lower()
    for key, mapped in LABEL_NORM.items():
        if re.search(key, name):
            return mapped
    return None

# ── Image collector ────────────────────────────────────────────────────────────
def collect_images(root: Path) -> list:
    """
    Returns a list of (image_path, label) pairs.
    Handles train/test subfolders (label from subfolder name) and the
    flat validation folder (label from filename).
    """
    items = []

    for split in ['train', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  [skip] {split_dir} not found")
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = label_from_folder(class_dir.name)
            if label is None:
                print(f"  [skip] unrecognized folder: {class_dir}")
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((img_path, label))

    # Validation: flat folder, label from filename
    for val_dir_name in ['validation', 'val']:
        val_dir = root / val_dir_name
        if val_dir.exists():
            for img_path in val_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = label_from_filename(img_path.name)
                if label is None:
                    print(f"  [skip] can't infer label from: {img_path.name}")
                    continue
                items.append((img_path, label))
            break

    return items

# ── Landmark extraction ────────────────────────────────────────────────────────
def extract_features(image_path: Path) -> Optional[np.ndarray]:
    """
    Returns a (60,) float array of wrist-normalized landmark coordinates,
    or None if MediaPipe failed to detect a hand.

    Layout: [x1,y1,z1, x2,y2,z2, ..., x20,y20,z20]
    Landmark 0 (wrist) is used for normalization and then discarded,
    so indices 1-20 -> 20 landmarks x 3 axes = 60 values.
    """
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    lms = results.multi_hand_landmarks[0].landmark  # take first hand only

    wrist_x, wrist_y, wrist_z = lms[0].x, lms[0].y, lms[0].z

    features = []
    for i in range(1, 21):  # skip landmark 0 (wrist) after normalizing
        features.append(lms[i].x - wrist_x)
        features.append(lms[i].y - wrist_y)
        features.append(lms[i].z - wrist_z)

    return np.array(features, dtype=np.float32)

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print(f"Scanning dataset at: {ROOT.resolve()}")
    images = collect_images(ROOT)
    print(f"Found {len(images)} images total\n")

    rows = []
    skipped = 0

    for i, (img_path, label) in enumerate(images):
        features = extract_features(img_path)
        if features is None:
            skipped += 1
            if skipped <= 10:  # only print first few to avoid spam
                print(f"  [no hand detected] {img_path.name}")
            continue
        rows.append([label] + features.tolist())

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images)}  (skipped so far: {skipped})")

    print(f"\nDone. Extracted {len(rows)} rows, skipped {skipped} images.")

    # Build column names: label, x1,y1,z1, ..., x20,y20,z20
    coord_cols = []
    for i in range(1, 21):
        coord_cols += [f'x{i}', f'y{i}', f'z{i}']

    df = pd.DataFrame(rows, columns=['label'] + coord_cols)
    print(df['label'].value_counts().to_string())

    df.to_csv(args.output, index=False)
    print(f"\nSaved to: {args.output}")

if __name__ == '__main__':
    main()