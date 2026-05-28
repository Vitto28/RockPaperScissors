"""
extract_landmarks.py

Downloads the Kaggle Rock-Paper-Scissors dataset via kagglehub and extracts
MediaPipe hand landmarks from every image, saving them as a CSV ready for classifier.py.
Applies robust 2D scale and rotation normalization relative to the wrist (0) and middle MCP (9).

Output CSV columns:
    label, x1,y1, x2,y2, ..., x20,y20 (40 columns total)
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
parser.add_argument('--output', default='landmarks.csv', help='Output CSV path')
args = parser.parse_args()

# ── Download dataset ───────────────────────────────────────────────────────────
print("Downloading dataset via kagglehub...")
download_path = Path(kagglehub.dataset_download("sanikamal/rock-paper-scissors-dataset"))
print(f"Downloaded to: {download_path}")

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
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.3,
)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
LABEL_NORM = {'rock': 'rock', 'paper': 'paper', 'scissors': 'scissors', 'scissor': 'scissors'}

def label_from_folder(folder_name: str) -> Optional[str]:
    return LABEL_NORM.get(folder_name.lower().strip())

def label_from_filename(filename: str) -> Optional[str]:
    name = filename.lower()
    for key, mapped in LABEL_NORM.items():
        if re.search(key, name):
            return mapped
    return None

def collect_images(root: Path) -> list:
    items = []
    for split in ['train', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = label_from_folder(class_dir.name)
            if label is None:
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    items.append((img_path, label))

    for val_dir_name in ['validation', 'val']:
        val_dir = root / val_dir_name
        if val_dir.exists():
            for img_path in val_dir.iterdir():
                if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                label = label_from_filename(img_path.name)
                if label is None:
                    continue
                items.append((img_path, label))
            break
    return items

# ── Geometric Normalization Helper ─────────────────────────────────────────────
def normalize_landmarks_2d(lms, w: int, h: int) -> Optional[list]:
    """
    Translates, scales, and rotates 2D landmarks so that:
    - Wrist (0) is at (0,0)
    - Middle MCP (9) is aligned straight up along the negative Y-axis at a distance of 1.0
    - Aspect ratio is corrected before normalization.
    """
    # 1. Translation Normalization & Aspect Ratio Correction
    wrist_x, wrist_y = lms[0].x, lms[0].y
    
    # Pre-multiply normalized coordinates by pixels to restore Euclidean geometry
    rel_x = np.array([(lm.x - wrist_x) * w for lm in lms], dtype=np.float32)
    rel_y = np.array([(lm.y - wrist_y) * h for lm in lms], dtype=np.float32)
    
    # 2. Scale Normalization (using distance from wrist (0) to middle MCP (9))
    scale = np.hypot(rel_x[9], rel_y[9])
    if scale < 1e-6:
        return None
    
    rel_x /= scale
    rel_y /= scale
    
    # 3. Rotation Normalization (align vector 0->9 straight UP)
    # In screen coordinates, up is negative Y. We want joint 9 to be at (0, -1)
    angle = np.arctan2(rel_y[9], rel_x[9])
    rotation_angle = -angle - np.pi / 2
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    
    features = []
    # Interleaved X, Y coordinates for landmarks 1-20
    for i in range(1, 21):
        rot_x = rel_x[i] * cos_a - rel_y[i] * sin_a
        rot_y = rel_x[i] * sin_a + rel_y[i] * cos_a
        features.extend([rot_x, rot_y])
    
    # 4. Feature Engineering: Fingertip Distances (Landmarks 4, 8, 12, 16, 20)
    # These provide explicit extension signals (Paper vs Scissors)
    for i in [4, 8, 12, 16, 20]:
        dist = np.hypot(rel_x[i], rel_y[i])
        features.append(dist)
        
    return features

def extract_features(image_path: Path) -> Optional[np.ndarray]:
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None
    
    h, w, _ = img_bgr.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    lms = results.multi_hand_landmarks[0].landmark
    norm_features = normalize_landmarks_2d(lms, w, h)
    
    return np.array(norm_features, dtype=np.float32) if norm_features is not None else None

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
            continue
        rows.append([label] + features.tolist())

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(images)} (skipped: {skipped})")

    print(f"\nDone. Extracted {len(rows)} rows, skipped {skipped} images.")

    coord_cols = []
    for i in range(1, 21):
        coord_cols += [f'x{i}', f'y{i}']
    
    # Add engineered distance columns
    dist_cols = [f'd{i}' for i in [4, 8, 12, 16, 20]]
    cols = ['label'] + coord_cols + dist_cols

    df = pd.DataFrame(rows, columns=cols)
    print(df['label'].value_counts().to_string())

    df.to_csv(args.output, index=False)
    print(f"Saved to: {args.output}")

if __name__ == '__main__':
    main()