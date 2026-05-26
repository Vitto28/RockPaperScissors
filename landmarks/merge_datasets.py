"""
merge_datasets.py

Merges old hand-gestures.csv with new landmarks.csv.
Applies 2D translation, scale, and rotation normalization to old raw landmarks
while discarding the Z-axis entirely to ensure structural compatibility.
"""

import argparse
import pandas as pd
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--old', default='hand-gestures.csv')
parser.add_argument('--new', default='landmarks.csv')
parser.add_argument('--output', default='merged_landmarks.csv')
args = parser.parse_args()

COORD_COLS = []
for i in range(1, 21):
    COORD_COLS += [f'x{i}', f'y{i}']

# Add engineered distance columns
DIST_COLS = [f'd{i}' for i in [4, 8, 12, 16, 20]]
ALL_COLS = ['label'] + COORD_COLS + DIST_COLS

print(f"Loading old dataset: {args.old}")
old = pd.read_csv(args.old)

# Drop thumbs_up and remap labels
old = old[old['label'] != 'thumbs_up'].copy()
label_map = {'close_palm': 'rock', 'open_palm': 'paper', 'peace': 'scissors'}
old['label'] = old['label'].map(label_map)
old = old.dropna(subset=['label'])

interleaved_rows = []
for _, row in old.iterrows():
    # 1. Translation relative to wrist (0)
    wx, wy = row['x0'], row['y0']
    
    # Note: We assume 1:1 aspect ratio for old data since original dimensions are unknown
    rx = np.array([row[f'x{i}'] - wx for i in range(21)])
    ry = np.array([row[f'y{i}'] - wy for i in range(21)])
    
    # 2. Scale normalization relative to landmark 9
    scale = np.hypot(rx[9], ry[9])
    if scale < 1e-6:
        continue
    rx /= scale
    ry /= scale
    
    # 3. Rotation alignment (Middle MCP points straight up)
    angle = np.arctan2(ry[9], rx[9])
    rotation_angle = -angle - np.pi / 2
    cos_a, sin_a = np.cos(rotation_angle), np.sin(rotation_angle)
    
    coords = []
    for i in range(1, 21):
        rot_x = rx[i] * cos_a - ry[i] * sin_a
        rot_y = rx[i] * sin_a + ry[i] * cos_a
        coords.extend([rot_x, rot_y])
    
    # 4. Feature Engineering: Fingertip Distances
    for i in [4, 8, 12, 16, 20]:
        dist = np.hypot(rx[i], ry[i])
        coords.append(dist)
        
    interleaved_rows.append([row['label']] + coords)

old_transformed = pd.DataFrame(interleaved_rows, columns=ALL_COLS)
print(f"Transformed old dataset: {len(old_transformed)} rows")

print(f"\nLoading new dataset: {args.new}")
new = pd.read_csv(args.new)
new = new[ALL_COLS]  # Slice down to just x and y columns

merged = pd.concat([old_transformed, new], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nFinal Merged dataset: {len(merged)} rows")
print(merged['label'].value_counts().to_string())

merged.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")