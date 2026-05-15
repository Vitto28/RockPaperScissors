"""
merge_datasets.py

Merges the original hand-gestures.csv (separated axes, all 21 landmarks,
raw gesture labels) with the new landmarks.csv (interleaved axes, landmarks
1-20, rps labels) into a single merged_landmarks.csv.

Transformations applied to the original CSV:
  - Drop 'thumbs_up' rows
  - Remap labels: close_palm->rock, open_palm->paper, peace->scissors
  - Re-normalize relative to wrist (landmark 0) — the old classifier.py did
    this at training time via df.apply(normalize), so the raw CSV values are
    NOT yet normalized. We do it here permanently.
  - Reorder from separated [x0..x20, y0..y20, z0..z20] to
    interleaved [x1,y1,z1, ..., x20,y20,z20], dropping landmark 0 after use.

Output columns (match landmarks.csv exactly):
  label, x1,y1,z1, x2,y2,z2, ..., x20,y20,z20

Usage:
  python merge_datasets.py \\
      --old hand-gestures.csv \\
      --new landmarks.csv \\
      --output merged_landmarks.csv
"""

import argparse
import pandas as pd
import sys

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--old', default='hand-gestures.csv',
                    help='Original CSV with separated axes and raw gesture labels')
parser.add_argument('--new', default='landmarks.csv',
                    help='New CSV from extract_landmarks.py')
parser.add_argument('--output', default='merged_landmarks.csv')
args = parser.parse_args()

# ── Target column order (must match landmarks.csv and gesture_detector.py) ────
COORD_COLS = []
for i in range(1, 21):
    COORD_COLS += [f'x{i}', f'y{i}', f'z{i}']
ALL_COLS = ['label'] + COORD_COLS  # 61 columns total

# ── Load and transform old CSV ─────────────────────────────────────────────────
print(f"Loading old dataset: {args.old}")
old = pd.read_csv(args.old)
print(f"  {len(old)} rows before filtering")
print(f"  Label counts:\n{old['label'].value_counts().to_string()}")

# Validate expected columns are present
expected_old_cols = (
    [f'x{i}' for i in range(21)] +
    [f'y{i}' for i in range(21)] +
    [f'z{i}' for i in range(21)] +
    ['label']
)
missing = [c for c in expected_old_cols if c not in old.columns]
if missing:
    print(f"\nERROR: Old CSV is missing expected columns: {missing}")
    print(f"Actual columns: {old.columns.tolist()}")
    sys.exit(1)

# Drop thumbs_up
old = old[old['label'] != 'thumbs_up'].copy()
print(f"\n  After dropping thumbs_up: {len(old)} rows")

# Remap labels
label_map = {
    'close_palm': 'rock',
    'open_palm':  'paper',
    'peace':      'scissors',
}
old['label'] = old['label'].map(label_map)
unmapped = old['label'].isna().sum()
if unmapped > 0:
    print(f"  WARNING: {unmapped} rows had unrecognized labels and will be dropped")
    old = old.dropna(subset=['label'])

# Normalize relative to wrist (landmark 0).
# The old classifier.py did this at train time via apply(normalize, axis=1),
# so the CSV values are still raw (un-normalized). We apply it here once.
for axis in ['x', 'y', 'z']:
    wrist_col = f'{axis}0'
    for i in range(21):
        old[f'{axis}{i}'] -= old[wrist_col]
# After subtraction, x0/y0/z0 are all 0.0 — drop them.

# Reorder: separated axes -> interleaved, landmarks 1-20 only
interleaved_rows = []
for _, row in old.iterrows():
    coords = []
    for i in range(1, 21):
        coords.append(row[f'x{i}'])
        coords.append(row[f'y{i}'])
        coords.append(row[f'z{i}'])
    interleaved_rows.append([row['label']] + coords)

old_transformed = pd.DataFrame(interleaved_rows, columns=ALL_COLS)
print(f"\n  Transformed old dataset: {len(old_transformed)} rows")
print(f"  Label counts:\n{old_transformed['label'].value_counts().to_string()}")

# ── Load new CSV ───────────────────────────────────────────────────────────────
print(f"\nLoading new dataset: {args.new}")
new = pd.read_csv(args.new)
print(f"  {len(new)} rows")
print(f"  Label counts:\n{new['label'].value_counts().to_string()}")

missing_new = [c for c in ALL_COLS if c not in new.columns]
if missing_new:
    print(f"\nERROR: New CSV is missing expected columns: {missing_new}")
    print(f"Actual columns: {new.columns.tolist()}")
    sys.exit(1)

new = new[ALL_COLS]  # enforce column order

# ── Merge ──────────────────────────────────────────────────────────────────────
merged = pd.concat([old_transformed, new], ignore_index=True)
merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"\nMerged dataset: {len(merged)} rows")
print(f"Label counts:\n{merged['label'].value_counts().to_string()}")

merged.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")