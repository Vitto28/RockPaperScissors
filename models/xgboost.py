"""
classifier.py

Trains an XGBoost classifier on landmarks extracted by extract_landmarks.py.

Input CSV columns:
    label, x1,y1,z1, x2,y2,z2, ..., x20,y20,z20

Outputs:
    rps_model.pkl        — trained XGBoost model
    label_encoder.pkl    — fitted LabelEncoder

Usage:
    python classifier.py --input landmarks.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--input', default='landmarks.csv', help='Landmarks CSV from extract_landmarks.py')
parser.add_argument('--model_out', default='rps_model.pkl')
parser.add_argument('--encoder_out', default='label_encoder.pkl')
args = parser.parse_args()

# ── Load ───────────────────────────────────────────────────────────────────────
print(f"Loading {args.input} ...")
df = pd.read_csv(args.input)
print(f"  {len(df)} rows, {df['label'].value_counts().to_dict()}")

# ── Augmentation ──────────────────────────────────────────────────────────────
def augment(df: pd.DataFrame, n_passes: int = 3, noise_level: float = 0.004) -> pd.DataFrame:
    """
    Three augmentation passes:
      1. Gaussian noise (positional jitter)
      2. Horizontal flip (negate all x columns — simulates left/right hand variation)
      3. Gaussian noise on the flipped version
    """
    coord_cols = [c for c in df.columns if c != 'label']
    x_cols = [c for c in coord_cols if c.startswith('x')]

    copies = [df]

    # Noise passes
    for _ in range(n_passes):
        noisy = df.copy()
        noisy[coord_cols] += np.random.normal(0, noise_level, noisy[coord_cols].shape)
        copies.append(noisy)

    # Horizontal flip
    flipped = df.copy()
    flipped[x_cols] *= -1
    copies.append(flipped)

    # Noise on flipped
    noisy_flipped = flipped.copy()
    noisy_flipped[coord_cols] += np.random.normal(0, noise_level, noisy_flipped[coord_cols].shape)
    copies.append(noisy_flipped)

    result = pd.concat(copies, ignore_index=True)
    print(f"  Augmented: {len(df)} -> {len(result)} rows")
    return result

print("Augmenting...")
df = augment(df)

# ── Prepare features ───────────────────────────────────────────────────────────
X = df.drop('label', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['label'])

print(f"Classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ── Train ──────────────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    tree_method='hist',
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    verbosity=0,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print("\nTest set results:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(
    xgb.XGBClassifier(
        n_estimators=model.best_iteration or 200,
        max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective='multi:softprob', tree_method='hist', verbosity=0
    ),
    X, y, cv=cv, scoring='accuracy'
)
print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# ── Save ───────────────────────────────────────────────────────────────────────
with open(args.model_out, 'wb') as f:
    pickle.dump(model, f)
with open(args.encoder_out, 'wb') as f:
    pickle.dump(le, f)

print(f"\nSaved model -> {args.model_out}")
print(f"Saved encoder -> {args.encoder_out}")