"""
classifier.py (XGBoost / Random Forest Version)
Trains a classifier over 2D scale/rotation invariant landmark profiles.
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--input', default='../landmarks/merged_landmarks.csv')
parser.add_argument('--model_out', default='./rps_model.pkl')
parser.add_argument('--encoder_out', default='./rps_label.pkl')
args = parser.parse_args()

print(f"Loading {args.input} ...")
df = pd.read_csv(args.input)

def augment(df: pd.DataFrame, n_passes: int = 2, noise_level: float = 0.005) -> pd.DataFrame:
    coord_cols = [c for c in df.columns if c != 'label']
    x_cols = [c for c in coord_cols if c.startswith('x')]
    y_cols = [c for c in coord_cols if c.startswith('y')]
    
    copies = [df]
    
    # 1. Jitter passes
    for _ in range(n_passes):
        noisy = df.copy()
        noisy[coord_cols] += np.random.normal(0, noise_level, noisy[coord_cols].shape)
        copies.append(noisy)
        
    # 2. Horizontal mirror flip
    flipped = df.copy()
    flipped[x_cols] *= -1
    copies.append(flipped)
    
    # 3. Small Rotation Augmentation (±10 degrees)
    # This helps the model handle cases where the middle finger isn't perfectly vertical
    for angle_deg in [-10, 10]:
        rotated = df.copy()
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # We need to rotate each x,y pair
        for i in range(1, 21):
            x_col, y_col = f'x{i}', f'y{i}'
            orig_x = df[x_col].values
            orig_y = df[y_col].values
            rotated[x_col] = orig_x * cos_a - orig_y * sin_a
            rotated[y_col] = orig_x * sin_a + orig_y * cos_a
        copies.append(rotated)
    
    result = pd.concat(copies, ignore_index=True)
    return result

print("Augmenting profiles...")
df = augment(df)

X = df.drop('label', axis=1).values
le = LabelEncoder()
y = le.fit_transform(df['label'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

print("\nTraining Model...")
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    tree_method='hist',
    eval_metric='mlogloss',
    early_stopping_rounds=20
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
print("\nTest Evaluation:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

with open(args.model_out, 'wb') as f:
    pickle.dump(model, f)
with open(args.encoder_out, 'wb') as f:
    pickle.dump(le, f)
print(f"Saved model configuration successfully.")