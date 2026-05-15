import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. Load and Map Labels
df = pd.read_csv('hand-gestures.csv')

label_map = {
    'close_palm': 'rock',
    'open_palm': 'paper',
    'peace': 'scissors',
    'thumbs_up': 'ignore'
}
df['label'] = df['label'].map(label_map)

# 2. Data Augmentation (Crucial for small datasets)
def add_noise(dataframe, noise_level=0.005):
    """Creates a 'shaken' version of the data to help the model generalize."""
    clean_data = dataframe.copy()
    coords = [c for c in clean_data.columns if c != 'label']
    # Add a tiny bit of random variation to each coordinate
    noise = np.random.normal(0, noise_level, clean_data[coords].shape)
    clean_data[coords] += noise
    return clean_data

# Double your dataset size by adding a noisy version of the rows
augmented_df = pd.concat([df, add_noise(df)], ignore_index=True)

# 3. Wrist-Centric Normalization
def normalize(row):
    # Shift everything relative to the wrist (landmark 0)
    for axis in ['x', 'y', 'z']:
        base = row[f'{axis}0']
        for i in range(21):
            row[f'{axis}{i}'] -= base
    return row

print("Normalizing and Augmenting...")
augmented_df = augmented_df.apply(normalize, axis=1)

# 4. Prepare for XGBoost
X = augmented_df.drop('label', axis=1)
y = augmented_df['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Train with Overfitting Protection
# 'max_depth=3' is better for small datasets like 120 rows per class
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3, 
    learning_rate=0.05,
    objective='multi:softprob',
    tree_method='hist' # Faster training
)

model.fit(X_train, y_train)

# 6. Save for your coworker
with open('rps_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print(f"Training Complete. Classes found: {le.classes_}")