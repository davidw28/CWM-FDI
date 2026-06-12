import sys
import time
import pickle
import pandas as pd

## Load names
assert len(sys.argv) == 3
IN_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]

## Hard code parameters
TARGET_LABEL = "pH"
FEATURE_LABELS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'total sulfur dioxide', 'density', 'free sulfur dioxide', 'alcohol', "quality", 'sulphates']

## Load data
data = pd.read_csv(IN_PATH)
total_features = data[FEATURE_LABELS]
target = data[TARGET_LABEL]

## MODEL-SPECIFIC
from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])

## TRAIN MODEL
start = time.time()
model.fit(total_features, target)
elapsed = time.time() - start
print(f"Time={elapsed:.6f} s")

## Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)