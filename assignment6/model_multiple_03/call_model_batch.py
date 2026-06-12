import sys
import time
import pickle
import pandas as pd
import numpy as np
import scipy.stats

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

## Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

## Evaluate model
data['actual'] = data[TARGET_LABEL]

start = time.time()
data['predicted'] = model.predict(total_features)
elapsed = time.time() - start
print(f"Time={elapsed:.6f} s")
    
data['error'] = data['predicted'] - data['actual']

rmse = np.sqrt(np.mean(data['error'] ** 2))
print(f"RMSE={rmse:.4f}")

r = scipy.stats.pearsonr(data['predicted'], data['actual']).statistic
print(f"PearsonR={r:.3f}")