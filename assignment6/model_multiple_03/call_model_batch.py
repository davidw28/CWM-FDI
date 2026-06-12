import scipy.stats
import numpy as np

import sys
import time
import pickle
import pandas as pd

## Load names
assert len(sys.argv) == 5
IN_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]
DATASET = sys.argv[3]
FEATURE_PATH = sys.argv[4]

## Load data
with open(FEATURE_PATH, "r") as f:
    FEATURE_INFO = eval(f.read())
    TARGET_LABEL = FEATURE_INFO["target"]
    FEATURE_LABELS = FEATURE_INFO["features"]

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