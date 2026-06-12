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
target = data[TARGET_LABEL]

## MODEL-SPECIFIC
from sklearn.neural_network import MLPRegressor
model = MLPRegressor(solver = "adam")

## TRAIN MODEL
start = time.time()
model.fit(total_features, target)
elapsed = time.time() - start
print(f"Time={elapsed:.6f} s")

## Save model
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)