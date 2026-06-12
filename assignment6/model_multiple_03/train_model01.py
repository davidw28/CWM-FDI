## Load names
import sys
assert len(sys.argv) == 3
IN_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]

## Hard code parameters
TARGET_LABEL = "pH"
FEATURE_LABELS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'total sulfur dioxide', 'density', 'free sulfur dioxide', 'alcohol', "quality", 'sulphates']

## Load data
import pandas as pd
data = pd.read_csv(IN_PATH)
total_features = data[FEATURE_LABELS]

## Train model
def train_model(Model, data, feature_labels, target_label):
    model = Model()
    model.fit(total_features, data[target_label])
    return model

from sklearn.linear_model import LinearRegression
import time
start = time.time()
model = train_model(LinearRegression, data, FEATURE_LABELS, TARGET_LABEL)
elapsed = time.time() - start
print(f"Time={elapsed:.6f} s")

## Save model
import pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)