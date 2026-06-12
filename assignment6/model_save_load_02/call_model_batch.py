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

## Load model
import pickle
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

## Evaluate model
import numpy as np
import scipy.stats

def calc(model, data, feature_labels, target_label):    
    data['actual'] = data[target_label]
    data['predicted'] = model.predict(data[feature_labels])
    
    data['error'] = data['predicted'] - data['actual']

    rmse = np.sqrt(np.mean(data['error'] ** 2))
    print(f"RMSE={rmse:.4f}")

    r = scipy.stats.pearsonr(data['predicted'], data['actual']).statistic
    print(f"PearsonR={r:.3f}")
    
    return data[['predicted', 'actual', 'error']]

pred = calc(model, data, FEATURE_LABELS, TARGET_LABEL)
