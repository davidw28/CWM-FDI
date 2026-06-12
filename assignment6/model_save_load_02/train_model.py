## Hard code parameters for now
## Hard code parameters for now
IN_PATH = "data/wine_train.csv"
MODEL_PATH = "models/model_1.pkl"

TARGET_LABEL = "pH"
FEATURE_LABELS = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'total sulfur dioxide', 'density', 'free sulfur dioxide', 'alcohol', "quality", 'sulphates']

## Load data, not in function for now
import pandas as pd
data = pd.read_csv(IN_PATH)
total_features = data[FEATURE_LABELS]

## Train model
## MODEL-SPECIFIC
def train_model(Model, data, feature_labels, target_label):
    model = Model()
    model.fit(total_features, data[target_label])
    return model

from sklearn.linear_model import LinearRegression
model = train_model(LinearRegression, data, FEATURE_LABELS, TARGET_LABEL)

## Save model
import pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)