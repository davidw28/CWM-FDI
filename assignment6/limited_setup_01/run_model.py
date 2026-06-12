## Hard code parameters for now
IN_PATH = "data/wine_train.csv"
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

## Evaluate model, not in separate script for now
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

### Plot results, not in separate script for now
# import plotly.express as px
# fig = px.scatter(pred, x = 'actual', y = 'predicted')
# fig.show()
