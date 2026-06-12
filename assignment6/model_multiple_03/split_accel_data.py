## Load names
import sys
assert len(sys.argv) == 5
TRAIN_PATH = sys.argv[1]
TEST_PATH = sys.argv[2]
SAMPLE_PATH = sys.argv[3]
FEATURE_PATH = sys.argv[4]

## Hardcode
IN_PATH = "../datasets/accelerometer.csv"
IN_SEP = ","
IN_DECIMAL = "."
TEST_FRACTION = 0.2 # train/test split

## Load data
import pandas as pd
df = pd.read_csv(IN_PATH, sep = IN_SEP, decimal = IN_DECIMAL)
df = df.dropna()
df = df.astype('float')

## Split data
import sklearn.model_selection
train, test = sklearn.model_selection.train_test_split(df, test_size = TEST_FRACTION)
sample = df.sample(n = 1000)

## Save files
train.to_csv(TRAIN_PATH, index = False)
test.to_csv(TEST_PATH, index = False)
sample.to_csv(SAMPLE_PATH, index = False)

## Save targets/features
FEATURE_INFO = {
    "target": "pctid",
    "features": ["wconfid", "x", "y", "z"],
    "instances": len(df)
}
with open(FEATURE_PATH, "w") as f:
    print(FEATURE_INFO, file = f)
