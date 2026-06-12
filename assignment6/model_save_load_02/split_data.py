import pandas as pd
import sklearn.model_selection

IN_PATH = "../datasets/wine_quality.csv"
IN_SEP = ";"
IN_DECIMAL = ","

TEST_FRACTION = 0.2 # train/test split
NAME = "wine"
OUT_PATH = "data/"

df = pd.read_csv(IN_PATH, sep = IN_SEP, decimal = IN_DECIMAL)
df = df.dropna()
df = df.astype('float')

train, test = sklearn.model_selection.train_test_split(df, test_size = TEST_FRACTION)
train.to_csv(OUT_PATH + NAME + "_train.csv", index = False)
test.to_csv(OUT_PATH + NAME + "_test.csv", index = False)

### Code for other potential versions
# batch = df.sample(100)
