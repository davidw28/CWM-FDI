import os
import pandas as pd
import collections
import subprocess
import os
import sys

# Read command line arguments
assert len(sys.argv) == 2
DATASET = sys.argv[1]

## Load results
model_names = sorted(x[:-4].lstrip(DATASET + "_") for x in os.listdir("results/") if DATASET + "_" in x and x[-4:] == ".txt")
results = collections.defaultdict(list)

for model_name in model_names:
    with open(f"results/{DATASET}_{model_name}.txt", "r") as f:
        result = eval(f.read())

        results["Model"].append(model_name)

        results["Training Energy (J)"].append(result["Training"]["Energy (J)"])
        results["RMSE (test)"].append(result["Calling (test)"]["RMSE"])

df = pd.DataFrame(results)

## Calculate CFP assuming CI of 80.846 gCO2/MJ based on 2023 IEA global estimate
## https://www.iea.org/world/emissions
NOMINAL_CI = 80.846e-6 # gCO2/J
df["Training CFP (gCO2)"] = df["Training Energy (J)"] * NOMINAL_CI

## Calculate Performance over Footprint
df["Performance/CFP"] = 1 / df["RMSE (test)"] / df["Training CFP (gCO2)"]

## Sort values by largest Performance/CFP, followed by smallest CFP
df = df.sort_values(by = ["Performance/CFP", "Training CFP (gCO2)"], ascending = [False, True])

print("====BEST MODELS====")
print(df.head(5))