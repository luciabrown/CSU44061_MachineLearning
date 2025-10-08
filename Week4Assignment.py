# In your assignments and projects, unless otherwise stated it
# is mandatory to present cross-validation analysis to support
# your choice of hyperparameter values.
# reasonable baseline too

# This assignment includes: 
# feature selection, model selection, model training and evaluation. Not all datasets
# are useful, e.g. sometimes the data measured fails to capture the important relationships
# or is just too noisy. You now have the tools to analyse the data to uncover such problems.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Preprocessing - splitting the datasets apart
with open("week4.php.txt", "r") as f:
    lines = f.readlines()
datasets = []
current = []
for line in lines:
    line = line.strip()
    if line.startswith("#"):
        if current:
            datasets.append(pd.DataFrame([x.split(",") for x in current]))
            current = []
        current.append(line)
    else:
        current.append(line)
if current:
    datasets.append(pd.DataFrame([x.split(",") for x in current]))
for i, df in enumerate(datasets):
    print(f"\nDataset {i+1}:")
    print(df)
    df.to_csv(f"dataset_{i+1}.csv", index=False, header=False)