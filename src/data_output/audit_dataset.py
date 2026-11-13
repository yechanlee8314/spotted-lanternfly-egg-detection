from pathlib import Path
import pandas as pd

path = "data/metadata/labels.csv"
df = pd.read_csv(path)

print("Number of labeled images:", len(df))
print("\nImages per label:")
print(df["label"].value_counts(dropna=False))

print("\nColumns:")
print(df.columns.tolist())