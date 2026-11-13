import pandas as pd

df = pd.read_csv("data/metadata/dataset_splits.csv")

# Check whether an observation appears in multiple splits
observation_splits = df.groupby("observation_id")["split"].nunique()

leaked_observations = observation_splits[observation_splits > 1]

print("Observations appearing in multiple splits:")
print(leaked_observations)

print("Number of potentially leaked observations:", len(leaked_observations))