from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

labels_path =  "data/metadata/labels.csv"
output_path = "data/metadata/dataset_splits.csv"

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

def main():
    df = pd.read_csv(labels_path)

    df = df[df["verified"] == True].copy()

    train_df, temp_df = train_test_split(
        df,
        test_size=validation_ratio + test_ratio,
        random_state=42,
        stratify=df["label"]
    )

    # 4. Split temporary set into validation and test
    relative_test_ratio = test_ratio / (
        validation_ratio + test_ratio
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=42,
        stratify=temp_df["label"]
    )

    # 5. Record each image's split
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # 6. Combine and save
    result = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    result.to_csv(output_path, index=False)

    print(f"Saved dataset splits to: {output_path}")
    print(result.groupby(["split", "label"]).size())

if __name__ == "__main__":
    main()
