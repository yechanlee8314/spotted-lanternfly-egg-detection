
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# Convert text labels into numerical labels
label_mapping = {
    "slf_egg_mass": 0,
    "other_eggs": 1,
    "lookalike": 2
}


class EggDataset(Dataset):
    def __init__(self, csv_path, split, transform=None):
        self.data = pd.read_csv(csv_path)

        # Filter images by split
        self.data = self.data[
            self.data["split"] == split
        ].reset_index(drop=True)

        self.transform = transform

        # Project root directory
        self.project_root = Path(__file__).resolve().parents[2]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        # Get image path
        image_path = self.project_root / row["local_image_path"]
        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # Load image
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # Convert label to numerical value
        label = label_mapping[row["label"]]

        # Apply image transformations
        if self.transform:
            image = self.transform(image)

        return image, label
