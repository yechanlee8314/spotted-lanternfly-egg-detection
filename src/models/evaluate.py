
from pathlib import Path
import argparse

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report

from src.models.dataset import EggDataset
from src.models.model import create_model
from src.models.transforms import eval_transform


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_splits.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = [
    "slf_egg_mass",
    "other_eggs",
    "lookalike"
]


# -----------------------------
# Arguments
# -----------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained spotted lanternfly egg classification model."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0", "mobilenet_v3"],
        help="Model architecture to evaluate"
    )

    return parser.parse_args()


# -----------------------------
# Load validation dataset
# -----------------------------

def load_data():

    dataset = EggDataset(
        csv_path=CSV_PATH,
        split="validation",
        transform=eval_transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return loader


# -----------------------------
# Evaluate model
# -----------------------------

def evaluate(model_name):

    checkpoint_path = CHECKPOINT_DIR / f"{model_name}_best.pth"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Train the model before evaluating it."
        )

    print(f"Model: {model_name}")
    print(f"Using device: {DEVICE}")

    loader = load_data()

    model = create_model(
        num_classes=len(CLASS_NAMES),
        model_name=model_name
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    # -----------------------------
    # Classification report
    # -----------------------------

    report_dict = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    print("\nClassification Report:\n")
    print(report_df)

    # -----------------------------
    # Confusion matrix
    # -----------------------------

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2]
    )

    print("\nConfusion Matrix:")
    print(matrix)

    # -----------------------------
    # Save results
    # -----------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report_df.to_csv(
        RESULTS_DIR / f"{model_name}_classification_report.csv"
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    matrix_df.to_csv(
        RESULTS_DIR / f"{model_name}_confusion_matrix.csv"
    )

    print(f"\nResults saved for {model_name} in results/")


if __name__ == "__main__":

    args = parse_args()
    evaluate(args.model)
