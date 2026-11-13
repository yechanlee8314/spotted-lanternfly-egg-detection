
from pathlib import Path

import csv
import time
from datetime import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.dataset import EggDataset
from src.models.model import create_model
from src.models.transforms import train_transform, eval_transform

import argparse

# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "metadata" / "dataset_splits.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

LOG_PATH = PROJECT_ROOT / "results" / "experiment_log.csv"

BATCH_SIZE = 8
LEARNING_RATE = 0.001
EPOCHS = 15

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# -----------------------------
# Load datasets
# -----------------------------

def load_data():

    train_dataset = EggDataset(
        csv_path=CSV_PATH,
        split="train",
        transform=train_transform
    )

    val_dataset = EggDataset(
        csv_path=CSV_PATH,
        split="validation",
        transform=eval_transform
    )

    generator = torch.Generator()
    generator.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator = generator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        generator = generator
    )

    return train_loader, val_loader


# -----------------------------
# Training function
# -----------------------------

def train_one_epoch(model, loader, criterion, optimizer):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Track performance
        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


# -----------------------------
# Validation function
# -----------------------------

def validate(model, loader, criterion):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trains a model to differentiate spotted lanternfly eggs from similar looking substances."
    )
    parser.add_argument("--model", type = str, default="resnet18", choices= ["resnet18","efficientnet_b0","mobilenet_v3"], help = "The model used")

    args = parser.parse_args()
    return args

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def log_experiment(
    experiment_id,
    model_name,
    epochs,
    learning_rate,
    batch_size,
    train_size,
    val_size,
    best_val_accuracy,
    best_val_loss,
    training_time,
    checkpoint_path,
    seed
):

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    experiment = {
        "experiment_id": experiment_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_size": train_size,
        "val_size": val_size,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,
        "training_time_seconds": round(training_time, 2),
        "checkpoint": str(checkpoint_path),
        "seed":seed,
    }

    file_exists = LOG_PATH.exists()

    with open(LOG_PATH, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=experiment.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(experiment)

    print(f"Experiment logged to: {LOG_PATH}")

def log_training_history(model_name, experiment_id, history):

    history_path = (
        PROJECT_ROOT / "results" / "experiments" /
        model_name / f"{experiment_id}_history.csv"
    )

    history_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "model",
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ]

    with open(history_path, "w", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for epoch_data in history:

            writer.writerow({
                "experiment_id": experiment_id,
                "model": model_name,
                **epoch_data
            })

    print(f"Training history saved to: {history_path}")

# -----------------------------
# Main training process
# -----------------------------

def main():

    set_seed(42)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    args = parse_args()
    model_name = args.model

    #creates unique ID for each time model is trained
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = (CHECKPOINT_DIR / args.model / f"{experiment_id}_best.pth")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Selected model: {args.model}")
    print(f"Checkpoint path: {checkpoint_path}")

    print(f"Using device: {DEVICE}")

    train_loader, val_loader = load_data()

    print(f"Training images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")

    model = create_model(NUM_CLASSES, model_name=model_name).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )
    

    
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    start_time = time.time()

    history = []
    for epoch in range(EPOCHS):

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_accuracy = validate(
            model,
            val_loader,
            criterion
        )

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f}"
        )

        print(
            f"Validation Loss: {val_loss:.4f} | "
            f"Validation Accuracy: {val_accuracy:.4f}"
        )
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })
        log_training_history(
            model_name=model_name,
            experiment_id=experiment_id,
            history=history
        )

        # Save best model
        if (
            val_accuracy > best_val_accuracy
            or (
                val_accuracy == best_val_accuracy
                and val_loss < best_val_loss
            )
        ):
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss

            torch.save(
                model.state_dict(),
                checkpoint_path
            )

    print("Saved new best model!")
    training_time = time.time() - start_time
    log_experiment(
        experiment_id = experiment_id,
        model_name = model_name,
        epochs = EPOCHS,
        learning_rate = LEARNING_RATE,
        batch_size = BATCH_SIZE,
        train_size = len(train_loader.dataset),
        val_size = len(val_loader.dataset),
        best_val_accuracy = best_val_accuracy,
        best_val_loss = best_val_loss,
        training_time = training_time,
        checkpoint_path = checkpoint_path,
        seed=42)

    print(f"\nTraining complete: {model_name}")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()
