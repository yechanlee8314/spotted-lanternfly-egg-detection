from pathlib import Path
import csv
import argparse
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = PROJECT_ROOT / "results" / "experiments"
GRAPH_DIR = PROJECT_ROOT / "results" / "graphs"


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate training graphs from experiment history."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0", "mobilenet_v3"],
        help="Model whose training history should be plotted."
    )

    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Specific experiment ID to plot. Defaults to the latest history file."
    )

    return parser.parse_args()


# -----------------------------
# Load experiment history
# -----------------------------

def load_history(model_name, experiment_id):

    model_dir = EXPERIMENT_DIR / model_name

    if experiment_id is None:

        history_files = sorted(model_dir.glob("*_history.csv"))

        if not history_files:
            raise FileNotFoundError(
                f"No experiment history found for {model_name}."
            )

        history_path = history_files[-1]

    else:

        history_path = (
            model_dir / f"{experiment_id}_history.csv"
        )

    if not history_path.exists():
        raise FileNotFoundError(
            f"Could not find experiment history: {history_path}"
        )

    with open(history_path, "r", newline="") as file:

        reader = csv.DictReader(file)
        history = list(reader)

    if not history:
        raise ValueError("The experiment history file is empty.")

    return history, history_path


# -----------------------------
# Generate graphs
# -----------------------------

def plot_metric(history, model_name, experiment_id, metric, title, ylabel, filename):

    epochs = [
        int(row["epoch"])
        for row in history
    ]

    train_values = [
        float(row[f"train_{metric}"])
        for row in history
    ]

    val_values = [
        float(row[f"val_{metric}"])
        for row in history
    ]

    plt.figure(figsize=(9, 5))

    plt.plot(
        epochs,
        train_values,
        marker="o",
        label=f"Training {metric}"
    )

    plt.plot(
        epochs,
        val_values,
        marker="o",
        label=f"Validation {metric}"
    )

    plt.title(f"{title} - {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)

    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = GRAPH_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{experiment_id}_{filename}.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Graph saved to: {output_path}")


# -----------------------------
# Main process
# -----------------------------

def main():

    args = parse_args()

    history, history_path = load_history(
        model_name=args.model,
        experiment_id=args.experiment_id
    )

    experiment_id = history[0]["experiment_id"]

    print(f"Model: {args.model}")
    print(f"Experiment: {experiment_id}")
    print(f"History loaded from: {history_path}")

    # Generate loss graph
    plot_metric(
        history,
        args.model,
        experiment_id,
        metric="loss",
        title="Training and Validation Loss",
        ylabel="Loss",
        filename="loss"
    )

    # Generate accuracy graph
    plot_metric(
        history,
        args.model,
        experiment_id,
        metric="accuracy",
        title="Training and Validation Accuracy",
        ylabel="Accuracy",
        filename="accuracy"
    )


if __name__ == "__main__":
    main()