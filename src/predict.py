from pathlib import Path
import argparse
import sys

import torch
import torch.nn.functional as F
from PIL import Image

from src.models.model import create_model
from src.models.transforms import eval_transform
from src.models.dataset import label_mapping


# -----------------------------
# Configuration
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CLASS_NAMES = {
    value: key
    for key, value in label_mapping.items()
}

NUM_CLASSES = len(CLASS_NAMES)

def load_model(model_name, experiment_id):

    checkpoint_path = (
        CHECKPOINT_DIR /
        model_name /
        f"{experiment_id}_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )

    model = create_model(
        num_classes=NUM_CLASSES,
        model_name=model_name,
        pretrained=False
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )

    model = model.to(DEVICE)
    model.eval()

    return model

def main():
    parser = argparse.ArgumentParser(
        description="Takes in an image and categorizes it using trained models"
    )

    parser.add_argument("input_path", help="Path to the input image file")

    args = parser.parse_args()
    try:
        #load the mobilenet_v3 model 
        model = load_model("mobilenet_v3", "20261110_010300")
        # Open the image
        with Image.open(args.input_path) as img:
            print(f"Successfully loaded: {args.input_path} ({img.size} {img.mode})")
            img = img.convert("RGB")
            img = eval_transform(img)
            img = img.unsqueeze(0)
            img = img.to(DEVICE)
        with torch.no_grad():
            outputs = model(img)
            probabilities = F.softmax(outputs, dim=1)
            predicted_index = probabilities.argmax(dim=1).item()
            predicted_label = CLASS_NAMES[predicted_index]
            print(f"Predicted class: {predicted_label}")
            print(f"Probability: {probabilities[0][predicted_index].item():.2%}")
            print("\nClass probabilities:")
            for index, probability in enumerate(probabilities[0]):
                print(f"{CLASS_NAMES[index]}: {probability.item():.2%}")
    except FileNotFoundError:
        print(f"Error: The file '{args.input_path}' could not be found.", file=sys.stderr)
if __name__ == "__main__":
    main()