# Computer Vision for Spotted Lanternfly Egg Identification

**Independent Research Project | Yechan Lee | Princeton University**

## Overview

This project explores the use of computer vision and deep learning to identify spotted lanternfly (*Lycorma delicatula*) egg masses through image classification.

The goal is to develop and evaluate machine learning models capable of distinguishing spotted lanternfly egg masses from other insect eggs and visually similar natural materials, such as lichen and fungi.

By combining ecological observations with computer vision, this project investigates the potential for machine learning to support invasive species monitoring and identification.

**Project Status:** In Development

## Research Motivation

The spotted lanternfly is an invasive insect that has spread throughout parts of the United States, including New Jersey. Its egg masses are an important stage in its life cycle and a potential target for monitoring and management efforts.

However, identifying egg masses can be challenging because they may resemble other biological materials found on tree bark and other natural surfaces.

This project investigates whether computer vision models can learn visual characteristics that distinguish spotted lanternfly egg masses from other insect eggs and environmental lookalikes.

## Research Question

Can deep learning image classification models reliably distinguish spotted lanternfly egg masses from other insect eggs and visually similar natural materials?

### Research Objectives

- Develop a labeled image dataset containing spotted lanternfly egg masses, other insect eggs, and visual lookalikes.
- Implement multiple convolutional neural network architectures for image classification.
- Compare model performance using consistent training and evaluation procedures.
- Investigate classification errors and potential sources of model bias.
- Explore the potential application of computer vision to invasive species monitoring.

## Methodology

The project follows a supervised image classification approach.

### 1. Data Collection

Images are collected from publicly available biodiversity observation resources, including iNaturalist.

The dataset includes three classification categories:

| Class | Description |
|---|---|
| `slf_egg_mass` | Spotted lanternfly egg masses |
| `other_eggs` | Eggs from other insect species |
| `lookalike` | Visually similar natural materials, including lichen and fungi |

### 2. Data Labeling and Verification

Images are organized and labeled according to their respective categories.

The dataset metadata tracks:
- Observation ID
- Local image path
- Classification label
- Verification status
- Additional notes

Images are reviewed to improve labeling consistency and reduce incorrect classifications.

### 3. Model Development

The project uses transfer learning with pretrained convolutional neural networks.

The architectures being investigated include:

- ResNet18
- EfficientNet-B0
- MobileNetV3

Each architecture will be evaluated using a consistent dataset and experimental setup to allow meaningful comparisons.

### 4. Model Evaluation

Model performance is evaluated using:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Precision
- Recall
- F1-score
- Confusion matrices

Training and validation metrics are tracked across epochs to investigate model convergence, generalization, and potential overfitting.

## Dataset

The most recent dataset snapshot contains **195 labeled images**.

| Classification | Number of Images |
|---|---:|
| Spotted lanternfly egg masses | 60 |
| Other insect eggs | 75 |
| Visual lookalikes | 60 |
| **Total** | **195** |

*These counts represent an early dataset snapshot and will be updated as additional images are collected and verified.*

### Dataset Sources

- iNaturalist biodiversity observations
- Additional publicly available biological image resources, where applicable

### Dataset Considerations

The dataset is still relatively small and is being expanded. Particular attention is being given to:
- Increasing the number of verified spotted lanternfly egg mass images.
- Improving species representation within the other-egg category.
- Maintaining consistent labeling standards.
- Preventing data leakage between training, validation, and test sets.

## Project Structure

```text
spotted-lanternfly-egg-detection/
│
├── data/
│   ├── raw/
│   │   └── images/
│   └── metadata/
│
├── docs/
│
├── notebooks/
│
├── papers/
│
├── results/
│
├── src/
│   ├── data_collection/
│   └── models/
│
├── README.md
└── requirements.txt
```

## Model Comparison

The project investigates three model architectures:

| Model | Description | Status |
|---|---|---|
| ResNet18 | Residual convolutional neural network | Baseline |
| EfficientNet-B0 | Efficient convolutional architecture | Under evaluation |
| MobileNetV3 | Lightweight architecture designed for efficiency | Under evaluation |

The objective is to compare classification performance, training behavior, and computational requirements.

### Experimental Results

Results will be documented as experiments are completed.

Evaluation will include:

- Accuracy and loss curves across training epochs.
- Classification reports for each model.
- Confusion matrices.
- Comparison of precision, recall, and F1-score.

See the `results/` directory for experiment outputs.

## Installation and Usage

### Requirements

- Python 3.10 or later (update to the tested version)
- PyTorch
- Torchvision
- Additional dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone https://github.com/yechanlee8314/spotted-lanternfly-egg-detection.git

cd spotted-lanternfly-egg-detection
```

Create and activate a virtual environment:

```bash
python -m venv test_env
source test_env/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### Training

To train a model, run:

```bash
python -m src.models.train --model resnet18
```

Other supported architectures can be selected using the `--model` argument.

### Evaluation

To evaluate a trained model:

```bash
python -m src.models.evaluate --model resnet18
```

Refer to the source code and project documentation for configuration details and available options.

## Limitations

Several limitations are currently being investigated:

- Limited availability of verified spotted lanternfly egg mass images.
- Dataset size and class distribution may affect model generalization.
- Images collected from different sources may have inconsistent lighting, backgrounds, and image quality.
- Models may learn background characteristics rather than biological features.
- Validation performance does not necessarily reflect performance in real-world field conditions.

These limitations will be considered when interpreting experimental results.

## Future Work

Potential future research directions include:

- Expanding the dataset with additional verified insect egg images.
- Improving class balance and species diversity.
- Comparing additional deep learning architectures.
- Investigating data augmentation and hyperparameter optimization.
- Evaluating models on independently collected images.
- Exploring object detection and image segmentation for identifying egg masses within larger photographs.
- Investigating the potential for real-world field identification applications.

## References and Resources

Research papers and supporting resources are maintained in the `papers/` directory.

Additional references will be added as the project develops.

## Author

**Yechan Lee**  
Princeton University

GitHub: [yechanlee8314](https://github.com/yechanlee8314)

---

*This project is an ongoing independent research effort exploring the intersection of computer vision, machine learning, and invasive species monitoring.*