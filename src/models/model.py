
import torch.nn as nn
from torchvision import models

def create_model(num_classes=3, pretrained=True, model_name="resnet18"):

    if model_name == "resnet18":
            # Load ResNet-18
        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

    

    elif model_name == "efficientnet_b0":
        # Load EfficientNet-B0
        weights = ( 
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )
        model = models.efficientnet_b0(weights=weights)


    elif model_name == "mobilenet_v3":
        # Load MobileNetV3
        weights = ( 
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )
        model = models.mobilenet_v3_large(weights=weights)
        

    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    if model_name == "resnet18":
        # Replace ResNet18 classifier
        in_features = model.fc.in_features

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "efficientnet_b0":
        # Replace EfficientNet classifier
        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "mobilenet_v3":
        # Replace MobileNet classifier
        in_features = model.classifier[3].in_features

        model.classifier[3] = nn.Linear(
            in_features,
            num_classes
        )

    trainable_params = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    return model