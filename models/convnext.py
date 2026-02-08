import torch
import torch.nn as nn
from torchvision import models

# ==============================================================
# 1️⃣ ConvNeXt Model Definition
# ==============================================================

class ConvNeXt(nn.Module):
    """
    ConvNeXt model (Base version).
    Supports pretrained weights and configurable number of output classes.
    """

    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False):
        super(ConvNeXt, self).__init__()

        # Load pretrained ConvNeXt-Base
        self.backbone = models.convnext_base(weights='DEFAULT' if pretrained else None)

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Replace classifier for custom number of classes
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, num_classes)

        # Ensure classifier is trainable even if backbone is frozen
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ==============================================================
# 2️⃣ Utility Functions
# ==============================================================

def print_parameter_details(model):
    """Print layer-wise parameter counts and trainability."""
    total_params = 0
    trainable_params = 0

    print("Layer-wise parameter count:")
    print("-" * 60)

    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params

        if parameter.requires_grad:
            trainable_params += params
            print(f"{name}: {params:,} (trainable)")
        else:
            print(f"{name}: {params:,} (frozen)")

    print("-" * 60)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")


def count_parameters(model):
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_size(model):
    """Calculate model size in MB."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# ==============================================================
# 3️⃣ Model Builder
# ==============================================================

def build_convnext(num_classes=11, pretrained=True, freeze_backbone=False):
    return ConvNeXt(num_classes, pretrained, freeze_backbone)


# ==============================================================
# 4️⃣ Test Run
# ==============================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("CONVNEXT MODEL")
    print("=" * 80)

    model = build_convnext(num_classes=11, pretrained=False).to(device)
    print("Model Architecture:")
    print("=" * 80)

    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print(f"\nOutput shape: {y.shape}")
