import torch
import torch.nn as nn
from torchvision import models

# ==============================================================
# EfficientNetV2 Model Definition
# ==============================================================

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False):
        """
        EfficientNetV2-S model wrapper.
        Args:
            num_classes (int): number of output classes.
            pretrained (bool): whether to use pretrained weights (ImageNet).
            freeze_backbone (bool): if True, freeze backbone feature extractor.
        """
        super(EfficientNetV2, self).__init__()

        # Load pretrained EfficientNetV2-S from torchvision
        self.backbone = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        )

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False

        # Get input dimension of the classifier
        in_features = self.backbone.classifier[1].in_features

        # Replace classifier with custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

        # Ensure classifier is always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


# ==============================================================
# Utility Functions
# ==============================================================

def print_parameter_details(model):
    total_params, trainable_params = 0, 0
    print("\nLayer-wise parameter count:")
    print("-" * 70)

    for name, param in model.named_parameters():
        num = param.numel()
        total_params += num
        if param.requires_grad:
            trainable_params += num
            print(f"{name:<60} {num:,} (trainable)")
        else:
            print(f"{name:<60} {num:,} (frozen)")

    print("-" * 70)
    print(f"Total parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Frozen parameters      : {total_params - trainable_params:,}\n")


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_size(model):
    """Estimate model size in MB"""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# ==============================================================
# Test model instantiation
# ==============================================================

if __name__ == "__main__":
    # Choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("=" * 80)
    print("EFFICIENTNETV2 MODEL INITIALIZATION")
    print("=" * 80)

    # Initialize model
    model = EfficientNetV2(num_classes=11, pretrained=False, freeze_backbone=False).to(device)

    # Print summary
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")