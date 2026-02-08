import torch
import torch.nn as nn
from torchvision import models


class SwinTransformer(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False):
        super(SwinTransformer, self).__init__()

        # Load Swin Transformer model (tiny version for efficiency)
        self.swin = models.swin_t(weights="DEFAULT" if pretrained else None)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False

        # Get input features of classifier
        num_features = self.swin.head.in_features

        # Replace classifier head
        self.swin.head = nn.Linear(num_features, num_classes)

        # Ensure classifier remains trainable if backbone frozen
        if freeze_backbone:
            for param in self.swin.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.swin(x)


# ======================================================================
# Utility functions for model inspection
# ======================================================================
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
    """Calculate model size in MB (assuming float32 parameters)."""
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# ======================================================================
# Model initialization and summary
# ======================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("SWIN TRANSFORMER MODEL")
    print("=" * 80)

    model = SwinTransformer(num_classes=11, pretrained=True, freeze_backbone=False).to(device)

    print("Model Architecture:")
    print("=" * 80)
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")
