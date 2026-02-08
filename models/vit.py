import torch
import torch.nn as nn
from torchvision import models

# ==============================================================
# Vision Transformer (ViT) Model
# ==============================================================

class VisionTransformer(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False):
        super(VisionTransformer, self).__init__()
        
        # Load pretrained ViT model (base variant)
        self.vit = models.vit_b_16(weights='DEFAULT' if pretrained else None)

        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # Get number of features from classifier head
        num_features = self.vit.heads.head.in_features

        # Replace classifier with custom head
        self.vit.heads.head = nn.Linear(num_features, num_classes)

        # Ensure classifier is trainable even if backbone is frozen
        if freeze_backbone:
            for param in self.vit.heads.head.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.vit(x)


# ==============================================================
# Utility Functions
# ==============================================================

def print_parameter_details(model):
    """Print detailed parameter count and trainability."""
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
    """Estimate model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# ==============================================================
# Run test (debug mode)
# ==============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("VISION TRANSFORMER (ViT) MODEL")
    print("=" * 80)

    model = VisionTransformer(num_classes=11, pretrained=False).to(device)
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Dummy forward test
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print(f"Output shape: {y.shape}")
