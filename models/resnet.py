import torch
import torch.nn as nn
from torchvision import models

# ==============================================================
# ResNet Model (ResNet50 by default)
# ==============================================================

class ResNet(nn.Module):
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=False, variant='resnet50'):
        super(ResNet, self).__init__()
        
        # Select ResNet variant
        if variant == 'resnet101':
            self.resnet = models.resnet101(weights='DEFAULT' if pretrained else None)
        else:
            self.resnet = models.resnet50(weights='DEFAULT' if pretrained else None)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Get the number of features for the classifier
        num_features = self.resnet.fc.in_features
        
        # Replace the final fully connected layer
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Ensure classifier is trainable
        if freeze_backbone:
            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)


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
    print("RESNET MODEL")
    print("=" * 80)

    model = ResNet(num_classes=11, pretrained=False, freeze_backbone=False, variant='resnet50').to(device)
    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Dummy forward test
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print(f"Output shape: {y.shape}")
