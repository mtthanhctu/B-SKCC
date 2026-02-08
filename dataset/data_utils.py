import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataModule:

    def __init__(self, data_config: dict):
        self.train_dir = data_config.get("train_dir", "data/train")
        self.val_dir = data_config.get("val_dir", "data/val")
        self.test_dir = data_config.get("test_dir", "data/test")
        self.image_size = data_config.get("image_size", 224)
        self.num_workers = data_config.get("num_workers", 4)
        self.batch_size = data_config.get("batch_size", 32)
        self.augmentation = data_config.get("augmentation", {})

        self._build_transforms()
        self._prepare_datasets()
        self._prepare_dataloaders()

    # ----------------------------------------------------------
    # TRANSFORMS
    # ----------------------------------------------------------
    def _build_transforms(self):
        """Tạo transform cho train và val/test"""
        train_transforms = [transforms.Resize((self.image_size, self.image_size))]

        aug = self.augmentation
        if aug.get("random_crop", False):
            train_transforms.append(transforms.RandomResizedCrop(self.image_size))
        if aug.get("horizontal_flip", False):
            train_transforms.append(transforms.RandomHorizontalFlip())
        if aug.get("rotation", 0) > 0:
            train_transforms.append(transforms.RandomRotation(aug["rotation"]))
        if aug.get("color_jitter", False):
            train_transforms.append(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
            )

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transforms.extend([transforms.ToTensor(), normalize])
        self.train_transform = transforms.Compose(train_transforms)
        self.val_test_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            normalize
        ])

    # ----------------------------------------------------------
    # DATASETS
    # ----------------------------------------------------------
    def _prepare_datasets(self):
        """Tạo dataset ImageFolder"""
        print(f"[INFO] Loading datasets from {os.path.abspath('data')} ...")
        self.train_dataset = datasets.ImageFolder(self.train_dir, transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(self.val_dir, transform=self.val_test_transform)
        self.test_dataset = datasets.ImageFolder(self.test_dir, transform=self.val_test_transform)
        self.classes = self.train_dataset.classes

        print(f"[INFO] Dataset loaded:")
        print(f"  Train: {len(self.train_dataset)} images, {len(self.classes)} classes")
        print(f"  Val:   {len(self.val_dataset)} images")
        print(f"  Test:  {len(self.test_dataset)} images")

    # ----------------------------------------------------------
    # DATALOADERS
    # ----------------------------------------------------------
    def _prepare_dataloaders(self):
        """Tạo DataLoader cho train/val/test"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    # ----------------------------------------------------------
    # GETTERS
    # ----------------------------------------------------------
    def get_loaders(self):
        """Trả về train, val, test DataLoader"""
        return self.train_loader, self.val_loader, self.test_loader

    def get_classes(self):
        """Trả về danh sách class names"""
        return self.classes


if __name__ == "__main__":
    # Debug mode
    import yaml
    with open("configs/efficientnet.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_module = DataModule(config["data"])
    train_loader, val_loader, test_loader = data_module.get_loaders()
    print(f"Classes: {data_module.get_classes()}")
