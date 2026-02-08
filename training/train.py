import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.data_utils import DataModule
from models import convnext, efficientnet, resnet, swin_transformer, vit, fastkan
from ensemble import EnsembleTrainer


# ==========================================================
# MODEL REGISTRY
# ==========================================================
MODEL_REGISTRY = {
    "convnext": ("models.convnext", "configs/convnext.yaml"),
    "efficientnet": ("models.efficientnet", "configs/efficientnet.yaml"),
    "resnet": ("models.resnet", "configs/resnet.yaml"),
    "swin": ("models.swin_transformer", "configs/swin.yaml"),
    "vit": ("models.vit", "configs/vit.yaml"),
    "fastkan": ("models.fastkan", "configs/fastkan.yaml"),
}

# Global used when spawning worker processes for bagging training
GLOBAL_EARLY_STOP = None


# ==========================================================
# MODEL BUILDER
# ==========================================================
def build_model(model_name, num_classes, pretrained=True, freeze_backbone=False):
    if model_name == "convnext":
        return convnext.ConvNeXt(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == "efficientnet":
        return efficientnet.EfficientNetV2(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == "resnet":
        return resnet.ResNet(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == "swin":
        return swin_transformer.SwinTransformer(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == "vit":
        return vit.VisionTransformer(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif model_name == "fastkan":
        return fastkan.FastKANClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


# Module-level worker for multiprocessing (must be picklable)
def _train_single(model_name):
    """Train a single model by name. Reads `GLOBAL_EARLY_STOP` for early stopping."""
    _, cfg_path = MODEL_REGISTRY[model_name]
    trainer = ModelTrainer(model_name, cfg_path, early_stop_patience=GLOBAL_EARLY_STOP)
    trainer.train()


# ==========================================================
# TRAINING CLASS
# ==========================================================
class ModelTrainer:
    def __init__(self, model_name, config_path, early_stop_patience=None):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # Load dataset
        self.data_module = DataModule(self.cfg["data"])
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_loaders()
        self.classes = self.data_module.get_classes()
        self.num_classes = len(self.classes)

        # Build model
        model_cfg = self.cfg["model"]
        self.model = build_model(
            model_name,
            num_classes=self.num_classes,
            pretrained=model_cfg.get("pretrained", True),
            freeze_backbone=model_cfg.get("freeze_backbone", False)
        ).to(self.device)

        # Training setup
        train_cfg = self.cfg["training"]
        self.epochs = train_cfg["epochs"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-4),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler()

        # Early stopping setup
        self.early_stop_patience = early_stop_patience
        self.best_val_loss = float("inf")
        self.early_stop_counter = 0

        # Logging setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("experiments", model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, "tensorboard", timestamp))
        self.log_path = os.path.join(self.output_dir, f"train_log_{timestamp}.txt")

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # ----------------------------------------------------------
    # TRAIN LOOP
    # ----------------------------------------------------------
    def train(self):
        best_acc = 0.0
        self._log(f"🚀 Starting training on {self.model_name.upper()} for {self.epochs} epochs.")

        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self._train_one_epoch(epoch)
            val_loss, val_acc = self._evaluate(self.val_loader, epoch, split="Validation")
            test_loss, test_acc = self._evaluate(self.test_loader, epoch, split="Test")
            self.scheduler.step()

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # TensorBoard logging
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Loss/Test", test_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            self.writer.add_scalar("Accuracy/Test", test_acc, epoch)

            # Save best checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = os.path.join(self.output_dir, f"{self.model_name}_best.pt")
                torch.save(self.model.state_dict(), ckpt_path)
                self._log(f"💾 Saved best checkpoint: {ckpt_path}")

            self._log(
                f"[Epoch {epoch}/{self.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%"
            )

            # Early stopping
            if self.early_stop_patience is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
                    self._log(f"⚠️ No improvement for {self.early_stop_counter}/{self.early_stop_patience} epochs.")

                if self.early_stop_counter >= self.early_stop_patience:
                    self._log("🛑 Early stopping triggered.")
                    break

        self._plot_metrics()
        self._log(f"✅ Training completed. Best validation accuracy: {best_acc:.2f}%")
        self.writer.close()

    # ----------------------------------------------------------
    # TRAIN ONE EPOCH
    # ----------------------------------------------------------
    def _train_one_epoch(self, epoch):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / total
        acc = 100 * correct / total
        return avg_loss, acc

    # ----------------------------------------------------------
    # EVAL (Validation/Test)
    # ----------------------------------------------------------
    @torch.no_grad()
    def _evaluate(self, loader, epoch: int, split: str):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0

        progress_bar = tqdm(loader, desc=f"Epoch {epoch} [{split}]", leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            progress_bar.set_postfix(loss=loss.item())

        avg_loss = running_loss / total if total else 0.0
        acc = 100 * correct / total if total else 0.0
        return avg_loss, acc

    # ----------------------------------------------------------
    # LOGGING & PLOTTING
    # ----------------------------------------------------------
    def _log(self, message):
        print(message)
        with open(self.log_path, "a") as f:
            f.write(message + "\n")

    def _plot_metrics(self):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Accuracy")
        plt.plot(self.history["val_acc"], label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"{self.model_name}_metrics.png"))
        plt.close()


# ==========================================================
# MAIN FUNCTION
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Train model(s) with optional early stopping and TensorBoard logging.")
    parser.add_argument("--model", type=str, help="Model name to train (e.g., resnet, vit, fastkan)")
    parser.add_argument("--all", action="store_true", help="Train all models in registry")
    parser.add_argument("--early_stop", type=int, nargs="?", const=10, help="Enable Early Stopping (default patience = 10)")
    parser.add_argument("--bagging", type=int, default=0, help="Number of models in bagging ensemble (train in parallel if possible)")
    parser.add_argument("--models", nargs="+", help="List of models for bagging ensemble")
    parser.add_argument("--ensemble_topk", type=int, default=3, help="Top-k classes to prioritize per model when initializing ensemble weights")
    parser.add_argument("--ensemble_lr", type=float, default=1e-2, help="Learning rate for ensemble weight learner")
    parser.add_argument("--ensemble_epochs", type=int, default=5, help="Number of epochs to learn ensemble weights")
    parser.add_argument("--ensemble_floor", type=float, default=1e-3, help="Floor weight for non-priority classes in ensemble init")
    args = parser.parse_args()

    if not args.model and not args.all and args.bagging == 0:
        print("⚠️ Usage examples:")
        print("   python train.py --model resnet")
        print("   python train.py --all")
        print("   python train.py --model vit --early_stop 15")
        print("   python train.py --bagging 3 --models resnet vit efficientnet")
        return

    if args.all:
        for model_name, (_, cfg_path) in MODEL_REGISTRY.items():
            print(f"\n🚀 Training {model_name.upper()} ...")
            trainer = ModelTrainer(model_name, cfg_path, early_stop_patience=args.early_stop)
            trainer.train()

    elif args.bagging > 0:
        if not args.models or len(args.models) != args.bagging:
            print("⚠️ Please provide --models matching the --bagging count.")
            return

        invalid = [m for m in args.models if m not in MODEL_REGISTRY]
        if invalid:
            print(f"❌ Invalid model(s) for ensemble: {', '.join(invalid)}")
            print("   Available models:", ", ".join(MODEL_REGISTRY.keys()))
            return

        print(f"\n🧩 Training bagging ensemble ({args.bagging} models): {args.models}")
        gpu_count = torch.cuda.device_count()
        print(f"➡️ Detected GPUs: {gpu_count}. Models to train in parallel: {len(args.models)}")

        # Train each constituent model in parallel (process-per-model) if checkpoints are missing.
        # If checkpoints already exist, train step will still run but is idempotent for the best weights.
        from torch.multiprocessing import Pool, set_start_method

        try:
            set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # already set

        # use module-level `_train_single` which reads `GLOBAL_EARLY_STOP`

        with Pool(processes=min(len(args.models), max(1, gpu_count or os.cpu_count() or 1))) as pool:
            pool.map(_train_single, args.models)

        config_paths = [MODEL_REGISTRY[name][1] for name in args.models]
        ensemble_trainer = EnsembleTrainer(
            args.models,
            config_paths,
            model_builder=build_model,
            top_k=args.ensemble_topk,
            weight_floor=args.ensemble_floor,
            weight_lr=args.ensemble_lr,
            weight_epochs=args.ensemble_epochs,
        )
        ensemble_trainer.train_weights()
        ensemble_trainer.evaluate()

    elif args.model:
        if args.model not in MODEL_REGISTRY:
            print(f"❌ Invalid model '{args.model}'. Available models:")
            print("   ", ", ".join(MODEL_REGISTRY.keys()))
            return
        _, cfg_path = MODEL_REGISTRY[args.model]
        trainer = ModelTrainer(args.model, cfg_path, early_stop_patience=args.early_stop)
        trainer.train()


if __name__ == "__main__":
    main()
