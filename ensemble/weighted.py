import math
import os
from typing import List, Callable, Sequence

import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import yaml

from dataset.data_utils import DataModule


class WeightedEnsemble(nn.Module):
    """
    Combines per-model logits with class-dependent weights.
    Weights are normalized across models for each class via softmax.
    """

    def __init__(
        self,
        models: List[nn.Module],
        device: str,
        num_classes: int,
        confusion_matrices: List[torch.Tensor],
        top_k: int = 3,
        weight_floor: float = 1e-3,
    ):
        super().__init__()
        self.models = models
        self.device = device
        self.num_models = len(models)
        self.num_classes = num_classes
        safe_floor = max(weight_floor, 1e-6)

        init_logits = torch.full((self.num_models, num_classes), math.log(safe_floor), device=self.device)
        for idx, cm in enumerate(confusion_matrices):
            diag_scores = torch.diag(cm)
            if diag_scores.numel() == 0:
                continue
            top_count = min(top_k, diag_scores.numel())
            top_indices = torch.topk(diag_scores, k=top_count).indices
            init_logits[idx, top_indices] = 0.0  # bias top classes to start

        self.logits = nn.Parameter(init_logits)

    def forward(self, model_logits: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(model_logits, dim=0)  # [num_models, batch, num_classes]
        weights = torch.softmax(self.logits, dim=0)  # normalize across models for each class
        weighted = (weights.unsqueeze(1) * stacked).sum(dim=0)
        return weighted


class EnsembleTrainer:
    """
    Learns ensemble weights on top of fixed, trained models.
    Confusion matrix per model is used to seed higher weights for top-k classes.
    """

    def __init__(
        self,
        model_names: List[str],
        config_paths: List[str],
        model_builder: Callable[[str, int, bool, bool], nn.Module],
        top_k: int = 3,
        weight_floor: float = 1e-3,
        weight_lr: float = 1e-2,
        weight_epochs: int = 5,
        device: str = None,
    ):
        self.model_names = model_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Data module (reuse config of first model for loaders/classes)
        with open(config_paths[0], "r") as f:
            first_cfg = yaml.safe_load(f)

        self.data_module = DataModule(first_cfg["data"])
        self.train_loader, self.val_loader, self.test_loader = self.data_module.get_loaders()
        self.classes = self.data_module.get_classes()
        self.num_classes = len(self.classes)

        # Build & freeze models
        self.models: List[nn.Module] = []
        confusion_matrices: List[torch.Tensor] = []
        for name, cfg_path in zip(model_names, config_paths):
            with open(cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            model_cfg = cfg.get("model", {})
            model = model_builder(
                name,
                self.num_classes,
                model_cfg.get("pretrained", False),
                model_cfg.get("freeze_backbone", False),
            ).to(self.device)

            ckpt_path = os.path.join("experiments", name, f"{name}_best.pt")
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=self.device)
                model.load_state_dict(state_dict)
                print(f"🚀 Loaded checkpoint for ensemble: {ckpt_path}")
            else:
                print(f"⚠️ Warning: checkpoint not found for {name} at {ckpt_path}. Proceeding with current weights.")

            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            self.models.append(model)
            confusion_matrices.append(self._compute_confusion_matrix(model))

        self.ensemble = WeightedEnsemble(
            self.models,
            self.device,
            self.num_classes,
            confusion_matrices,
            top_k=top_k,
            weight_floor=weight_floor,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW([self.ensemble.logits], lr=weight_lr)
        self.weight_epochs = weight_epochs

    @torch.no_grad()
    def _compute_confusion_matrix(self, model: nn.Module) -> torch.Tensor:
        cm = torch.zeros((self.num_classes, self.num_classes), dtype=torch.float32, device=self.device)
        for images, labels in tqdm(self.val_loader, desc="Confusion matrix", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            cm.index_put_((labels, preds), torch.ones_like(labels, dtype=torch.float32), accumulate=True)
        return cm

    def train_weights(self):
        self.ensemble.train()
        for epoch in range(1, self.weight_epochs + 1):
            running_loss, correct, total = 0.0, 0, 0
            progress_bar = tqdm(self.val_loader, desc=f"Ensemble Epoch {epoch}", leave=False)
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                with torch.no_grad():
                    model_logits = [model(images) for model in self.models]

                ensemble_logits = self.ensemble(model_logits)
                loss = self.criterion(ensemble_logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * labels.size(0)
                preds = ensemble_logits.argmax(dim=1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = running_loss / total if total else 0.0
            acc = 100 * correct / total if total else 0.0
            print(f"🚀 Ensemble training epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.2f}%")

    @torch.no_grad()
    def evaluate(self, loader=None) -> float:
        eval_loader = loader or self.test_loader
        self.ensemble.eval()
        correct, total = 0, 0
        for images, labels in tqdm(eval_loader, desc="Ensemble Eval", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            model_logits = [model(images) for model in self.models]
            ensemble_logits = self.ensemble(model_logits)
            preds = ensemble_logits.argmax(dim=1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()
        acc = 100 * correct / total if total else 0.0
        print(f"🚀 Ensemble evaluation accuracy: {acc:.2f}%")
        return acc
