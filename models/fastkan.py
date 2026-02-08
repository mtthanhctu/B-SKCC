# Copyright 2024 Li, Ziyao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# -----------------------------
# Core FastKAN building blocks
# -----------------------------
class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: Optional[float] = None,
    ):
        super().__init__()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_grids = num_grids
        grid = torch.linspace(grid_min, grid_max, num_grids)
        # store grid as non-trainable parameter (so it moves with device)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., input_dim]
        # return: [..., input_dim, num_grids]
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layernorm = None
        if use_layernorm:
            assert input_dim > 1, "Do not use LayerNorm on 1D inputs. Set `use_layernorm=False`."
            self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, use_layernorm: bool = True) -> torch.Tensor:
        # x: [..., input_dim]
        if self.layernorm is not None and use_layernorm:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        # spline_basis: [..., input_dim, num_grids]
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret

    def plot_curve(
        self,
        input_index: int,
        output_index: int,
        num_pts: int = 1000,
        num_extrapolate_bins: int = 2
    ):
        ng = self.rbf.num_grids
        h = self.rbf.denominator
        assert input_index < self.input_dim
        assert output_index < self.output_dim
        w = self.spline_linear.weight[
            output_index, input_index * ng : (input_index + 1) * ng
        ]   # num_grids
        x = torch.linspace(
            self.rbf.grid_min - num_extrapolate_bins * h,
            self.rbf.grid_max + num_extrapolate_bins * h,
            num_pts
        )
        with torch.no_grad():
            y = (w * self.rbf(x.to(w.dtype))).sum(-1)
        return x, y


class FastKAN(nn.Module):
    """
    Stack of FastKANLayer, flexible hidden sizes defined by layers_hidden.
    layers_hidden: List like [in_dim, hid1, hid2, ..., out_dim]
    """
    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation = F.silu,
        spline_weight_init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        assert len(layers_hidden) >= 2, "layers_hidden must have at least [in, out]"
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_dim, out_dim,
                grid_min=grid_min,
                grid_max=grid_max,
                num_grids=num_grids,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionWithFastKANTransform(nn.Module):
    """
    Attention module where linear projections are replaced by FastKAN transforms.
    Useful as a drop-in alternative to linear-based attention.
    """
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        self.norm = head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q: [*, q_len, q_dim], k: [*, k_len, k_dim], v: [*, k_len, v_dim]
        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm  # [...,1,h,c]
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)  # [...,1,k,h,c]
        att = (wq * wk).sum(-1).softmax(-2)  # [..., q, k, h]
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]
        wv = self.linear_v(v).view(*v.shape[:-2], 1, v.shape[-2], self.num_heads, -1)  # [...,1,k,h,c]
        o = (att[..., None] * wv).sum(-3)  # [..., q, h, c]
        del att, wv
        o = o.view(*o.shape[:-2], -1)  # [..., q, h*c]
        if self.linear_g is not None:
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o
        o = self.linear_o(o)
        return o


# -----------------------------
# FastKAN classifier wrapper
# -----------------------------
class FastKANClassifier(nn.Module):
    """
    A convenience classifier: lightweight CNN backbone -> global pool -> FastKAN MLP head.
    - num_classes: output classes
    - kan_hidden: list of hidden dims for FastKAN (will prepend input dim automatically)
    - backbone_channels: final channel count of CNN backbone (default 256 matching small backbone)
    """
    def __init__(
        self,
        num_classes: int = 11,
        backbone_channels: int = 256,
        kan_hidden: Optional[List[int]] = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        if kan_hidden is None:
            # default: [backbone_channels, 512, 256, num_classes]
            kan_hidden = [backbone_channels, 512, 256, num_classes]
        else:
            # ensure output equals num_classes
            if kan_hidden[-1] != num_classes:
                kan_hidden = list(kan_hidden)
                kan_hidden[-1] = num_classes

        # Simple CNN backbone (lightweight) — replaceable by any other backbone externally
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),   # [B,32,224,224]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                     # [B,32,112,112]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B,64,112,112]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                     # [B,64,56,56]

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # [B,128,56,56]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                     # [B,128,28,28]

            nn.Conv2d(128, backbone_channels, kernel_size=3, stride=1, padding=1), # [B,backbone_channels,28,28]
            nn.BatchNorm2d(backbone_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))                            # [B, backbone_channels, 1, 1]
        )

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.flatten = nn.Flatten()
        # FastKAN expects input dim = backbone_channels
        self.kan = FastKAN(layers_hidden=kan_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)             # [B, C, 1, 1]
        x = self.flatten(x)              # [B, C]
        x = self.kan(x)                  # [B, num_classes]
        return x


# -----------------------------
# Utilities
# -----------------------------
def print_parameter_details(model: nn.Module) -> None:
    total_params = 0
    trainable_params = 0

    print("Layer-wise parameter count:")
    print("-" * 80)
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        total_params += params
        if parameter.requires_grad:
            trainable_params += params
            print(f"{name:<70} {params:,} (trainable)")
        else:
            print(f"{name:<70} {params:,} (frozen)")
    print("-" * 80)
    print(f"Total parameters       : {total_params:,}")
    print(f"Trainable parameters   : {trainable_params:,}")
    print(f"Frozen parameters      : {total_params - trainable_params:,}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_size(model: nn.Module) -> float:
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


# -----------------------------
# Test / Debug
# -----------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("=" * 80)
    print("FASTKAN (core) & FastKANClassifier test")
    print("=" * 80)

    # instantiate classifier
    model = FastKANClassifier(num_classes=11, backbone_channels=256, kan_hidden=None, freeze_backbone=False)
    model = model.to(device)

    print_parameter_details(model)
    print(f"Model size: {count_model_size(model):.2f} MB")
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # forward test
    x = torch.randn(2, 3, 224, 224).to(device)
    y = model(x)
    print(f"Output shape: {y.shape}")
