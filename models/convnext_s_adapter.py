"""
Baseline ConvNeXt + S-Adapter (satu cabang) - C1 (Tiny), C2 (Small), C3 (Base)
===============================================================================
S-Adapter mengikuti paper Cai dkk. (IEEE TIFS 2024); implementasi di models/s_adapter.py,
SAMA dengan yang dipakai cabang ConvNeXt pada model hybrid.

Alur (C4 = 768 untuk Tiny/Small, 1024 untuk Base):
  Input (B,3,224,224) -> Stem -> Stage 1-4 -> (B,C4,7,7)
  -> + S-Adapter (grid 7x7 = 49 token)       [residual]
  -> LayerNorm akhir ConvNeXt (pretrained)
  -> rata-rata 49 token -> (B,C4)
  -> Classifier: Linear(C4 -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
Freeze: stage 1-3 + LayerNorm akhir dibekukan; stage 4, S-Adapter, classifier dilatih.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from models.s_adapter import StatisticalAdapter


class ConvNeXtSAdapter(nn.Module):
    VARIANT_MAP = {
        "tiny": "convnext_tiny", "small": "convnext_small", "base": "convnext_base",
        "convnext_tiny": "convnext_tiny", "convnext_small": "convnext_small", "convnext_base": "convnext_base",
    }

    def __init__(
        self,
        variant: str = "small",
        adapter_bottleneck: int = 8,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.variant = variant.lower()
        if self.variant not in self.VARIANT_MAP:
            raise ValueError(f"Variant '{variant}' tidak dikenal. Pilih: 'tiny', 'small', 'base'.")
        self.model_name = self.VARIANT_MAP[self.variant]
        self.freeze_backbone = freeze_backbone

        base = create_model(self.model_name, pretrained=pretrained, num_classes=0)
        self.stem = base.stem
        self.stages = base.stages
        self.final_norm = base.head.norm             # LayerNorm akhir ConvNeXt (norm_pre di timm = Identity)
        self.num_features = base.num_features        # 768 atau 1024

        if freeze_backbone:
            for p in base.parameters():
                p.requires_grad = False
            for p in self.stages[-1].parameters():   # stage 4 dilatih
                p.requires_grad = True

        self.s_adapter = StatisticalAdapter(self.num_features, hidden_dim=adapter_bottleneck)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) -> peta fitur (B,C4,7,7) setelah S-Adapter dan LayerNorm akhir."""
        f = self.stages(self.stem(x))
        f = f + self.s_adapter.forward_map(f)
        return self.final_norm(f)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.forward_features(x).mean(dim=(2, 3))   # rata-rata 49 token
        logits = self.classifier(pooled)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1), "features": pooled}

    def print_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"=== ConvNeXt-{self.variant.upper()} + S-Adapter === total {total:,} | "
              f"dilatih {trainable:,} ({100 * trainable / total:.2f}%)")


def build_convnext_tiny_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="tiny", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


def build_convnext_small_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="small", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


def build_convnext_base_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="base", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)
