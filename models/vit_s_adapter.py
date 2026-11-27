"""
Baseline ViT + S-Adapter (satu cabang) - V4 (Tiny), V5 (Small), V6 (Base)
==========================================================================
S-Adapter mengikuti paper Cai dkk. (IEEE TIFS 2024); implementasi di models/s_adapter.py,
SAMA dengan yang dipakai cabang ViT pada model hybrid.

Alur (d = 192 / 384 / 768, 12 blok):
  Input -> normalisasi ulang ke mean/std 0.5 (standar pretrained ViT timm)
  -> Patch Embed 16x16 -> [CLS] + 196 token + pos embed -> (B,197,d)
  -> 12 blok:  x += MHSA(LN x); x += S-Adapter(x);  x += MLP(LN x); x += S-Adapter(x)
  -> LayerNorm akhir -> rata-rata 196 token patch (CLS tidak dipakai, sama seperti hybrid)
  -> Classifier: Linear(d -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
Freeze: seluruh backbone ViT dibekukan; 24 S-Adapter + classifier dilatih.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from models.s_adapter import StatisticalAdapter, ViTInputRenorm, vit_forward_dengan_s_adapter


class ViTSAdapter(nn.Module):
    VARIANT_MAP = {
        "tiny": "vit_tiny_patch16_224", "small": "vit_small_patch16_224", "base": "vit_base_patch16_224",
        "vit_tiny": "vit_tiny_patch16_224", "vit_small": "vit_small_patch16_224", "vit_base": "vit_base_patch16_224",
        "vit_tiny_patch16_224": "vit_tiny_patch16_224", "vit_small_patch16_224": "vit_small_patch16_224",
        "vit_base_patch16_224": "vit_base_patch16_224",
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

        self.input_renorm = ViTInputRenorm()
        self.vit = create_model(self.model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim
        n = len(self.vit.blocks)
        self.adapters_msa = nn.ModuleList([StatisticalAdapter(self.embed_dim, adapter_bottleneck) for _ in range(n)])
        self.adapters_mlp = nn.ModuleList([StatisticalAdapter(self.embed_dim, adapter_bottleneck) for _ in range(n)])

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) -> token (B,197,d) setelah 12 blok + S-Adapter + LayerNorm akhir."""
        return vit_forward_dengan_s_adapter(self.vit, self.adapters_msa, self.adapters_mlp, self.input_renorm(x))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.forward_features(x)[:, 1:].mean(dim=1)   # rata-rata 196 token patch
        logits = self.classifier(pooled)
        return {"logits": logits, "probs": F.softmax(logits, dim=-1), "features": pooled}

    def print_parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"=== ViT-{self.variant.upper()}/16 + S-Adapter === total {total:,} | "
              f"dilatih {trainable:,} ({100 * trainable / total:.2f}%)")


def build_vit_tiny_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="tiny", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


def build_vit_small_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="small", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


def build_vit_base_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="base", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)
