"""
Single-Branch ConvNeXt + S-Adapter Architecture for Face Anti-Spoofing
======================================================================
Model Ablasi Khusus ConvNeXt (Local Feature) + S-Adapter (PEFT)
Tanpa Cabang Vision Transformer (ViT) dan Tanpa Cross-Attention Fusion.

Varian yang didukung:
- convnext_tiny  (Backbone: 28M params, Feature channels: 768)
- convnext_small (Backbone: 50M params, Feature channels: 768)
- convnext_base  (Backbone: 88M params, Feature channels: 1024)

Alur Arsitektur Single-Branch:
1. Input Image: (B, 3, 224, 224)
2. ConvNeXt Backbone: Stem (4x4, stride 4) -> Stages 1 s/d 4 -> (B, C, 7, 7)
3. ConvNeXt Spatial S-Adapter (TSR):
   - 1x1 Conv (Down-projection: C -> Bottleneck) + BatchNorm + GELU
   - 3x3 Depthwise Conv (Local texture & spoofing artifacts) + BatchNorm + GELU
   - 1x1 Conv (Up-projection: Bottleneck -> C) + BatchNorm
   - Learnable scaling factor (scale * adapter_out) + Residual Connection
4. Global Average Pooling (GAP): (B, C, 7, 7) -> (B, C)
5. Classification Head: Linear(C -> 256) -> GELU -> Dropout(0.3) -> Linear(256 -> 2)
6. Output: Logits & Probabilities (Live vs Spoof)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ─────────────────────────────────────────────────────────────
# 1. SPATIAL S-ADAPTER KHUSUS CONVNEXT (TOKEN SPATIAL REFINEMENT)
# ─────────────────────────────────────────────────────────────
class ConvNeXtSpatialSAdapter(nn.Module):
    """
    S-Adapter 2D yang diadaptasikan khusus untuk ConvNeXt Feature Maps:
    1. Down-projection (1x1 Conv) ke bottleneck dimensi rendah untuk efisiensi parameter.
    2. Depthwise Convolution 3x3 untuk menangkap anomali tekstur spoofing lokal
       (moire pattern layar, batas kertas cetak, distorsi spekular).
    3. Up-projection (1x1 Conv) mengembalikan ke dimensi kanal asal.
    4. Residual connection dengan parameter skala terpelajar (scale factor).
    """
    def __init__(
        self,
        channels: int,
        bottleneck_dim: int = 64,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.bottleneck_dim = bottleneck_dim

        self.adapter = nn.Sequential(
            # 1. Proyeksi Turun
            nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            # 2. Depthwise 3x3 Conv (Spasial Lokal)
            nn.Conv2d(
                bottleneck_dim,
                bottleneck_dim,
                kernel_size=3,
                padding=1,
                groups=bottleneck_dim,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            # 3. Proyeksi Naik
            nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        # Skala terpelajar untuk residual adapter
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, C, H, W) dari Stage 4 ConvNeXt
        Returns:
            Tensor (B, C, H, W) yang telah diperkaya adapter
        """
        residual = x
        adapted = self.adapter(x)
        out = residual + self.scale * adapted

        # LayerNorm pada dimensi kanal (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1)
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)
        return out


# ─────────────────────────────────────────────────────────────
# 2. MODEL UTAMA: CONVNEXT + S-ADAPTER (SINGLE-BRANCH)
# ─────────────────────────────────────────────────────────────
class ConvNeXtSAdapter(nn.Module):
    """
    Arsitektur Single-Branch ConvNeXt + S-Adapter untuk Face Anti-Spoofing.

    Mendukung 3 varian:
      - 'tiny'  : convnext_tiny  (768 features)
      - 'small' : convnext_small (768 features)
      - 'base'  : convnext_base  (1024 features)
    """
    VARIANT_MAP = {
        "tiny": "convnext_tiny",
        "small": "convnext_small",
        "base": "convnext_base",
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
    }

    def __init__(
        self,
        variant: str = "small",
        adapter_bottleneck: int = 64,
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
        self.adapter_bottleneck = adapter_bottleneck

        # 1. Muat Backbone ConvNeXt dari timm
        base_model = create_model(self.model_name, pretrained=pretrained, num_classes=0)
        self.stem = base_model.stem
        self.stages = base_model.stages
        self.norm_pre = base_model.norm_pre
        self.num_features = base_model.num_features  # 768 (tiny/small) atau 1024 (base)

        # 2. Pembekuan Parameter Backbone (PEFT)
        if freeze_backbone:
            for p in base_model.parameters():
                p.requires_grad = False
            # Opsi adaptif: Unfreeze stage terakhir (Stage 4) agar beradaptasi dengan domain FAS
            for p in self.stages[-1].parameters():
                p.requires_grad = True

        # 3. Pasang S-Adapter Spasial
        self.spatial_adapter = ConvNeXtSpatialSAdapter(
            channels=self.num_features,
            bottleneck_dim=adapter_bottleneck,
            init_scale=0.1,
        )

        # 4. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 5. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # Inisialisasi bobot classifier head
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass melalui Backbone ConvNeXt + S-Adapter.
        Input:  (B, 3, 224, 224)
        Output: Feature maps (B, C, 7, 7)
        """
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        x = self.spatial_adapter(x)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass lengkap.
        Returns dictionary yang kompatibel dengan format proyek:
          - 'logits': Tensor (B, 2)
          - 'probs': Tensor (B, 2)
          - 'features': Tensor (B, num_features)
        """
        feat_map = self.forward_features(x)       # (B, C, 7, 7)
        pooled = self.gap(feat_map).flatten(1)     # (B, C)
        logits = self.classifier(pooled)           # (B, 2)
        probs = F.softmax(logits, dim=-1)          # (B, 2)

        return {
            "logits": logits,
            "probs": probs,
            "features": pooled,
        }

    def print_parameter_summary(self):
        """Mencetak statistik efisiensi parameter (PEFT)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        pct = 100.0 * trainable / total if total > 0 else 0.0

        print(f"=== Parameter Summary: ConvNeXt-{self.variant.upper()} + S-Adapter ===")
        print(f"  Model Variant        : {self.model_name}")
        print(f"  Total Parameters     : {total:,}")
        print(f"  Trainable Parameters : {trainable:,} ({pct:.2f}%)")
        print(f"  Frozen Parameters    : {frozen:,} ({100 - pct:.2f}%)")
        print(f"  Backbone Freezing    : {'Enabled (PEFT)' if self.freeze_backbone else 'Disabled'}")
        print("===================================================================\n")


# ─────────────────────────────────────────────────────────────
# 3. HELPER FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def build_convnext_tiny_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="tiny", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)

def build_convnext_small_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="small", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)

def build_convnext_base_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ConvNeXtSAdapter(variant="base", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


if __name__ == "__main__":
    # Pengujian mandiri (Sanity Check)
    print("Menjalankan Sanity Check ConvNeXt + S-Adapter (Tiny, Small, Base)...\n")
    dummy_input = torch.randn(2, 3, 224, 224)

    for var in ["tiny", "small", "base"]:
        model = ConvNeXtSAdapter(variant=var, pretrained=False, freeze_backbone=True)
        model.print_parameter_summary()
        out = model(dummy_input)
        print(f"[{var.upper()}] Input: {dummy_input.shape} -> Logits: {out['logits'].shape}, Probs: {out['probs'].shape}\n")

