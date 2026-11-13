"""
Single-Branch Vision Transformer (ViT) + S-Adapter Architecture for Face Anti-Spoofing
======================================================================================
Model Ablasi Khusus Vision Transformer (Global Feature) + S-Adapter (PEFT)
Tanpa Cabang ConvNeXt dan Tanpa Cross-Attention Fusion.

Varian yang didukung:
- vit_tiny_patch16_224  (Embedding Dim d: 192,  Heads: 3,  Blocks: 12, ~5.8M params)
- vit_small_patch16_224 (Embedding Dim d: 384,  Heads: 6,  Blocks: 12, ~22.3M params)
- vit_base_patch16_224  (Embedding Dim d: 768,  Heads: 12, Blocks: 12, ~87.0M params)

Alur Arsitektur Single-Branch:
1. Input Image: (B, 3, 224, 224)
2. Patch Embedding: 16x16 patch -> 14x14 = 196 patches -> Linear Projection ke d (B, 196, d)
3. Prepend [CLS] token + 1D Learnable Positional Embedding -> (B, 197, d)
4. 12 Transformer Blocks dengan S-Adapter di setiap blok:
   - Multi-Head Self-Attention (MHSA) + Add & Norm
   - S-Adapter: Down-proj (d -> 64) -> GELU -> Up-proj (64 -> d) + Learnable Scale + Residual
   - Feed-Forward Network (FFN/MLP) + Add & Norm
5. Final LayerNorm: (B, 197, d)
6. Global Token Pooling (GAP over 197 tokens): (B, 197, d) -> (B, d)
7. Classification Head: Linear(d -> 256) -> GELU -> Dropout(0.3) -> Linear(256 -> 2)
8. Output: Logits & Probabilities (Live vs Spoof)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ─────────────────────────────────────────────────────────────
# 1. S-ADAPTER MODULE UNTUK SETIAP TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────
class SAdapter(nn.Module):
    """
    Sequential Adapter (S-Adapter) untuk Vision Transformer:
    - Down-projection: d_model -> bottleneck_dim (64)
    - Non-linearitas: GELU
    - Up-projection: bottleneck_dim (64) -> d_model
    - Residual Connection dengan parameter skala terpelajar (scale factor)
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64, init_scale: float = 0.1):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.act       = nn.GELU()
        self.up_proj   = nn.Linear(bottleneck_dim, d_model)
        self.scale     = nn.Parameter(torch.ones(1) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor (B, N, d) dari output Transformer Block
        Returns:
            Tensor (B, N, d) yang diadaptasi secara residual
        """
        residual = x
        out = self.down_proj(x)
        out = self.act(out)
        out = self.up_proj(out)
        return residual + self.scale * out


# ─────────────────────────────────────────────────────────────
# 2. MODEL UTAMA: VIT + S-ADAPTER (SINGLE-BRANCH)
# ─────────────────────────────────────────────────────────────
class ViTSAdapter(nn.Module):
    """
    Arsitektur Single-Branch Vision Transformer (ViT) + S-Adapter untuk Face Anti-Spoofing.

    Mendukung 3 varian:
      - 'tiny'  : vit_tiny_patch16_224  (d = 192)
      - 'small' : vit_small_patch16_224 (d = 384)
      - 'base'  : vit_base_patch16_224  (d = 768)
    """
    VARIANT_MAP = {
        "tiny": "vit_tiny_patch16_224",
        "small": "vit_small_patch16_224",
        "base": "vit_base_patch16_224",
        "vit_tiny": "vit_tiny_patch16_224",
        "vit_small": "vit_small_patch16_224",
        "vit_base": "vit_base_patch16_224",
        "vit_tiny_patch16_224": "vit_tiny_patch16_224",
        "vit_small_patch16_224": "vit_small_patch16_224",
        "vit_base_patch16_224": "vit_base_patch16_224",
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

        # 1. Muat Backbone ViT dari timm
        self.vit = create_model(self.model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim  # 192 (tiny), 384 (small), atau 768 (base)
        self.num_blocks = len(self.vit.blocks)  # 12 blocks

        # 2. Sisipkan S-Adapter ke setiap Transformer Block
        self.adapters = nn.ModuleList([
            SAdapter(self.embed_dim, bottleneck_dim=adapter_bottleneck, init_scale=0.1)
            for _ in range(self.num_blocks)
        ])

        # 3. Pembekuan Parameter Backbone ViT (PEFT)
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False
            # S-Adapter harus selalu trainable
            for p in self.adapters.parameters():
                p.requires_grad = True

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
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
        Forward pass melalui Patch Embedding + 12 Transformer Blocks dengan S-Adapter.
        Input:  (B, 3, 224, 224)
        Output: Sequence of Tokens (B, 197, d)
        """
        B = x.shape[0]

        # 1. Patch Embedding: (B, 3, 224, 224) -> (B, 196, d)
        x = self.vit.patch_embed(x)

        # 2. Prepend CLS token + Positional Embedding
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)  # (B, 197, d)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # 3. Proses 12 Transformer Blocks + S-Adapter berurutan
        for block, adapter in zip(self.vit.blocks, self.adapters):
            x = block(x)       # MHSA + FFN standar
            x = adapter(x)     # S-Adapter residual adaptation

        # 4. Final LayerNorm
        x = self.vit.norm(x)  # (B, 197, d)
        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass lengkap.
        Returns dictionary yang kompatibel dengan format proyek:
          - 'logits': Tensor (B, 2)
          - 'probs': Tensor (B, 2)
          - 'features': Tensor (B, embed_dim)
        """
        tokens = self.forward_features(x)   # (B, 197, d)

        # Global Average Pooling atas seluruh token (CLS + 196 patch tokens)
        pooled = tokens.mean(dim=1)         # (B, d)

        logits = self.classifier(pooled)    # (B, 2)
        probs = F.softmax(logits, dim=-1)   # (B, 2)

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

        print(f"=== Parameter Summary: ViT-{self.variant.upper()}/16 + S-Adapter ===")
        print(f"  Model Variant        : {self.model_name}")
        print(f"  Embedding Dim (d)    : {self.embed_dim}")
        print(f"  Total Parameters     : {total:,}")
        print(f"  Trainable Parameters : {trainable:,} ({pct:.2f}%)")
        print(f"  Frozen Parameters    : {frozen:,} ({100 - pct:.2f}%)")
        print(f"  Backbone Freezing    : {'Enabled (PEFT)' if self.freeze_backbone else 'Disabled'}")
        print("===================================================================\n")


# ─────────────────────────────────────────────────────────────
# 3. HELPER FACTORY FUNCTIONS
# ─────────────────────────────────────────────────────────────
def build_vit_tiny_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="tiny", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)

def build_vit_small_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="small", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)

def build_vit_base_s_adapter(pretrained: bool = True, freeze_backbone: bool = True, **kwargs):
    return ViTSAdapter(variant="base", pretrained=pretrained, freeze_backbone=freeze_backbone, **kwargs)


if __name__ == "__main__":
    # Pengujian mandiri (Sanity Check)
    print("Menjalankan Sanity Check ViT + S-Adapter (Tiny, Small, Base)...\n")
    dummy_input = torch.randn(2, 3, 224, 224)

    for var in ["tiny", "small", "base"]:
        model = ViTSAdapter(variant=var, pretrained=False, freeze_backbone=True)
        model.print_parameter_summary()
        out = model(dummy_input)
        print(f"[{var.upper()}] Input: {dummy_input.shape} -> Logits: {out['logits'].shape}, Probs: {out['probs'].shape}\n")
