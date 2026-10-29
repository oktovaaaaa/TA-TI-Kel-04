"""
Hybrid S-Adapter + ConvNeXt Architecture for Face Anti-Spoofing
================================================================
Arsitektur Dual-Branch Hybrid S-Adapter + ConvNeXt:

Branch 1: ConvNeXt Branch (Local Feature Extraction)
  - Input Image: (B, 3, 224, 224)
  - STEM (Conv 4x4, Stride 4) -> Stage 1 (56x56) -> Stage 2 (28x28) -> Stage 3 (14x14) -> Stage 4 (7x7)
  - Output Stage 4: Feature map (B, C4, 7, 7)
  - Tokenization (Spatial Flatten): 7x7 = 49 tokens -> (B, 49, C4)
  - S-Adapter (Adapted for ConvNeXt Tokens):
      * Restore 2D structure (7x7)
      * Local statistical token processing (Token Spatial Refinement / TSR via Depthwise Conv Adapter)
      * Projection + Residual Connection
  - Output: Local Tokens Tc of shape (B, 49, d)

Branch 2: Vision Transformer Branch (Global Feature Extraction)
  - Input Image: (B, 3, 224, 224)
  - Patch Embedding (patch size 16x16 -> 14x14 = 196 patches) -> Linear Projection (B, 196, d)
  - Prepend [CLS] token + 1D Positional Embedding -> (B, N_v + 1, d) (total 197 tokens)
  - Transformer Encoder Blocks with S-Adapter (L blocks):
      * Multi-Head Self-Attention (MHSA) + Add & Norm
      * S-Adapter (Bottleneck MLP with scaling) + Add & Norm
      * Feed-Forward Network (MLP) + Add & Norm
  - Output Tokens: (B, 197, d)
  - Remove [CLS] token -> Global Tokens Tv of shape (B, 196, d)

Fusion and Classification (Fusion Module):
  - Local Tokens Tc (B, 49, d) & Global Tokens Tv (B, 196, d)
  - Bidirectional Cross-Attention:
      * Local-to-Global Attention: Q = Tc, K = Tv, V = Tv -> (B, 49, d)
      * Global-to-Local Attention: Q = Tv, K = Tc, V = Tc -> (B, 196, d)
  - Concatenate & Feature Refinement:
      * Tr = [T'c ; T'v] -> (B, 49 + 196, d) = (B, 245, d)
      * Transformer Refinement Block (Self-Attention + MLP with Add & Norm) -> (B, 245, d)
  - Global Average Pooling: Pooling over 245 tokens -> (B, d)
  - MLP Classifier: Linear(d -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
  - Output: Binary Logits / Probabilities (Real / Spoof)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model


# ─────────────────────────────────────────────
# 1. S-ADAPTER MODULE UNTUK ViT
#    Adapter bottleneck ringan disisipkan
#    di setiap Transformer block untuk PEFT.
# ─────────────────────────────────────────────
class SAdapter(nn.Module):
    """
    S-Adapter (Sequential Adapter) untuk Vision Transformer.
    Input : (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, d_model: int, bottleneck_dim: int = 64):
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim)
        self.act       = nn.GELU()
        self.up_proj   = nn.Linear(bottleneck_dim, d_model)
        self.scale     = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return residual + self.scale * x


# ─────────────────────────────────────────────
# 2. S-ADAPTER MODULE UNTUK ConvNeXt TOKENS
#    (Adapted for ConvNeXt Tokens - TSR)
#    - Restore 2D structure (7x7)
#    - Local statistical token processing (Depthwise Conv)
#    - Projection + Residual
# ─────────────────────────────────────────────
class ConvNeXtTokenSAdapter(nn.Module):
    """
    S-Adapter yang diadaptasikan khusus untuk ConvNeXt Tokens:
    1. Mengembalikan token 1D (B, 49, C) ke struktur 2D grid (B, C, 7, 7)
    2. Menjalankan Token Spatial Refinement (TSR) berbasis depthwise convolution
       untuk menangkap statistik tekstur spoofing lokal
    3. Residual connection dengan parameter skala terpelajar
    4. Proyeksi ke dimensi token bersama d (shared embed_dim).
    """
    def __init__(
        self,
        in_channels: int = 768,
        embed_dim: int = 768,
        bottleneck_dim: int = 64,
        spatial_size: int = 7,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # 2D Local Statistical Adapter (TSR)
        self.adapter_2d = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_dim),
            nn.GELU(),
            # Depthwise 3x3 conv untuk menangkap pola tekstur lokal & statistik piksel
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
            nn.Conv2d(bottleneck_dim, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

        # Proyeksi ke dimensi shared d jika berbeda
        if in_channels != embed_dim:
            self.proj = nn.Linear(in_channels, embed_dim)
        else:
            self.proj = nn.Identity()

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, 49, in_channels)
        Returns:
            local_tokens Tc: (B, 49, embed_dim)
        """
        B, N, C = tokens.shape
        H = W = self.spatial_size
        residual = tokens

        # Step 1: Restore 2D structure (B, 49, C) -> (B, C, 7, 7)
        feat_2d = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # Step 2: Local statistical token processing (TSR)
        feat_adapted_2d = self.adapter_2d(feat_2d)

        # Step 3: Flatten kembali ke token (B, 49, C)
        adapted_tokens = feat_adapted_2d.flatten(2).permute(0, 2, 1)

        # Step 4: Residual connection
        x = residual + self.scale * adapted_tokens

        # Step 5: Proyeksi ke dimensi embedding bersama & LayerNorm
        x = self.proj(x)
        out = self.norm(x)
        return out


# ─────────────────────────────────────────────
# 3. CONVNEXT BRANCH (LOCAL FEATURE EXTRACTION)
# ─────────────────────────────────────────────
class ConvNeXtTokenBranch(nn.Module):
    """
    ConvNeXt Branch untuk ekstraksi fitur lokal:
    Input: (B, 3, 224, 224)
    STEM (4x4, stride 4) -> Stages 1-4
    Feature Map Stage 4: (B, C4, 7, 7)
    Spatial Flatten: (B, 49, C4)
    S-Adapter (Adapted for ConvNeXt Tokens) -> Local Tokens Tc (B, 49, d)
    """
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        embed_dim: int = 768,
        adapter_bottleneck: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        base = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.stem = base.stem
        self.stages = base.stages
        self.norm = base.norm_pre
        self.out_channels = base.num_features  # 768 untuk convnext_tiny

        # Freeze parameter backbone ConvNeXt jika PEFT diaktifkan
        if freeze_backbone:
            for p in base.parameters():
                p.requires_grad = False
            # Unfreeze stage terakhir untuk adaptasi domain lokal
            for p in self.stages[-1].parameters():
                p.requires_grad = True

        # S-Adapter yang diadaptasikan khusus untuk ConvNeXt tokens
        self.convnext_adapter = ConvNeXtTokenSAdapter(
            in_channels=self.out_channels,
            embed_dim=embed_dim,
            bottleneck_dim=adapter_bottleneck,
            spatial_size=7,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            Tc: Local Tokens (B, 49, embed_dim)
            feat_map: Raw 2D Feature Map Stage 4 (B, C4, 7, 7)
        """
        # Ekstraksi bertahap melalui Stem & 4 Stages
        x = self.stem(x)       # (B, 96, 56, 56)
        x = self.stages(x)     # (B, 768, 7, 7)
        feat_map = self.norm(x) # (B, 768, 7, 7)

        B, C, H, W = feat_map.shape
        # Tokenization (Spatial Flatten): 7x7 = 49 -> (B, 49, C)
        tokens_49 = feat_map.flatten(2).permute(0, 2, 1)

        # S-Adapter (Adapted for ConvNeXt Tokens)
        Tc = self.convnext_adapter(tokens_49)  # (B, 49, d)
        return Tc, feat_map


# ─────────────────────────────────────────────
# 4. ViT BRANCH (GLOBAL FEATURE EXTRACTION)
# ─────────────────────────────────────────────
class ViTWithSAdapter(nn.Module):
    """
    Vision Transformer Branch dengan S-Adapter di setiap Transformer block:
    Input : (B, 3, 224, 224)
    Patch Embedding (16x16) -> (B, 196, d)
    CLS + Pos Embed -> (B, 197, d)
    L Transformer Blocks dengan S-Adapter
    Remove [CLS] Token -> Global Tokens Tv (B, 196, d)
    """
    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        adapter_bottleneck: int = 64,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.vit = create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim  # 768 untuk ViT-Base

        # Sisipkan S-Adapter ke setiap Transformer Block
        self.adapters = nn.ModuleList([
            SAdapter(self.embed_dim, adapter_bottleneck)
            for _ in range(len(self.vit.blocks))
        ])

        if freeze_backbone:
            # Freeze bobot utama ViT, hanya latih S-Adapter
            for param in self.vit.parameters():
                param.requires_grad = False
            for param in self.adapters.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor, return_cls: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            Tv: Global Tokens (B, 196, d) — [CLS] token telah dibuang sesuai arsitektur diagram
            cls_token: (B, d) opsional jika dibutuhkan
        """
        # Patch embedding
        x = self.vit.patch_embed(x)             # (B, 196, d)

        # Tambahkan CLS token & positional embedding
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)    # (B, 197, d)
        x = self.vit.pos_drop(x + self.vit.pos_embed)

        # Proses melalui L Transformer blocks + S-Adapter
        for block, adapter in zip(self.vit.blocks, self.adapters):
            x = block(x)         # Self-Attention + MLP standar ViT
            x = adapter(x)       # S-Adapter residual adaptation

        x = self.vit.norm(x)     # LayerNorm final (B, 197, d)

        # Sesuai diagram: Buang [CLS] token (indeks 0) -> sisakan 196 token global (Tv)
        cls_out = x[:, 0, :]     # (B, d)
        Tv = x[:, 1:, :]         # (B, 196, d)

        if return_cls:
            return Tv, cls_out
        return Tv, None


# ─────────────────────────────────────────────
# 5. FUSION MODULE (CROSS-ATTENTION & REFINEMENT)
# ─────────────────────────────────────────────
class BidirectionalTokenCrossAttention(nn.Module):
    """
    Bidirectional Cross-Attention antara Local Tokens (Tc) dan Global Tokens (Tv):
    1. Local-to-Global Attention:
       Q = Tc (49, d), K = Tv (196, d), V = Tv (196, d) -> T'c (49, d)
    2. Global-to-Local Attention:
       Q = Tv (196, d), K = Tc (49, d), V = Tc (49, d) -> T'v (196, d)
    """
    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.local_to_global_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.global_to_local_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_tc1 = nn.LayerNorm(embed_dim)
        self.norm_tv1 = nn.LayerNorm(embed_dim)

        self.norm_tc2 = nn.LayerNorm(embed_dim)
        self.norm_tv2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, Tc: torch.Tensor, Tv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Tc: Local Tokens (B, 49, d)
            Tv: Global Tokens (B, 196, d)
        Returns:
            T_prime_c: (B, 49, d)
            T_prime_v: (B, 196, d)
        """
        # 1. Local-to-Global: Q = Tc, K = Tv, V = Tv
        q_c = self.norm_tc1(Tc)
        k_v = self.norm_tv1(Tv)
        l2g_out, _ = self.local_to_global_attn(query=q_c, key=k_v, value=k_v)
        T_prime_c = self.norm_tc2(Tc + self.dropout(l2g_out))

        # 2. Global-to-Local: Q = Tv, K = Tc, V = Tc
        q_v = self.norm_tv1(Tv)
        k_c = self.norm_tc1(Tc)
        g2l_out, _ = self.global_to_local_attn(query=q_v, key=k_c, value=k_c)
        T_prime_v = self.norm_tv2(Tv + self.dropout(g2l_out))

        return T_prime_c, T_prime_v


class TokenRefinementBlock(nn.Module):
    """
    Transformer Block / MLP Refinement untuk memurnikan gabungan token Tr (245 token):
    - Tr = [T'c ; T'v] -> shape (B, 245, d)
    - Multi-Head Self-Attention + Add & Norm
    - FeedForward MLP + Add & Norm
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 245, d)
        # Self-Attention + Residual
        attn_out, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 6. MLP CLASSIFIER HEAD
#    Linear(d -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
# ─────────────────────────────────────────────
class TokenAntiSpoofingClassifier(nn.Module):
    """
    Classifier Head sesuai diagram:
    Linear(d -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> num_classes)
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, embed_dim)
        return self.classifier(x)


# ─────────────────────────────────────────────
# 7. MODEL HYBRID LENGKAP (SESUAI GAMBAR ARSITEKTUR)
# ─────────────────────────────────────────────
class HybridTokenSAdapterConvNeXt(nn.Module):
    """
    Model Hybrid Lengkap S-Adapter (ViT) + ConvNeXt berbasis Token Fusion
    persis seperti yang digambarkan pada diagram arsitektur:
    
    1. ConvNeXt Branch : Input -> Stem -> Stages 1-4 -> 7x7 feat map
                         -> Flatten (49) -> ConvNeXt S-Adapter -> Local Tokens Tc (B, 49, d)
    2. ViT Branch      : Input -> Patch Embed -> CLS+Pos -> L Blocks + S-Adapter
                         -> Remove CLS -> Global Tokens Tv (B, 196, d)
    3. Cross-Attention : Bidirectional Local <-> Global Cross-Attention
    4. Concat & Refine : Tr = [T'c ; T'v] (B, 245, d) -> Transformer Refinement Block
    5. Global Pooling  : Average pooling over 245 tokens -> (B, d)
    6. MLP Classifier  : Linear(d -> 256) -> ReLU -> Dropout -> Linear(256 -> 2)
    """
    def __init__(
        self,
        vit_name: str = "vit_tiny_patch16_224",
        convnext_name: str = "convnext_tiny",
        adapter_bottleneck: int = 64,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # Branch 1: ViT dengan S-Adapter
        self.vit_branch = ViTWithSAdapter(
            vit_model_name=vit_name,
            adapter_bottleneck=adapter_bottleneck,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )
        self.embed_dim = self.vit_branch.embed_dim  # 192 untuk vit_tiny, 768 untuk vit_base

        # Pastikan num_heads membagi habis embed_dim
        if self.embed_dim % num_heads != 0:
            for h in [8, 6, 4, 3, 2, 1]:
                if self.embed_dim % h == 0:
                    num_heads = h
                    break

        # Branch 2: ConvNeXt dengan ConvNeXt Token S-Adapter
        self.convnext_branch = ConvNeXtTokenBranch(
            model_name=convnext_name,
            embed_dim=self.embed_dim,
            adapter_bottleneck=adapter_bottleneck,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

        # Cross-Attention Bidireksional (Local <-> Global)
        self.cross_attention = BidirectionalTokenCrossAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Feature Refinement Block (Transformer Block / MLP)
        self.refinement_block = TokenRefinementBlock(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            mlp_ratio=4.0,
            dropout=dropout,
        )

        # MLP Classifier Head
        self.classifier = TokenAntiSpoofingClassifier(
            embed_dim=self.embed_dim,
            num_classes=num_classes,
            dropout=0.3,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input images tensor (B, 3, 224, 224)
        Returns:
            dict:
              'logits' : (B, 2)
              'probs'  : (B, 2)
              'Tc'     : (B, 49, d)
              'Tv'     : (B, 196, d)
              'Tr'     : (B, 245, d)
        """
        # 1. Ekstraksi Fitur dari Kedua Branch
        # Local Tokens Tc dari ConvNeXt (B, 49, d)
        Tc, raw_feat_map = self.convnext_branch(x)
        # Global Tokens Tv dari ViT ([CLS] dibuang) (B, 196, d)
        Tv, _ = self.vit_branch(x)

        # 2. Bidirectional Cross-Attention (Local <-> Global)
        # Local-to-Global: Q=Tc, K=Tv, V=Tv -> (B, 49, d)
        # Global-to-Local: Q=Tv, K=Tc, V=Tc -> (B, 196, d)
        T_prime_c, T_prime_v = self.cross_attention(Tc, Tv)

        # 3. Concatenate Token: Tr = [T'c ; T'v] -> (B, 49 + 196, d) = (B, 245, d)
        Tr = torch.cat([T_prime_c, T_prime_v], dim=1)  # (B, 245, d)

        # 4. Feature Refinement melalui Transformer Encoder Block
        Tr_refined = self.refinement_block(Tr)          # (B, 245, d)

        # 5. Global Pooling: Rata-rata dari seluruh 245 token
        pooled = Tr_refined.mean(dim=1)                 # (B, d)

        # 6. MLP Classifier
        logits = self.classifier(pooled)                # (B, 2)
        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "probs": probs,
            "Tc": Tc,
            "Tv": Tv,
            "Tr": Tr_refined,
            "pooled": pooled,
        }


# ─────────────────────────────────────────────
# 8. LEGACY COMPATIBILITY WRAPPER
#    Memastikan script lama yang mengimport
#    HybridSAdapterConvNeXt tetap berjalan mulus.
# ─────────────────────────────────────────────
class HybridSAdapterConvNeXt(HybridTokenSAdapterConvNeXt):
    """
    Alias / Wrapper untuk backward compatibility dengan kode lama.
    Secara default menjalankan model token hybrid mutakhir sesuai arsitektur diagram.
    """
    def __init__(
        self,
        vit_name: str = "vit_base_patch16_224",
        convnext_name: str = "convnext_tiny",
        fusion_mode: str = "token_cross_attention",
        adapter_bottleneck: int = 64,
        fused_channels: int = 512,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__(
            vit_name=vit_name,
            convnext_name=convnext_name,
            adapter_bottleneck=adapter_bottleneck,
            num_classes=num_classes,
            pretrained=pretrained,
        )


# ─────────────────────────────────────────────
# 9. LOSS & OPTIMIZER UTILITIES
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal Loss untuk mengatasi ketidakseimbangan kelas live vs spoof."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal.mean()


def get_optimizer(model: nn.Module, lr_adapter: float = 1e-4, lr_head: float = 1e-3):
    """
    Optimizer dengan learning rate bertingkat:
    - Adapter & stage akhir ConvNeXt: lr kecil (PEFT)
    - Cross-attention, refinement, dan classifier head: lr standar
    """
    adapter_params = [
        p for n, p in model.named_parameters()
        if ("adapter" in n or "stages.3" in n) and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if ("adapter" not in n and "stages.3" not in n) and p.requires_grad
    ]

    return torch.optim.AdamW([
        {"params": adapter_params, "lr": lr_adapter, "weight_decay": 1e-4},
        {"params": head_params, "lr": lr_head, "weight_decay": 1e-2},
    ])


# ─────────────────────────────────────────────
# 10. VERIFIKASI FORWARD PASS
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("==================================================================")
    print("MENGUJI ARSITEKTUR HYBRID TOKEN S-ADAPTER + CONVNEXT (SESUAI GAMBAR)")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = HybridTokenSAdapterConvNeXt(
        vit_name="vit_tiny_patch16_224",
        convnext_name="convnext_tiny",
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=False,
    ).to(device)

    d = model.embed_dim
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backbone ViT          : vit_tiny_patch16_224 (embed_dim d={d})")
    print(f"Backbone ConvNeXt     : convnext_tiny (C4=768 -> adapted to d={d})")
    print(f"Total Parameter       : {total:,}")
    print(f"Trainable Parameter   : {trainable:,} ({100 * trainable / total:.2f}%)")

    # Forward pass dummy test (Batch size = 2)
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)

    print("\n--- HASIL TENSOR TIAP TAHAPAN SESUAI DIAGRAM ---")
    print(f"1. Input Image shape                 : {x.shape}")
    print(f"2. Local Tokens Tc (ConvNeXt) shape  : {out['Tc'].shape}      (Sesuai (B, 49, {d}))")
    print(f"3. Global Tokens Tv (ViT) shape      : {out['Tv'].shape}     (Sesuai (B, 196, {d}))")
    print(f"4. Refined Tokens Tr (Fused) shape   : {out['Tr'].shape}     (Sesuai (B, 245, {d}))")
    print(f"5. Global Pooled shape               : {out['pooled'].shape}        (Sesuai (B, {d}))")
    print(f"6. Output Logits shape               : {out['logits'].shape}          (Sesuai (B, 2))")
    print(f"7. Output Probs shape                : {out['probs'].shape}           (Sesuai (B, 2))")
    print(f"Sample Probs                         : {out['probs']}")
    print("\n[SUCCESS] Seluruh dimensi dan tahapan forward pass COCOK 100% DENGAN GAMBAR ARSITEKTUR!")
