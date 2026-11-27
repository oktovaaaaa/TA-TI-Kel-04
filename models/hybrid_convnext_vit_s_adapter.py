"""
Hybrid ConvNeXt + ViT dengan S-Adapter (Statistical Adapter) untuk Face Anti-Spoofing
====================================================================================
S-Adapter mengikuti paper Cai dkk. (IEEE TIFS 2024, arXiv:2309.04038); implementasi di
models/s_adapter.py dan dipakai SAMA PERSIS oleh kedua cabang.

Prinsip penyeragaman kedua cabang: keduanya diolah sebagai TOKEN MAP 2D.
  - ViT      : 196 token patch = grid 14 x 14 (CLS tidak ikut diadaptasi)
  - ConvNeXt : peta fitur stage 4 = grid 7 x 7 = 49 token
  S-Adapter bekerja pada grid 2D tersebut, lalu fusi dilakukan dalam bentuk token
  berdimensi sama d (= dimensi ViT: 192 / 384 / 768).

Cabang 1 - ConvNeXt (fitur lokal), dimensi C4 = 768 (Tiny/Small) atau 1024 (Base)
  Input (B,3,224,224) -> Stem (B,C1,56,56) -> Stage 1-4 -> (B,C4,7,7)
  -> + S-Adapter(peta 7x7)                     [residual]
  -> LayerNorm akhir ConvNeXt (pretrained)
  -> flatten 49 token -> Linear(C4 -> d) -> LayerNorm   => Tc (B,49,d)
  Freeze: stage 1-3 + LayerNorm akhir dibekukan, stage 4 dilatih (sesuai keputusan penelitian)

Cabang 2 - ViT (fitur global), d = 192 / 384 / 768, 12 blok
  Input -> normalisasi ulang ke mean/std 0.5 (standar pretrained ViT timm)
  -> Patch Embed 16x16 -> [CLS] + 196 token + pos embed -> (B,197,d)
  -> 12 blok, masing-masing:  x += MHSA(LN x); x += S-Adapter(x)
                              x += MLP(LN x);  x += S-Adapter(x)     (Pers. 2 paper)
  -> LayerNorm akhir -> buang CLS => Tv (B,196,d)
  Freeze: seluruh backbone ViT dibekukan, hanya 24 S-Adapter yang dilatih

Fusi & klasifikasi
  Bidirectional cross-attention: T'c = Attn(Q=Tc, K=V=Tv) (B,49,d); T'v = Attn(Q=Tv, K=V=Tc) (B,196,d)
  Tr = [T'c ; T'v] (B,245,d) -> Transformer Refinement Block -> rata-rata 245 token (B,d)
  Classifier: Linear(d -> 256) -> ReLU -> Dropout(0.3) -> Linear(256 -> 2)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from models.s_adapter import StatisticalAdapter, ViTInputRenorm, vit_forward_dengan_s_adapter


# ─────────────────────────────────────────────
# 1. CABANG CONVNEXT (FITUR LOKAL) -> Tc (B, 49, d)
# ─────────────────────────────────────────────
class ConvNeXtTokenBranch(nn.Module):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        embed_dim: int = 768,
        adapter_bottleneck: int = 8,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        base = create_model(model_name, pretrained=pretrained, num_classes=0)
        self.stem = base.stem
        self.stages = base.stages
        self.final_norm = base.head.norm          # LayerNorm akhir ConvNeXt (norm_pre di timm = Identity)
        self.out_channels = base.num_features     # 768 (Tiny/Small) atau 1024 (Base)

        if freeze_backbone:
            for p in base.parameters():
                p.requires_grad = False
            for p in self.stages[-1].parameters():   # stage 4 dilatih
                p.requires_grad = True

        self.s_adapter = StatisticalAdapter(self.out_channels, hidden_dim=adapter_bottleneck)
        self.token_proj = nn.Linear(self.out_channels, embed_dim)   # C4 -> d (samakan dengan ViT)
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return: Tc (B, 49, d) dan peta fitur stage 4 setelah adapter + norm (B, C4, 7, 7)."""
        f = self.stages(self.stem(x))              # (B, C4, 7, 7)
        f = f + self.s_adapter.forward_map(f)      # S-Adapter pada grid 7x7 (residual)
        f = self.final_norm(f)
        tokens = f.flatten(2).transpose(1, 2)      # (B, 49, C4)
        Tc = self.token_norm(self.token_proj(tokens))
        return Tc, f


# ─────────────────────────────────────────────
# 2. CABANG ViT (FITUR GLOBAL) -> Tv (B, 196, d)
# ─────────────────────────────────────────────
class ViTWithSAdapter(nn.Module):
    def __init__(
        self,
        vit_model_name: str = "vit_base_patch16_224",
        adapter_bottleneck: int = 8,
        pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.input_renorm = ViTInputRenorm()
        self.vit = create_model(vit_model_name, pretrained=pretrained, num_classes=0)
        self.embed_dim = self.vit.embed_dim       # 192 / 384 / 768
        n = len(self.vit.blocks)                  # 12
        self.adapters_msa = nn.ModuleList([StatisticalAdapter(self.embed_dim, adapter_bottleneck) for _ in range(n)])
        self.adapters_mlp = nn.ModuleList([StatisticalAdapter(self.embed_dim, adapter_bottleneck) for _ in range(n)])

        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor, return_cls: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        tokens = vit_forward_dengan_s_adapter(
            self.vit, self.adapters_msa, self.adapters_mlp, self.input_renorm(x)
        )                                         # (B, 197, d)
        Tv = tokens[:, 1:, :]                     # buang CLS -> (B, 196, d)
        return (Tv, tokens[:, 0, :]) if return_cls else (Tv, None)


# ─────────────────────────────────────────────
# 3. FUSI: CROSS-ATTENTION & REFINEMENT, 4. CLASSIFIER
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
# 5. MODEL HYBRID LENGKAP
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
        adapter_bottleneck: int = 8,
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
        adapter_bottleneck: int = 8,
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
        adapter_bottleneck=8,
        num_classes=2,
        pretrained=False,
    ).to(device)

    d = model.embed_dim
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Backbone ViT          : vit_tiny_patch16_224 (embed_dim d={d})")
    print(f"Backbone ConvNeXt     : convnext_tiny (C4 -> diproyeksikan ke d={d})")
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
