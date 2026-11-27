"""
H8 - Hybrid ConvNeXt-Base + ViT-Small (S-Adapter di kedua cabang)
=================================================================
Cabang lokal  : ConvNeXt-Base -> peta 7x7 (49 token) + S-Adapter -> proyeksi ke d
Cabang global : ViT-Small/16  -> 196 token patch, 2 S-Adapter per blok (setelah MHSA & MLP)
Fusi          : Bidirectional Token Cross-Attention (6 head) -> Refinement -> Classifier

Strategi pelatihan: FREEZE (Parameter-Efficient Fine-Tuning)
  - Backbone ViT      : DIBEKUKAN seluruhnya, adaptasi lewat 24 S-Adapter (2 per blok)
  - Backbone ConvNeXt : stage 1-3 DIBEKUKAN, stage 4 DILATIH
  - Dilatih           : S-Adapter (kedua cabang), ConvNeXt stage 4, bidirectional
                        cross-attention, refinement block, classifier head

Arsitektur lengkap: models/hybrid_convnext_vit_s_adapter.py (HybridTokenSAdapterConvNeXt)
"""

from models.hybrid_convnext_vit_s_adapter import HybridTokenSAdapterConvNeXt

KODE = "H8"
NAMA = "Hybrid ConvNeXt-Base + ViT-Small (S-Adapter di kedua cabang)"
KELOMPOK = "Hybrid"
CONVNEXT = "convnext_base"
VIT = "vit_small_patch16_224"
NUM_HEADS = 6


def build(pretrained: bool = True) -> HybridTokenSAdapterConvNeXt:
    """Membangun model H8 dalam mode freeze."""
    return HybridTokenSAdapterConvNeXt(
        vit_name=VIT,
        convnext_name=CONVNEXT,
        adapter_bottleneck=8,
        num_heads=NUM_HEADS,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=True,
    )
