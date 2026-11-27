"""
H4 - Hybrid ConvNeXt-Small + ViT-Tiny (S-Adapter di kedua cabang)
=================================================================
Cabang lokal  : ConvNeXt-Small -> token 7x7 (49) -> S-Adapter token ConvNeXt
Cabang global : ViT-Tiny/16  -> 196 patch token, S-Adapter di setiap blok
Fusi          : Bidirectional Token Cross-Attention (3 head) -> Refinement -> Classifier

Strategi pelatihan: FREEZE (Parameter-Efficient Fine-Tuning)
  - Backbone ViT      : DIBEKUKAN seluruhnya, adaptasi lewat S-Adapter di setiap blok
  - Backbone ConvNeXt : stage 1-3 DIBEKUKAN, stage 4 DILATIH
  - Dilatih           : S-Adapter (kedua cabang), ConvNeXt stage 4, bidirectional
                        cross-attention, refinement block, classifier head

Arsitektur lengkap: models/hybrid_convnext_vit_s_adapter.py (HybridTokenSAdapterConvNeXt)
"""

from models.hybrid_convnext_vit_s_adapter import HybridTokenSAdapterConvNeXt

KODE = "H4"
NAMA = "Hybrid ConvNeXt-Small + ViT-Tiny (S-Adapter di kedua cabang)"
KELOMPOK = "Hybrid"
CONVNEXT = "convnext_small"
VIT = "vit_tiny_patch16_224"
NUM_HEADS = 3


def build(pretrained: bool = True) -> HybridTokenSAdapterConvNeXt:
    """Membangun model H4 dalam mode freeze."""
    return HybridTokenSAdapterConvNeXt(
        vit_name=VIT,
        convnext_name=CONVNEXT,
        adapter_bottleneck=64,
        num_heads=NUM_HEADS,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=True,
    )
