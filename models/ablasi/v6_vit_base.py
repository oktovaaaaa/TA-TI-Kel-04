"""
V6 - Baseline ViT-Base + S-Adapter (satu cabang)
================================================
Strategi pelatihan: FREEZE (Parameter-Efficient Fine-Tuning)
  - Backbone ViT : DIBEKUKAN seluruhnya
  - Dilatih      : S-Adapter di setiap blok Transformer, classifier head

Arsitektur lengkap: models/vit_s_adapter.py (ViTSAdapter)
"""

from models.vit_s_adapter import ViTSAdapter

KODE = "V6"
NAMA = "Baseline ViT-Base + S-Adapter (satu cabang)"
KELOMPOK = "Baseline ViT"
VARIAN = "base"


def build(pretrained: bool = True) -> ViTSAdapter:
    """Membangun model V6 dalam mode freeze."""
    return ViTSAdapter(
        variant=VARIAN,
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=True,
    )
