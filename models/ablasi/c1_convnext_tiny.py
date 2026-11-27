"""
C1 - Baseline ConvNeXt-Tiny + S-Adapter (satu cabang)
=====================================================
Strategi pelatihan: FREEZE (Parameter-Efficient Fine-Tuning)
  - Backbone ConvNeXt : stage 1-3 DIBEKUKAN, stage 4 DILATIH
  - Dilatih           : S-Adapter spasial, ConvNeXt stage 4, classifier head

Arsitektur lengkap: models/convnext_s_adapter.py (ConvNeXtSAdapter)
"""

from models.convnext_s_adapter import ConvNeXtSAdapter

KODE = "C1"
NAMA = "Baseline ConvNeXt-Tiny + S-Adapter (satu cabang)"
KELOMPOK = "Baseline ConvNeXt"
VARIAN = "tiny"


def build(pretrained: bool = True) -> ConvNeXtSAdapter:
    """Membangun model C1 dalam mode freeze."""
    return ConvNeXtSAdapter(
        variant=VARIAN,
        adapter_bottleneck=64,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=True,
    )
