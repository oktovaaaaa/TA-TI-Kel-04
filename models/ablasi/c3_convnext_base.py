"""
C3 - Baseline ConvNeXt-Base + S-Adapter (satu cabang)
=====================================================
Strategi pelatihan: FREEZE (Parameter-Efficient Fine-Tuning)
  - Backbone ConvNeXt : stage 1-3 DIBEKUKAN, stage 4 DILATIH
  - Dilatih           : S-Adapter (grid 7x7), ConvNeXt stage 4, classifier head

Arsitektur lengkap: models/convnext_s_adapter.py (ConvNeXtSAdapter)
"""

from models.convnext_s_adapter import ConvNeXtSAdapter

KODE = "C3"
NAMA = "Baseline ConvNeXt-Base + S-Adapter (satu cabang)"
KELOMPOK = "Baseline ConvNeXt"
VARIAN = "base"


def build(pretrained: bool = True) -> ConvNeXtSAdapter:
    """Membangun model C3 dalam mode freeze."""
    return ConvNeXtSAdapter(
        variant=VARIAN,
        adapter_bottleneck=8,
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=True,
    )
