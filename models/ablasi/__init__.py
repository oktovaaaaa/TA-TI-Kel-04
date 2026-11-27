"""
15 Model Ablasi Penelitian (semua mode FREEZE)
==============================================

Hybrid (ConvNeXt + ViT, S-Adapter di kedua cabang):
              ViT-Tiny   ViT-Small   ViT-Base
  ConvNeXt-T     H1         H2          H3
  ConvNeXt-S     H4         H5          H6
  ConvNeXt-B     H7         H8          H9

Baseline satu cabang + S-Adapter:
  ConvNeXt : C1 (Tiny), C2 (Small), C3 (Base)
  ViT      : V4 (Tiny), V5 (Small), V6 (Base)

Setiap model punya file sendiri di folder ini dengan fungsi `build()`.
Pemakaian:
    from models.ablasi import buat_model
    model = buat_model("h5")
"""

from models.ablasi import h1_convnext_tiny_vit_tiny as _h1
from models.ablasi import h2_convnext_tiny_vit_small as _h2
from models.ablasi import h3_convnext_tiny_vit_base as _h3
from models.ablasi import h4_convnext_small_vit_tiny as _h4
from models.ablasi import h5_convnext_small_vit_small as _h5
from models.ablasi import h6_convnext_small_vit_base as _h6
from models.ablasi import h7_convnext_base_vit_tiny as _h7
from models.ablasi import h8_convnext_base_vit_small as _h8
from models.ablasi import h9_convnext_base_vit_base as _h9
from models.ablasi import c1_convnext_tiny as _c1
from models.ablasi import c2_convnext_small as _c2
from models.ablasi import c3_convnext_base as _c3
from models.ablasi import v4_vit_tiny as _v4
from models.ablasi import v5_vit_small as _v5
from models.ablasi import v6_vit_base as _v6

DAFTAR_MODEL = {
    "h1": _h1,
    "h2": _h2,
    "h3": _h3,
    "h4": _h4,
    "h5": _h5,
    "h6": _h6,
    "h7": _h7,
    "h8": _h8,
    "h9": _h9,
    "c1": _c1,
    "c2": _c2,
    "c3": _c3,
    "v4": _v4,
    "v5": _v5,
    "v6": _v6,
}

# Nama lama yang pernah dipakai di server -> kode resmi
ALIAS = {
    "c_tiny_v_tiny": "h1", "c_tiny_v_small": "h2", "c_tiny_v_base": "h3",
    "c_small_v_tiny": "h4", "c_small_v_small": "h5", "c_small_v_base": "h6",
    "c_base_v_tiny": "h7", "c_base_v_small": "h8", "c_base_v_base": "h9",
    "hybrid": "h5", "convnext": "c2",
}


def kode_resmi(nama: str) -> str:
    nama = nama.lower()
    return ALIAS.get(nama, nama)


def buat_model(nama: str, pretrained: bool = True):
    kode = kode_resmi(nama)
    if kode not in DAFTAR_MODEL:
        raise ValueError(f"Model '{nama}' tidak dikenal. Pilihan: {list(DAFTAR_MODEL)}")
    return DAFTAR_MODEL[kode].build(pretrained=pretrained)


def keterangan(nama: str) -> str:
    m = DAFTAR_MODEL[kode_resmi(nama)]
    return f"{m.KODE} - {m.NAMA}"
