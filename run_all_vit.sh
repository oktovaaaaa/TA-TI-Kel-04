#!/bin/bash
# ============================================================
# Skrip Antrian Otomatis Pelatihan 3 Varian ViT + S-Adapter
# Varian: Tiny -> Small -> Base
# ============================================================
set -e

echo "============================================================"
echo "MEMULAI ANTRIAN PELATIHAN VIT + S-ADAPTER (3 VARIAN)"
echo "Waktu mulai: $(date)"
echo "============================================================"

# Pastikan conda environment ta_env aktif
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ta_env
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate ta_env
fi

# 1. ViT-Tiny/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [1/3] Menjalankan: ViT-Tiny/16 + S-Adapter ..."
echo "============================================================"
python train_vit_s_adapter.py --variant tiny --epochs 25 --batch_size 64 --workers 8

# 2. ViT-Small/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [2/3] Menjalankan: ViT-Small/16 + S-Adapter ..."
echo "============================================================"
python train_vit_s_adapter.py --variant small --epochs 25 --batch_size 64 --workers 8

# 3. ViT-Base/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [3/3] Menjalankan: ViT-Base/16 + S-Adapter ..."
echo "============================================================"
python train_vit_s_adapter.py --variant base --epochs 25 --batch_size 64 --workers 8

echo ""
echo "============================================================"
echo "SEMUA PELATIHAN VIT SELESAI DENGAN SUKSES!"
echo "Waktu selesai: $(date)"
echo "============================================================"

