#!/bin/bash
# ============================================================
# Skrip Pelatihan ConvNeXt Small & Base
# (Tiny SUDAH SELESAI & SUKSES dengan Akurasi 99.50%, HTER 0.47%)
# ============================================================
set -e

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ta_env
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    conda activate ta_env
fi

echo "============================================================"
echo "MEMULAI PELATIHAN CONVNEXT: SMALL & BASE"
echo "Waktu mulai: $(date)"
echo "============================================================"

# 1. ConvNeXt-Small + S-Adapter
echo ""
echo "============================================================"
echo ">>> [1/2] Menjalankan: ConvNeXt-Small + S-Adapter ..."
echo "============================================================"
python train_convnext_s_adapter.py --variant small --epochs 25 --batch_size 64 --workers 8

# 2. ConvNeXt-Base + S-Adapter
echo ""
echo "============================================================"
echo ">>> [2/2] Menjalankan: ConvNeXt-Base + S-Adapter ..."
echo "============================================================"
python train_convnext_s_adapter.py --variant base --epochs 25 --batch_size 64 --workers 8

echo ""
echo "============================================================"
echo "SEMUA PELATIHAN CONVNEXT SELESAI DENGAN SUKSES!"
echo "Waktu selesai: $(date)"
echo "============================================================"

