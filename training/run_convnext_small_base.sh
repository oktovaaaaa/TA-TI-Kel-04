#!/bin/bash
# ============================================================
# Skrip Pelatihan ConvNeXt Small & Base
# (Tiny SUDAH SELESAI & SUKSES dengan Akurasi 99.50%, HTER 0.47%)
# ============================================================
set -e

# Pindah ke direktori skrip (agar aman jika dijalankan dari home directory)
cd "$(dirname "$0")"

# Cari Python environment ta_env
if [ -f "/home/glory/miniconda3/envs/ta_env/bin/python" ]; then
    PYTHON_EXEC="/home/glory/miniconda3/envs/ta_env/bin/python"
elif [ -f "$HOME/miniconda3/envs/ta_env/bin/python" ]; then
    PYTHON_EXEC="$HOME/miniconda3/envs/ta_env/bin/python"
else
    PYTHON_EXEC="python"
fi

echo "============================================================"
echo "MEMULAI PELATIHAN CONVNEXT: SMALL & BASE"
echo "Direktori Kerja: $(pwd)"
echo "Python Exec    : $PYTHON_EXEC"
echo "Waktu mulai    : $(date)"
echo "============================================================"

# 1. ConvNeXt-Small + S-Adapter
echo ""
echo "============================================================"
echo ">>> [1/2] Menjalankan: ConvNeXt-Small + S-Adapter ..."
echo "============================================================"
$PYTHON_EXEC train_convnext_s_adapter.py --variant small --epochs 25 --batch_size 64 --workers 8

# 2. ConvNeXt-Base + S-Adapter
echo ""
echo "============================================================"
echo ">>> [2/2] Menjalankan: ConvNeXt-Base + S-Adapter ..."
echo "============================================================"
$PYTHON_EXEC train_convnext_s_adapter.py --variant base --epochs 25 --batch_size 64 --workers 8

echo ""
echo "============================================================"
echo "SEMUA PELATIHAN CONVNEXT SELESAI DENGAN SUKSES!"
echo "Waktu selesai: $(date)"
echo "============================================================"
