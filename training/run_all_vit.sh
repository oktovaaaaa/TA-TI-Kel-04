#!/bin/bash
# ============================================================
# Skrip Antrian Otomatis Pelatihan 3 Varian ViT + S-Adapter
# Varian: Tiny -> Small -> Base
# ============================================================
set -e

# Pindah ke direktori skrip (agar aman jika dijalankan dari mana saja)
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
echo "MEMULAI ANTRIAN PELATIHAN VIT + S-ADAPTER (3 VARIAN)"
echo "Direktori Kerja: $(pwd)"
echo "Python Exec    : $PYTHON_EXEC"
echo "Waktu mulai    : $(date)"
echo "============================================================"

# 1. ViT-Tiny/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [1/3] Menjalankan: ViT-Tiny/16 + S-Adapter ..."
echo "============================================================"
$PYTHON_EXEC train_vit_s_adapter.py --variant tiny --epochs 25 --batch_size 64 --workers 8

# 2. ViT-Small/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [2/3] Menjalankan: ViT-Small/16 + S-Adapter ..."
echo "============================================================"
$PYTHON_EXEC train_vit_s_adapter.py --variant small --epochs 25 --batch_size 64 --workers 8

# 3. ViT-Base/16 + S-Adapter
echo ""
echo "============================================================"
echo ">>> [3/3] Menjalankan: ViT-Base/16 + S-Adapter ..."
echo "============================================================"
$PYTHON_EXEC train_vit_s_adapter.py --variant base --epochs 25 --batch_size 64 --workers 8

echo ""
echo "============================================================"
echo "SEMUA PELATIHAN VIT SELESAI DENGAN SUKSES!"
echo "Waktu selesai: $(date)"
echo "============================================================"
