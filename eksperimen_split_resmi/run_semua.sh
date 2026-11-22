#!/bin/bash
# ============================================================
# Menjalankan eksperimen split resmi di server (H200)
# Jalankan dari ROOT repo:  bash eksperimen_split_resmi/run_semua.sh
# Bisa juga satu model saja: bash eksperimen_split_resmi/run_semua.sh hybrid fullft
# ============================================================
set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-./CelebA_Spoof}
WORKERS=${WORKERS:-8}

# 1. Buat split (sekali saja; dilewati bila split.csv sudah ada)
if [ ! -f eksperimen_split_resmi/split/split.csv ]; then
    $PYTHON eksperimen_split_resmi/buat_split.py --data_dir "$DATA_DIR" --cek_file
fi

# 2. Daftar eksperimen (model mode)
if [ $# -eq 2 ]; then
    DAFTAR=("$1 $2")
else
    DAFTAR=("hybrid freeze" "hybrid fullft" "convnext freeze" "convnext fullft")
fi

for cfg in "${DAFTAR[@]}"; do
    set -- $cfg
    echo "================================================================"
    echo ">>> $1 / $2   mulai: $(date)"
    echo "================================================================"
    $PYTHON eksperimen_split_resmi/train.py --model "$1" --mode "$2" \
        --data_dir "$DATA_DIR" --workers "$WORKERS" --resume
done
echo "SELESAI: $(date)"
