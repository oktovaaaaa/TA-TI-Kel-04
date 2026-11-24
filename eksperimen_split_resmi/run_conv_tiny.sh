#!/bin/bash
# ==============================================================================
# Script Training Otomatis Seri ConvNeXt-Tiny Hybrid (H1, H2, H3) - Freeze Mode
# S-Adapter & Classifier Head dilatih, Backbone Pretrained dibekukan (PEFT)
# ==============================================================================
# Cara menjalankan:
#   1. Jalankan semua (H1 -> H2 -> H3 berurutan):
#      bash eksperimen_split_resmi/run_conv_tiny.sh
#   2. Jalankan satu model tertentu saja:
#      bash eksperimen_split_resmi/run_conv_tiny.sh c_tiny_v_tiny freeze
#      bash eksperimen_split_resmi/run_conv_tiny.sh c_tiny_v_small freeze
#      bash eksperimen_split_resmi/run_conv_tiny.sh c_tiny_v_base freeze
# ==============================================================================
set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-./CelebA_Spoof}
# Penting: WORKERS=2 agar hemat RAM fisik (server 8GB RAM) dan aman dari OOM Killer
WORKERS=${WORKERS:-2}

# Pastikan split.csv sudah ada
if [ ! -f eksperimen_split_resmi/split/split.csv ]; then
    echo "Membuat split.csv resmi terlebih dahulu..."
    $PYTHON eksperimen_split_resmi/buat_split.py --data_dir "$DATA_DIR" --cek_file
fi

if [ $# -ge 1 ]; then
    MODEL=$1
    MODE=${2:-freeze}
    DAFTAR=("$MODEL $MODE")
else
    # Default: latih ketiga varian ConvNeXt-Tiny dalam mode freeze
    DAFTAR=(
        "c_tiny_v_tiny freeze"
        "c_tiny_v_small freeze"
        "c_tiny_v_base freeze"
    )
fi

echo "================================================================"
echo "Memulai Eksperimen ConvNeXt-Tiny Hybrid (Mode: FREEZE)"
echo "Jumlah model dalam antrean: ${#DAFTAR[@]}"
echo "Workers: $WORKERS"
echo "Tanggal mulai: $(date)"
echo "================================================================"

for cfg in "${DAFTAR[@]}"; do
    set -- $cfg
    M_NAME=$1
    M_MODE=$2
    LOG_FILE="training_${M_NAME}_${M_MODE}.log"

    echo ""
    echo "================================================================"
    echo ">>> Memulai: $M_NAME | Mode: $M_MODE | Waktu: $(date)"
    echo ">>> Log disimpan di: $LOG_FILE"
    echo "================================================================"

    $PYTHON eksperimen_split_resmi/train.py \
        --model "$M_NAME" \
        --mode "$M_MODE" \
        --data_dir "$DATA_DIR" \
        --workers "$WORKERS" \
        --resume 2>&1 | tee -a "$LOG_FILE"

    echo ">>> Selesai: $M_NAME ($M_MODE) pada $(date)"
done

echo ""
echo "================================================================"
echo "SEMUA EKSPERIMEN CONVNEXT-TINY SELESAI: $(date)"
echo "================================================================"
