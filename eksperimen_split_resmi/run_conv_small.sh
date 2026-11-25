#!/bin/bash
# ==============================================================================
# Script Training Otomatis Seri ConvNeXt-Small Hybrid (H4, H5, H6) - Freeze Mode
# S-Adapter & Classifier Head dilatih, Backbone Pretrained dibekukan (PEFT)
# ==============================================================================
# Catatan:
#   H5 (ConvNeXt-Small + ViT-Small) sudah pernah dijalankan sebagai `hybrid_freeze`.
#   Jika folder `eksperimen_split_resmi/hasil/hybrid_freeze/final_summary.json`
#   sudah ada, script default ini hanya melatih H4 dan H6 agar tidak kerja 2x.
#
# Cara menjalankan:
#   1. Jalankan yang belum selesai (H4 -> H6 berurutan):
#      bash eksperimen_split_resmi/run_conv_small.sh
#   2. Jalankan satu model tertentu saja:
#      bash eksperimen_split_resmi/run_conv_small.sh c_small_v_tiny freeze
#      bash eksperimen_split_resmi/run_conv_small.sh c_small_v_small freeze
#      bash eksperimen_split_resmi/run_conv_small.sh c_small_v_base freeze
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
    DAFTAR=("c_small_v_tiny freeze")
    if [ ! -f eksperimen_split_resmi/hasil/hybrid_freeze/final_summary.json ]; then
        DAFTAR+=("c_small_v_small freeze")
    else
        echo "[INFO] H5 (hybrid_freeze) sudah selesai, melewati c_small_v_small."
    fi
    DAFTAR+=("c_small_v_base freeze")
fi

echo "================================================================"
echo "Memulai Eksperimen ConvNeXt-Small Hybrid (Mode: FREEZE)"
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
echo "SEMUA EKSPERIMEN CONVNEXT-SMALL SELESAI: $(date)"
echo "================================================================"

