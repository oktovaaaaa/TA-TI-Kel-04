#!/bin/bash
# ==============================================================================
# Training ulang 15 model ablasi dengan split resmi (SEMUA MODE FREEZE)
#   Hybrid  : H1..H9  (ConvNeXt Tiny/Small/Base x ViT Tiny/Small/Base)
#   Baseline: C1..C3  (ConvNeXt Tiny/Small/Base) dan V4..V6 (ViT Tiny/Small/Base)
# Definisi tiap model: models/ablasi/
#
# Jalankan dari ROOT repo (sebaiknya di dalam tmux):
#   bash eksperimen_split_resmi/run_15_model.sh            # semua 15 model berurutan
#   bash eksperimen_split_resmi/run_15_model.sh h7 h8 v4   # hanya model tertentu
#
# - Model yang sudah punya final_summary.json DILEWATI (termasuk yang dulu dijalankan
#   dengan nama lama, misalnya c_tiny_v_tiny_freeze atau hybrid_freeze).
# - Model yang terputus di tengah jalan DILANJUTKAN (--resume) dari folder yang sama.
# - Jika satu model gagal, dicatat lalu lanjut ke model berikutnya; ringkasan di akhir.
# ==============================================================================
set -o pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DATA_DIR=${DATA_DIR:-./CelebA_Spoof}
WORKERS=${WORKERS:-2}          # server RAM 8 GB -> 2 worker
HASIL=eksperimen_split_resmi/hasil

if [ ! -f eksperimen_split_resmi/split/split.csv ]; then
    $PYTHON eksperimen_split_resmi/buat_split.py --data_dir "$DATA_DIR" --cek_file || exit 1
fi

# Nama lama yang mungkin sudah dipakai di server untuk model yang sama
alias_lama() {
    case "$1" in
        h1) echo "c_tiny_v_tiny" ;;   h2) echo "c_tiny_v_small" ;;  h3) echo "c_tiny_v_base" ;;
        h4) echo "c_small_v_tiny" ;;  h5) echo "c_small_v_small hybrid" ;;
        h6) echo "c_small_v_base" ;;  h7) echo "c_base_v_tiny" ;;   h8) echo "c_base_v_small" ;;
        h9) echo "c_base_v_base" ;;   c2) echo "convnext" ;;        *) echo "" ;;
    esac
}

if [ $# -ge 1 ]; then
    DAFTAR=("$@")
else
    DAFTAR=(h1 h2 h3 h4 h5 h6 h7 h8 h9 c1 c2 c3 v4 v5 v6)
fi

SELESAI=(); DILEWATI=(); GAGAL=()
echo "================================================================"
echo "TRAINING 15 MODEL (FREEZE) | workers=$WORKERS | mulai $(date)"
echo "Antrean: ${DAFTAR[*]}"
echo "================================================================"

for kode in "${DAFTAR[@]}"; do
    # Cari folder hasil yang sudah ada (nama resmi dulu, lalu nama lama)
    nama_dipakai="$kode"; sudah_selesai=0
    for n in "$kode" $(alias_lama "$kode"); do
        if [ -f "$HASIL/${n}_freeze/final_summary.json" ]; then
            sudah_selesai=1; nama_dipakai="$n"; break
        fi
        if [ -f "$HASIL/${n}_freeze/last_checkpoint.pth" ]; then
            nama_dipakai="$n"; break
        fi
    done

    if [ $sudah_selesai -eq 1 ]; then
        echo ">>> [$kode] sudah selesai di $HASIL/${nama_dipakai}_freeze -> dilewati"
        DILEWATI+=("$kode"); continue
    fi

    echo ""
    echo "================================================================"
    echo ">>> [$kode] mulai $(date)  (folder hasil: $HASIL/${nama_dipakai}_freeze)"
    echo "================================================================"
    if $PYTHON eksperimen_split_resmi/train.py --model "$nama_dipakai" --mode freeze \
            --data_dir "$DATA_DIR" --workers "$WORKERS" --resume; then
        SELESAI+=("$kode")
    else
        echo "!!! [$kode] GAGAL - lihat $HASIL/${nama_dipakai}_freeze/log_training.txt"
        GAGAL+=("$kode")
    fi
done

echo ""
echo "================================================================"
echo "RINGKASAN ($(date))"
echo "  Selesai sekarang : ${SELESAI[*]:-(tidak ada)}"
echo "  Sudah selesai    : ${DILEWATI[*]:-(tidak ada)}"
echo "  GAGAL            : ${GAGAL[*]:-(tidak ada)}"
echo "================================================================"
[ ${#GAGAL[@]} -eq 0 ]
