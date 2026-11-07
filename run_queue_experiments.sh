#!/bin/bash
# ==============================================================================
# Script Antrean Eksekusi Berurutan (Sequential Training Queue)
# TA-TI-Kel-04: Dual-Branch Hybrid Token S-Adapter (ConvNeXt + Vision Transformer)
# ==============================================================================
# Menjalankan 7 varian kombinasi skala model satu per satu secara berurutan.
# Setiap kali satu model selesai:
#   1. Checkpoint terbaik disimpan
#   2. Log metrik CSV & JSON diperbarui
#   3. Evaluasi Test Set selesai
#   4. RAM & VRAM server dibersihkan otomatis oleh sistem operasi
#   5. Model berikutnya langsung dimulai otomatis
# ==============================================================================

set -e

echo "===================================================================="
echo "MEMULAI ANTREAN TRAINING 7 KOMBINASI MODEL HYBRID TOKEN"
echo "Waktu Mulai: $(date)"
echo "===================================================================="

# Pastikan conda environment ta_env aktif
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ta_env
fi

# Antrean 1: ConvNeXt-Tiny + ViT-Small
echo ""
echo ">>> [1/7] Menjalankan: ConvNeXt-Tiny + ViT-Small ..."
python train_c_tiny_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 2: ConvNeXt-Tiny + ViT-Base
echo ""
echo ">>> [2/7] Menjalankan: ConvNeXt-Tiny + ViT-Base ..."
python train_c_tiny_v_base.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 3: ConvNeXt-Small + ViT-Tiny
echo ""
echo ">>> [3/7] Menjalankan: ConvNeXt-Small + ViT-Tiny ..."
python train_c_small_v_tiny.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 4: ConvNeXt-Small + ViT-Small
echo ""
echo ">>> [4/7] Menjalankan: ConvNeXt-Small + ViT-Small ..."
python train_c_small_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 5: ConvNeXt-Small + ViT-Base
echo ""
echo ">>> [5/7] Menjalankan: ConvNeXt-Small + ViT-Base ..."
python train_c_small_v_base.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 6: ConvNeXt-Base + ViT-Tiny
echo ""
echo ">>> [6/7] Menjalankan: ConvNeXt-Base + ViT-Tiny ..."
python train_c_base_v_tiny.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# Antrean 7: ConvNeXt-Base + ViT-Small
echo ""
echo ">>> [7/7] Menjalankan: ConvNeXt-Base + ViT-Small ..."
python train_c_base_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

echo ""
echo "===================================================================="
echo "SEMUA ANTREAN 7 KOMBINASI TELAH SELESAI DENGAN SUKSES!"
echo "Waktu Selesai: $(date)"
echo "===================================================================="

