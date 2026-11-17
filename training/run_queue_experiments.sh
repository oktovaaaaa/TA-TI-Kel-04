#!/bin/bash
# ==============================================================================
# Script Antrean Eksekusi Berurutan (Lengkap 9 Kombinasi Model Matriks 3x3)
# TA-TI-Kel-04: Dual-Branch Hybrid Token S-Adapter (ConvNeXt + Vision Transformer)
# ==============================================================================
# Menjalankan seluruh 9 varian kombinasi secara berurutan:
#   1. ConvNeXt-Tiny  + ViT-Tiny
#   2. ConvNeXt-Tiny  + ViT-Small
#   3. ConvNeXt-Tiny  + ViT-Base
#   4. ConvNeXt-Small + ViT-Tiny
#   5. ConvNeXt-Small + ViT-Small
#   6. ConvNeXt-Small + ViT-Base
#   7. ConvNeXt-Base  + ViT-Tiny
#   8. ConvNeXt-Base  + ViT-Small
#   9. ConvNeXt-Base  + ViT-Base
#
# Setiap kali satu model selesai:
#   - Checkpoint terbaik disimpan berdasarkan Multi-Metric Composite Score
#   - Log CSV & JSON tersimpan
#   - Evaluasi Test Set selesai
#   - Memori RAM & GPU dibersihkan otomatis
#   - Model berikutnya langsung dimulai otomatis
# ==============================================================================

set -e

echo "===================================================================="
echo "MEMULAI ANTREAN TRAINING LENGKAP 9 KOMBINASI MODEL HYBRID TOKEN"
echo "Waktu Mulai: $(date)"
echo "===================================================================="

# Pastikan conda environment ta_env aktif
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate ta_env
fi

# [1/9] ConvNeXt-Tiny + ViT-Tiny
echo ""
echo ">>> [1/9] Menjalankan: ConvNeXt-Tiny + ViT-Tiny ..."
python train_c_tiny_v_tiny.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [2/9] ConvNeXt-Tiny + ViT-Small
echo ""
echo ">>> [2/9] Menjalankan: ConvNeXt-Tiny + ViT-Small ..."
python train_c_tiny_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [3/9] ConvNeXt-Tiny + ViT-Base
echo ""
echo ">>> [3/9] Menjalankan: ConvNeXt-Tiny + ViT-Base ..."
python train_c_tiny_v_base.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [4/9] ConvNeXt-Small + ViT-Tiny
echo ""
echo ">>> [4/9] Menjalankan: ConvNeXt-Small + ViT-Tiny ..."
python train_c_small_v_tiny.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [5/9] ConvNeXt-Small + ViT-Small
echo ""
echo ">>> [5/9] Menjalankan: ConvNeXt-Small + ViT-Small ..."
python train_c_small_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [6/9] ConvNeXt-Small + ViT-Base
echo ""
echo ">>> [6/9] Menjalankan: ConvNeXt-Small + ViT-Base ..."
python train_c_small_v_base.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [7/9] ConvNeXt-Base + ViT-Tiny
echo ""
echo ">>> [7/9] Menjalankan: ConvNeXt-Base + ViT-Tiny ..."
python train_c_base_v_tiny.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [8/9] ConvNeXt-Base + ViT-Small
echo ""
echo ">>> [8/9] Menjalankan: ConvNeXt-Base + ViT-Small ..."
python train_c_base_v_small.py --epochs 25 --patience 5 --batch_size 64 --workers 8

# [9/9] ConvNeXt-Base + ViT-Base
echo ""
echo ">>> [9/9] Menjalankan: ConvNeXt-Base + ViT-Base ..."
python train_c_base_v_base.py --epochs 25 --patience 5 --batch_size 64 --workers 8

echo ""
echo "===================================================================="
echo "SEMUA ANTREAN LENGKAP 9 KOMBINASI MODEL TELAH SELESAI DENGAN SUKSES!"
echo "Waktu Selesai: $(date)"
echo "===================================================================="
