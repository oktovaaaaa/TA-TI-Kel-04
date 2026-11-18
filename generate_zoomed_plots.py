"""
Skrip Pembangkit Grafik Kurva Pelatihan Standar Bab 4 (Format 3 Panel Zoomed)
=============================================================================
Menghasilkan grafik kurva pelatihan berstandar publikasi (3 panel) dengan
skala akurasi yang di-zoom secara dinamis (persis seperti format model Hybrid).

3 Panel:
  1. Loss (Train Loss vs Val Loss)
  2. Accuracy (Train Acc vs Val Acc - Zoomed)
  3. AUC & HTER (Val AUC vs Val HTER)

Output disimpan di:
  - assets/kurva_pelatihan/ (koleksi lengkap untuk Bab 4 Skripsi)
  - folder checkpoints_* masing-masing model
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def generate_plot(csv_path, title_name, save_paths):
    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Gagal membaca {csv_path}: {e}")
        return False

    epochs = df["epoch"].tolist() if "epoch" in df else list(range(1, len(df) + 1))
    tr_loss = df["train_loss"].tolist()
    val_loss = df["val_loss"].tolist()
    tr_acc = df["train_acc"].tolist()
    val_acc = df["val_acc"].tolist()
    val_auc = df["val_auc"].tolist()
    val_hter = df["val_hter"].tolist()

    # Buat figure 3 panel persis seperti format Hybrid
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), dpi=200)

    # 1. Panel Loss
    axes[0].plot(epochs, tr_loss, label="Train Loss", color="blue", linewidth=1.6)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="red", linewidth=1.6)
    axes[0].set_title(f"{title_name}: Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Panel Accuracy (Auto-Zoomed seperti Hybrid)
    axes[1].plot(epochs, tr_acc, label="Train Acc", color="blue", linewidth=1.6)
    axes[1].plot(epochs, val_acc, label="Val Acc", color="green", linewidth=1.6)
    axes[1].set_title(f"{title_name}: Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Panel AUC & HTER
    axes[2].plot(epochs, val_auc, label="Val AUC", color="purple", linewidth=1.6)
    axes[2].plot(epochs, val_hter, label="Val HTER", color="orange", linewidth=1.6)
    axes[2].set_title(f"{title_name}: AUC & HTER", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].legend(loc="center right", frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    for p in save_paths:
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=200)
    plt.close()
    return True

def main():
    target_models = [
        # --- BASELINE CONVNEXT ---
        ("checkpoints_convnext_tiny/training_history.csv", "ConvNeXt-Tiny + S-Adapter", [
            "assets/kurva_pelatihan/kurva_convnext_tiny.png",
            "checkpoints_convnext_tiny/training_history_tiny.png",
            "checkpoints_convnext_tiny/training_history_zoomed.png"
        ]),
        ("checkpoints_convnext_small/training_history.csv", "ConvNeXt-Small + S-Adapter", [
            "assets/kurva_pelatihan/kurva_convnext_small.png",
            "checkpoints_convnext_small/training_history_small.png",
            "checkpoints_convnext_small/training_history_zoomed.png"
        ]),
        ("checkpoints_convnext_base/training_history.csv", "ConvNeXt-Base + S-Adapter", [
            "assets/kurva_pelatihan/kurva_convnext_base.png",
            "checkpoints_convnext_base/training_history_base.png",
            "checkpoints_convnext_base/training_history_zoomed.png"
        ]),

        # --- BASELINE VIT ---
        ("checkpoints_vit_tiny/training_history.csv", "ViT-Tiny + S-Adapter", [
            "assets/kurva_pelatihan/kurva_vit_tiny.png",
            "checkpoints_vit_tiny/training_history_tiny.png",
            "checkpoints_vit_tiny/training_history_zoomed.png"
        ]),
        ("checkpoints_vit_small/training_history.csv", "ViT-Small + S-Adapter", [
            "assets/kurva_pelatihan/kurva_vit_small.png",
            "checkpoints_vit_small/training_history_small.png",
            "checkpoints_vit_small/training_history_zoomed.png"
        ]),
        ("checkpoints_vit_base/training_history.csv", "ViT-Base + S-Adapter", [
            "assets/kurva_pelatihan/kurva_vit_base.png",
            "checkpoints_vit_base/training_history_base.png",
            "checkpoints_vit_base/training_history_zoomed.png"
        ]),

        # --- HYBRID CHAMPION (H5) ---
        ("checkpoints_c_small_v_small/training_history.csv", "ConvNeXt-Small + ViT-Small [BEST]", [
            "assets/kurva_pelatihan/kurva_hybrid_c_small_v_small_champion.png"
        ]),
        ("checkpoints_c_tiny_v_base/training_history.csv", "ConvNeXt-Tiny + ViT-Base", [
            "assets/kurva_pelatihan/kurva_hybrid_c_tiny_v_base.png"
        ]),
    ]

    print("=" * 80)
    print("MEMBANGKITKAN GRAFIK KURVA PELATIHAN DENGAN FORMAT 3-PANEL ZOOMED")
    print("=" * 80)

    success_count = 0
    for csv_file, title, out_paths in target_models:
        if os.path.exists(csv_file):
            ok = generate_plot(csv_file, title, out_paths)
            if ok:
                success_count += 1
                print(f"[OK] {title:<35} -> Disimpan di: {out_paths[0]}")
        else:
            print(f"[SKIP] {title:<35} (CSV belum ada di: {csv_file})")

    print("=" * 80)
    print(f"[SELESAI] Sebanyak {success_count} grafik berhasil dibangkitkan dengan resolusi tinggi (200 DPI)!")
    print(f"Folder koleksi Bab 4: assets/kurva_pelatihan/\n")

if __name__ == "__main__":
    main()
