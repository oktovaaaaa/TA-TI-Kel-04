"""
Pembangkit Koleksi Lengkap 15 Grafik Kurva Pelatihan Bab 4
==========================================================
Menghasilkan 15 file grafik beresolusi tinggi (200 DPI) dengan format
3-panel konsisten (Loss, Zoomed Accuracy, AUC & HTER) dan menyimpannya
dalam 1 folder tunggal: assets/kurva_pelatihan_15_model/
"""

import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("assets/kurva_pelatihan_15_model")

MODELS_SPEC = [
    # --- 9 MODEL USULAN HYBRID TOKEN (H1 - H9) ---
    {
        "id": "H1",
        "csv": "checkpoints_c_tiny_v_tiny/training_history.csv",
        "title": "H1: Hybrid (ConvNeXt-Tiny + ViT-Tiny)",
        "filename": "01_Hybrid_ConvNeXt-Tiny_ViT-Tiny.png",
    },
    {
        "id": "H2",
        "csv": "checkpoints_c_tiny_v_small/training_history.csv",
        "title": "H2: Hybrid (ConvNeXt-Tiny + ViT-Small)",
        "filename": "02_Hybrid_ConvNeXt-Tiny_ViT-Small.png",
    },
    {
        "id": "H3",
        "csv": "checkpoints_c_tiny_v_base/training_history.csv",
        "title": "H3: Hybrid (ConvNeXt-Tiny + ViT-Base)",
        "filename": "03_Hybrid_ConvNeXt-Tiny_ViT-Base.png",
    },
    {
        "id": "H4",
        "csv": "checkpoints_c_small_v_tiny/training_history.csv",
        "title": "H4: Hybrid (ConvNeXt-Small + ViT-Tiny)",
        "filename": "04_Hybrid_ConvNeXt-Small_ViT-Tiny.png",
    },
    {
        "id": "H5",
        "csv": "checkpoints_c_small_v_small/training_history.csv",
        "title": "H5: Hybrid (ConvNeXt-Small + ViT-Small) [BEST CHAMPION]",
        "filename": "05_Hybrid_ConvNeXt-Small_ViT-Small_BEST.png",
    },
    {
        "id": "H6",
        "csv": "checkpoints_c_small_v_base/training_history.csv",
        "title": "H6: Hybrid (ConvNeXt-Small + ViT-Base)",
        "filename": "06_Hybrid_ConvNeXt-Small_ViT-Base.png",
    },
    {
        "id": "H7",
        "csv": "checkpoints_c_base_v_tiny/training_history.csv",
        "title": "H7: Hybrid (ConvNeXt-Base + ViT-Tiny)",
        "filename": "07_Hybrid_ConvNeXt-Base_ViT-Tiny.png",
    },
    {
        "id": "H8",
        "csv": "checkpoints_c_base_v_small/training_history.csv",
        "title": "H8: Hybrid (ConvNeXt-Base + ViT-Small)",
        "filename": "08_Hybrid_ConvNeXt-Base_ViT-Small.png",
    },
    {
        "id": "H9",
        "csv": "checkpoints_c_base_v_base/training_history.csv",
        "title": "H9: Hybrid (ConvNeXt-Base + ViT-Base)",
        "filename": "09_Hybrid_ConvNeXt-Base_ViT-Base.png",
    },

    # --- 3 MODEL BASELINE CONVNEXT (B1 - B3) ---
    {
        "id": "B1",
        "csv": "checkpoints_convnext_tiny/training_history.csv",
        "title": "B1: Baseline ConvNeXt-Tiny + S-Adapter",
        "filename": "10_Baseline_ConvNeXt-Tiny.png",
    },
    {
        "id": "B2",
        "csv": "checkpoints_convnext_small/training_history.csv",
        "title": "B2: Baseline ConvNeXt-Small + S-Adapter",
        "filename": "11_Baseline_ConvNeXt-Small.png",
    },
    {
        "id": "B3",
        "csv": "checkpoints_convnext_base/training_history.csv",
        "title": "B3: Baseline ConvNeXt-Base + S-Adapter",
        "filename": "12_Baseline_ConvNeXt-Base.png",
    },

    # --- 3 MODEL BASELINE VIT (B4 - B6) ---
    {
        "id": "B4",
        "csv": "checkpoints_vit_tiny/training_history.csv",
        "title": "B4: Baseline ViT-Tiny + S-Adapter",
        "filename": "13_Baseline_ViT-Tiny.png",
    },
    {
        "id": "B5",
        "csv": "checkpoints_vit_small/training_history.csv",
        "title": "B5: Baseline ViT-Small + S-Adapter",
        "filename": "14_Baseline_ViT-Small.png",
    },
    {
        "id": "B6",
        "csv": "checkpoints_vit_base/training_history.csv",
        "title": "B6: Baseline ViT-Base + S-Adapter",
        "filename": "15_Baseline_ViT-Base.png",
    },
]

def render_plot(csv_file, title, out_path):
    df = pd.read_csv(csv_file)
    epochs = df["epoch"].tolist() if "epoch" in df else list(range(1, len(df) + 1))
    tr_loss = df["train_loss"].tolist()
    val_loss = df["val_loss"].tolist()
    tr_acc = df["train_acc"].tolist()
    val_acc = df["val_acc"].tolist()
    val_auc = df["val_auc"].tolist()
    val_hter = df["val_hter"].tolist()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), dpi=200)

    # 1. Panel Loss
    axes[0].plot(epochs, tr_loss, label="Train Loss", color="blue", linewidth=1.6)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="red", linewidth=1.6)
    axes[0].set_title(f"{title}\nLoss History", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Panel Accuracy (Auto-Zoomed dinamis)
    axes[1].plot(epochs, tr_acc, label="Train Acc", color="blue", linewidth=1.6)
    axes[1].plot(epochs, val_acc, label="Val Acc", color="green", linewidth=1.6)
    axes[1].set_title(f"{title}\nAccuracy (Zoomed)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("Accuracy", fontsize=10)
    axes[1].legend(loc="lower right", frameon=True)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Panel AUC & HTER
    axes[2].plot(epochs, val_auc, label="Val AUC", color="purple", linewidth=1.6)
    axes[2].plot(epochs, val_hter, label="Val HTER", color="orange", linewidth=1.6)
    axes[2].set_title(f"{title}\nAUC & HTER", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].legend(loc="center right", frameon=True)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 85)
    print(f"MEMBUAT KOLEKSI LENGKAP 15 GRAFIK MODEL DI FOLDER: {OUTPUT_DIR}")
    print("=" * 85)

    success = 0
    missing = []
    for spec in MODELS_SPEC:
        csv_path = Path(spec["csv"])
        out_file = OUTPUT_DIR / spec["filename"]

        if csv_path.exists():
            render_plot(csv_path, spec["title"], out_file)
            success += 1
            print(f"[OK] {spec['id']:<3} | {spec['title']:<52} -> {spec['filename']}")
        else:
            missing.append(spec)
            print(f"[BELUM ADA CSV] {spec['id']:<3} | {spec['title']:<52} ({csv_path})")

    print("=" * 85)
    print(f"[SELESAI] Sebanyak {success}/15 grafik model berhasil disimpan di:")
    print(f"👉 {OUTPUT_DIR.resolve()}\n")

    if missing:
        print("Model yang CSV-nya belum terunduh ke laptop:")
        for m in missing:
            print(f"  - {m['id']}: {m['title']}")
        print("\nUntuk mengunduh CSV yang kurang dalam 1 detik, jalankan di PowerShell laptop:")
        for m in missing:
            folder = Path(m['csv']).parent.name
            print(f"  scp glory@TA-Server-Tailscale:~/TA-TI-Kel4/{folder}/training_history.* {folder}/")

if __name__ == "__main__":
    main()
