"""
Skrip Master Rekapitulasi Seluruh 15 Model Face Anti-Spoofing
=============================================================
Menggabungkan hasil evaluasi Test Set Unseen Subjects (83.658 citra)
untuk seluruh eksperimen Tugas Akhir:
1. Model Usulan Dual-Branch Hybrid Token S-Adapter (H1 s/d H9)
2. Baseline Single-Branch ConvNeXt + S-Adapter (B1 s/d B3)
3. Baseline Single-Branch Vision Transformer + S-Adapter (B4 s/d B6)

Penggunaan:
  python rekap_semua_15_model.py
"""

import json
from pathlib import Path
import pandas as pd

def main():
    experiments = [
        # --- KELOMPOK 1: METODE USULAN HYBRID TOKEN S-ADAPTER (H1 - H9) ---
        ("H1", "Hybrid_CTiny_VTiny", "checkpoints_c_tiny_v_tiny", "Hybrid (ConvNeXt-Tiny + ViT-Tiny)", 192),
        ("H2", "Hybrid_CTiny_VSmall", "checkpoints_c_tiny_v_small", "Hybrid (ConvNeXt-Tiny + ViT-Small)", 384),
        ("H3", "Hybrid_CTiny_VBase", "checkpoints_c_tiny_v_base", "Hybrid (ConvNeXt-Tiny + ViT-Base)", 768),
        ("H4", "Hybrid_CSmall_VTiny", "checkpoints_c_small_v_tiny", "Hybrid (ConvNeXt-Small + ViT-Tiny)", 192),
        ("H5", "Hybrid_CSmall_VSmall", "checkpoints_c_small_v_small", "Hybrid (ConvNeXt-Small + ViT-Small) [BEST]", 384),
        ("H6", "Hybrid_CSmall_VBase", "checkpoints_c_small_v_base", "Hybrid (ConvNeXt-Small + ViT-Base)", 768),
        ("H7", "Hybrid_CBase_VTiny", "checkpoints_c_base_v_tiny", "Hybrid (ConvNeXt-Base + ViT-Tiny)", 192),
        ("H8", "Hybrid_CBase_VSmall", "checkpoints_c_base_v_small", "Hybrid (ConvNeXt-Base + ViT-Small)", 384),
        ("H9", "Hybrid_CBase_VBase", "checkpoints_c_base_v_base", "Hybrid (ConvNeXt-Base + ViT-Base)", 768),

        # --- KELOMPOK 2: BASELINE CONVNEXT-ONLY (B1 - B3) ---
        ("B1", "ConvNeXt_Tiny_SAdapter", "checkpoints_convnext_tiny", "Baseline ConvNeXt-Tiny Only", 768),
        ("B2", "ConvNeXt_Small_SAdapter", "checkpoints_convnext_small", "Baseline ConvNeXt-Small Only", 768),
        ("B3", "ConvNeXt_Base_SAdapter", "checkpoints_convnext_base", "Baseline ConvNeXt-Base Only", 1024),

        # --- KELOMPOK 3: BASELINE VIT-ONLY (B4 - B6) ---
        ("B4", "ViT_Tiny_SAdapter", "checkpoints_vit_tiny", "Baseline ViT-Tiny Only", 192),
        ("B5", "ViT_Small_SAdapter", "checkpoints_vit_small", "Baseline ViT-Small Only", 384),
        ("B6", "ViT_Base_SAdapter", "checkpoints_vit_base", "Baseline ViT-Base Only", 768),
    ]

    records = []

    print("=" * 125)
    print(f"{'Kode':<4} | {'Kategori / Arsitektur':<42} | {'d':<5} | {'Best Ep':<7} | {'Akurasi':<8} | {'AUC':<8} | {'HTER':<8} | {'APCER':<8} | {'BPCER':<8}")
    print("-" * 125)

    last_group = None
    for code, name, folder_name, desc, dim in experiments:
        group = "HYBRID" if code.startswith("H") else ("CONVNEXT" if int(code[1:]) <= 3 else "VIT")
        if last_group and group != last_group:
            print("-" * 125)
        last_group = group

        folder = Path(folder_name)
        sum_file = folder / "final_test_summary.json"
        cfg_file = folder / "training_config.json"

        if sum_file.exists():
            try:
                with open(sum_file, "r") as fp:
                    data = json.load(fp)
                m = data.get("final_test_set_evaluation", {})
                best_ep = data.get("best_checkpoint", {}).get("best_epoch", "-")
                dur = data.get("total_duration_minutes", 0.0)

                acc = m.get("test_accuracy_pct", f"{m.get('test_accuracy', 0)*100:.2f}%")
                auc = f"{m.get('test_auc', 0):.4f}"
                hter = m.get("test_hter_pct", f"{m.get('test_hter', 0)*100:.2f}%")
                apcer = m.get("test_apcer_pct", f"{m.get('test_apcer', 0)*100:.2f}%")
                bpcer = m.get("test_bpcer_pct", f"{m.get('test_bpcer', 0)*100:.2f}%")

                records.append({
                    "Kode": code,
                    "Nama Eksperimen": name,
                    "Deskripsi": desc,
                    "Dimensi (d)": dim,
                    "Best Epoch": best_ep,
                    "Akurasi (%)": acc,
                    "AUC": auc,
                    "HTER (%)": hter,
                    "APCER (%)": apcer,
                    "BPCER (%)": bpcer,
                    "Durasi (menit)": dur,
                })

                print(f"{code:<4} | {desc:<42} | {dim:<5} | {str(best_ep):<7} | {acc:<8} | {auc:<8} | {hter:<8} | {apcer:<8} | {bpcer:<8}")
            except Exception as e:
                print(f"{code:<4} | {desc:<42} | Gagal membaca ({e})")
        else:
            print(f"{code:<4} | {desc:<42} | Checkpoint belum ditemukan di {folder_name}")

    print("=" * 125)

    if records:
        df = pd.DataFrame(records)
        csv_path = "rekap_semua_15_model.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Success] Tabel master 15 model berhasil diekspor ke: {csv_path}\n")

if __name__ == "__main__":
    main()
