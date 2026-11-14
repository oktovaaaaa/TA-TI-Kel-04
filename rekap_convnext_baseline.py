"""
Skrip Rekapitulasi & Perbandingan Hasil Pelatihan ConvNeXt + S-Adapter (Baseline B1, B2, B3)
=============================================================================================
Menampilkan tabel perbandingan metrik evaluasi Test Set Unseen Subjects (83.658 citra)
serta mengekspor hasilnya ke format CSV untuk Naskah Skripsi Bab 4.

Penggunaan:
  python rekap_convnext_baseline.py
"""

import os
import json
import glob
from pathlib import Path
import pandas as pd

def main():
    folders = [
        ("B1", "ConvNeXt_Tiny_SAdapter", "checkpoints_convnext_tiny", 768),
        ("B2", "ConvNeXt_Small_SAdapter", "checkpoints_convnext_small", 768),
        ("B3", "ConvNeXt_Base_SAdapter", "checkpoints_convnext_base", 1024),
    ]

    records = []

    print("=" * 115)
    print(f"{'Kode':<4} | {'Model Baseline':<25} | {'d':<5} | {'Best Ep':<7} | {'Akurasi':<8} | {'AUC':<8} | {'HTER':<8} | {'APCER':<8} | {'BPCER':<8} | {'Ukuran .pth':<11}")
    print("-" * 115)

    for code, name, folder_name, dim in folders:
        folder = Path(folder_name)
        sum_file = folder / "final_test_summary.json"
        cfg_file = folder / "training_config.json"

        # Cari file .pth
        pth_files = list(folder.glob("best_*.pth"))
        pth_size_mb = f"{pth_files[0].stat().st_size / (1024*1024):.1f} MB" if pth_files else "N/A"

        if sum_file.exists():
            try:
                with open(sum_file, "r") as fp:
                    data = json.load(fp)
                m = data.get("final_test_set_evaluation", {})
                best_ep = data.get("best_checkpoint", {}).get("best_epoch", "-")
                dur = data.get("total_duration_minutes", 0.0)

                # Baca config parameter jika ada
                params_str = "-"
                pct_str = "-"
                if cfg_file.exists():
                    try:
                        with open(cfg_file, "r") as cfp:
                            cfg = json.load(cfp)
                            pe = cfg.get("parameter_efficiency", {})
                            params_str = f"{pe.get('total_parameters', 0)/1e6:.1f}M"
                            pct_str = pe.get("trainable_percentage", "-")
                    except Exception:
                        pass

                records.append({
                    "Kode": code,
                    "Model": name,
                    "Dimensi (d)": dim,
                    "Total Params": params_str,
                    "Trainable %": pct_str,
                    "Best Epoch": best_ep,
                    "Akurasi (%)": m.get("test_accuracy_pct", f"{m.get('test_accuracy', 0)*100:.2f}%"),
                    "AUC": f"{m.get('test_auc', 0):.4f}",
                    "HTER (%)": m.get("test_hter_pct", f"{m.get('test_hter', 0)*100:.2f}%"),
                    "APCER (%)": m.get("test_apcer_pct", f"{m.get('test_apcer', 0)*100:.2f}%"),
                    "BPCER (%)": m.get("test_bpcer_pct", f"{m.get('test_bpcer', 0)*100:.2f}%"),
                    "Ukuran File": pth_size_mb,
                    "Durasi (menit)": dur,
                })

                acc = m.get("test_accuracy_pct", f"{m.get('test_accuracy', 0)*100:.2f}%")
                auc = f"{m.get('test_auc', 0):.4f}"
                hter = m.get("test_hter_pct", f"{m.get('test_hter', 0)*100:.2f}%")
                apcer = m.get("test_apcer_pct", f"{m.get('test_apcer', 0)*100:.2f}%")
                bpcer = m.get("test_bpcer_pct", f"{m.get('test_bpcer', 0)*100:.2f}%")

                print(f"{code:<4} | {name:<25} | {dim:<5} | {str(best_ep):<7} | {acc:<8} | {auc:<8} | {hter:<8} | {apcer:<8} | {bpcer:<8} | {pth_size_mb:<11}")
            except Exception as e:
                print(f"{code:<4} | {name:<25} | Gagal membaca file ({e})")
        else:
            print(f"{code:<4} | {name:<25} | Belum selesai / File tidak ditemukan")

    print("=" * 115)

    if records:
        df = pd.DataFrame(records)
        csv_path = "rekap_convnext_baseline.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[Info] Hasil perbandingan berhasil diekspor ke: {csv_path}\n")

if __name__ == "__main__":
    main()
