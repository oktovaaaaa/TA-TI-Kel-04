"""
Script Rekap & Perbandingan Formal Seluruh 9 Kombinasi Model Hybrid Token
========================================================================
Membaca semua log 'final_test_summary.json' dan 'training_config.json'
dari folder checkpoints_* untuk menghasilkan:
1. Tabel perbandingan komprehensif di terminal
2. File CSV resmi (rekap_evaluasi_9_model.csv) untuk Bab 4 Skripsi / Excel
"""

import json
import glob
from pathlib import Path
import pandas as pd

def main():
    json_paths = sorted(glob.glob("checkpoints_*/final_test_summary.json"))
    if not json_paths:
        print("[Peringatan] Tidak ditemukan file 'final_test_summary.json' di dalam folder checkpoints_*")
        return

    records = []
    for summary_path in json_paths:
        folder = Path(summary_path).parent
        config_path = folder / "training_config.json"
        
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                sum_data = json.load(f)
        except Exception as e:
            print(f"Error membaca {summary_path}: {e}")
            continue

        total_params = "N/A"
        trainable_pct = "N/A"
        if config_path.exists():
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                    pe = cfg_data.get("parameter_efficiency", {})
                    total_params = f"{pe.get('total_parameters', 0):,}"
                    trainable_pct = pe.get("trainable_percentage", "N/A")
            except:
                pass

        eval_m = sum_data.get("final_test_set_evaluation", {})
        best_ckpt = sum_data.get("best_checkpoint", {})
        
        # Ekstrak metrik
        acc = eval_m.get("test_accuracy_pct", f"{eval_m.get('test_accuracy', 0)*100:.2f}%")
        auc = eval_m.get("test_auc", 0.0)
        hter = eval_m.get("test_hter_pct", f"{eval_m.get('test_hter', 0)*100:.2f}%")
        apcer = eval_m.get("test_apcer_pct", f"{eval_m.get('test_apcer', 0)*100:.2f}%")
        bpcer = eval_m.get("test_bpcer_pct", f"{eval_m.get('test_bpcer', 0)*100:.2f}%")
        best_ep = best_ckpt.get("best_epoch", "N/A")
        hter_num = eval_m.get("test_hter", 999.0)

        records.append({
            "Folder": folder.name,
            "Model Kombinasi": sum_data.get("experiment_name", folder.name),
            "Best Epoch": best_ep,
            "Total Params": total_params,
            "Trainable (%)": trainable_pct,
            "Test Accuracy": acc,
            "Test AUC": f"{auc:.4f}",
            "Test HTER": hter,
            "Test APCER": apcer,
            "Test BPCER": bpcer,
            "_hter_num": hter_num
        })

    df = pd.DataFrame(records)
    # Urutkan berdasarkan HTER terendah (model terbaik)
    df = df.sort_values(by="_hter_num", ascending=True).reset_index(drop=True)
    df_clean = df.drop(columns=["_hter_num"])

    output_csv = "rekap_evaluasi_9_model.csv"
    df_clean.to_csv(output_csv, index=False)

    print("\n" + "="*105)
    print("                     TABEL REKAPITULASI RESMI HASIL PENGUJIAN TEST SET 9 MODEL")
    print("                              (Diurutkan dari HTER Terbaik / Terendah)")
    print("="*105)
    header = f"{'No':<3} | {'Model Kombinasi':<28} | {'Params':<12} | {'Epoch':<5} | {'Acc':<8} | {'AUC':<7} | {'HTER':<7} | {'APCER':<7} | {'BPCER':<7}"
    print(header)
    print("-" * 105)
    for idx, row in df_clean.iterrows():
        print(f"{idx+1:<3} | {row['Model Kombinasi']:<28} | {row['Total Params']:<12} | {str(row['Best Epoch']):<5} | {row['Test Accuracy']:<8} | {row['Test AUC']:<7} | {row['Test HTER']:<7} | {row['Test APCER']:<7} | {row['Test BPCER']:<7}")
    print("="*105)
    print(f"\n[SUKSES] Hasil rekap lengkap telah disimpan ke file: {output_csv}")
    print(f"[INFO] File CSV ini bisa langsung Anda buka di Microsoft Excel atau Google Sheets untuk Bab 4!\n")

if __name__ == "__main__":
    main()

