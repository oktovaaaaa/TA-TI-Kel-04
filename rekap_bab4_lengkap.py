"""
Skrip Otomatis Pembangkit Tabel Bab 4 Skripsi (Format Lengkap Seluruh Model)
===========================================================================
Mengekstrak seluruh metrik pelatihan (Train/Val/Test/Hyperparameter/ISO 30107-3)
dari file JSON di seluruh folder checkpoint (Hybrid, ConvNeXt, ViT, & Baseline).

Menghasilkan tabel komparatif komprehensif dan mengekspornya ke:
  1. tabel_bab4_rekap_seluruh_model.csv (bisa langsung dibuka di Microsoft Excel)
  2. tabel_bab4_rekap_seluruh_model.md  (format Markdown untuk Naskah Skripsi)

Penggunaan:
  python rekap_bab4_lengkap.py
"""

import os
import json
from pathlib import Path
import pandas as pd

def extract_model_data():
    models_def = [
        # --- KELOMPOK 1: MODEL USULAN DUAL-BRANCH HYBRID TOKEN S-ADAPTER (H1 - H9) ---
        ("H1", "Hybrid", "ConvNeXt-Tiny + ViT-Tiny", "checkpoints_c_tiny_v_tiny", 192),
        ("H2", "Hybrid", "ConvNeXt-Tiny + ViT-Small", "checkpoints_c_tiny_v_small", 384),
        ("H3", "Hybrid", "ConvNeXt-Tiny + ViT-Base", "checkpoints_c_tiny_v_base", 768),
        ("H4", "Hybrid", "ConvNeXt-Small + ViT-Tiny", "checkpoints_c_small_v_tiny", 192),
        ("H5", "Hybrid", "ConvNeXt-Small + ViT-Small [BEST]", "checkpoints_c_small_v_small", 384),
        ("H6", "Hybrid", "ConvNeXt-Small + ViT-Base", "checkpoints_c_small_v_base", 768),
        ("H7", "Hybrid", "ConvNeXt-Base + ViT-Tiny", "checkpoints_c_base_v_tiny", 192),
        ("H8", "Hybrid", "ConvNeXt-Base + ViT-Small", "checkpoints_c_base_v_small", 384),
        ("H9", "Hybrid", "ConvNeXt-Base + ViT-Base", "checkpoints_c_base_v_base", 768),

        # --- KELOMPOK 2: BASELINE SINGLE-BRANCH CONVNEXT + S-ADAPTER (B1 - B3) ---
        ("B1", "Baseline ConvNeXt", "ConvNeXt-Tiny + S-Adapter", "checkpoints_convnext_tiny", 768),
        ("B2", "Baseline ConvNeXt", "ConvNeXt-Small + S-Adapter", "checkpoints_convnext_small", 768),
        ("B3", "Baseline ConvNeXt", "ConvNeXt-Base + S-Adapter", "checkpoints_convnext_base", 1024),

        # --- KELOMPOK 3: BASELINE SINGLE-BRANCH VIT + S-ADAPTER (B4 - B6) ---
        ("B4", "Baseline ViT", "ViT-Tiny + S-Adapter", "checkpoints_vit_tiny", 192),
        ("B5", "Baseline ViT", "ViT-Small + S-Adapter", "checkpoints_vit_small", 384),
        ("B6", "Baseline ViT", "ViT-Base + S-Adapter", "checkpoints_vit_base", 768),

        # --- KELOMPOK 4: PROTOTIPE AWAL SEBELUM OPTIMASI TOKEN ADAPTIF (MODEL KE-16) ---
        ("P1", "Prototipe Awal", "ConvNeXt-Base + ViT-Base (Awal)", "checkpoints_base", 768),
    ]

    records = []

    for code, category, arch, folder_name, dim in models_def:
        folder = Path(folder_name)
        cfg_file = folder / "training_config.json"
        hist_file = folder / "training_history.json"
        sum_file = folder / "final_test_summary.json"

        # Nilai default
        bs = 64
        lr_adapter = "-"
        lr_head = "-"
        wd = "1e-4"
        total_p = "-"
        trainable_p = "-"
        trainable_pct = "-"
        best_ep = "-"
        tr_loss = "-"
        tr_acc = "-"
        val_loss = "-"
        val_acc = "-"
        test_acc = "-"
        test_auc = "-"
        test_hter = "-"
        test_apcer = "-"
        test_bpcer = "-"

        # 1. Baca Konfigurasi
        if cfg_file.exists():
            try:
                with open(cfg_file, "r") as fp:
                    cfg = json.load(fp)
                # Cek struktur config (hybrid vs baseline)
                hyp = cfg.get("training_hyperparameters", cfg)
                bs = hyp.get("batch_size", 64)
                wd = hyp.get("weight_decay", 1e-4)

                if "lr_adapter" in hyp:
                    lr_adapter = f"{hyp['lr_adapter']:.1e}"
                    lr_head = f"{hyp.get('lr_head', 1e-3):.1e}"
                elif "learning_rate_adapter" in hyp:
                    lr_adapter = f"{hyp['learning_rate_adapter']:.1e}"
                    lr_head = f"{hyp.get('learning_rate_head', 1e-3):.1e}"
                elif "learning_rate" in hyp:
                    lr_adapter = f"{hyp['learning_rate']:.1e}"
                    lr_head = "-"

                if "weight_decay_adapter" in hyp:
                    wd = f"{hyp['weight_decay_adapter']:.1e}"
                elif "weight_decay" in hyp:
                    wd = f"{hyp['weight_decay']:.1e}"

                pe = cfg.get("parameter_efficiency", {})
                if pe:
                    tot = pe.get("total_parameters", 0)
                    trn = pe.get("trainable_parameters", 0)
                    total_p = f"{tot/1e6:.1f}M"
                    trainable_p = f"{trn/1e6:.1f}M"
                    trainable_pct = pe.get("trainable_percentage", f"{(trn/tot*100):.2f}%" if tot > 0 else "-")
            except Exception:
                pass

        # 2. Baca Summary Test Set
        if sum_file.exists():
            try:
                with open(sum_file, "r") as fp:
                    s = json.load(fp)
                best_ep = s.get("best_checkpoint", {}).get("best_epoch", "-")
                m = s.get("final_test_set_evaluation", s.get("metrics_iso_30107_3", {}))
                test_acc = m.get("test_accuracy_pct", f"{m.get('test_accuracy', 0)*100:.2f}%")
                test_auc = f"{m.get('test_auc', 0):.4f}"
                test_hter = m.get("test_hter_pct", f"{m.get('test_hter', 0)*100:.2f}%")
                test_apcer = m.get("test_apcer_pct", f"{m.get('test_apcer', 0)*100:.2f}%")
                test_bpcer = m.get("test_bpcer_pct", f"{m.get('test_bpcer', 0)*100:.2f}%")
            except Exception:
                pass

        # 3. Baca History Training (Loss & Acc pada Best Epoch)
        if hist_file.exists():
            try:
                with open(hist_file, "r") as fp:
                    h = json.load(fp)
                if isinstance(h, list) and len(h) > 0:
                    target_ep = best_ep if isinstance(best_ep, int) else len(h)
                    bh = next((x for x in h if x.get("epoch") == target_ep), h[-1])
                    if "train_loss" in bh:
                        tr_loss = f"{bh['train_loss']:.5f}"
                    if "train_acc" in bh:
                        tr_acc = f"{bh['train_acc']*100:.2f}%"
                    if "val_loss" in bh:
                        val_loss = f"{bh['val_loss']:.5f}"
                    if "val_acc" in bh:
                        val_acc = f"{bh['val_acc']*100:.2f}%"
            except Exception:
                pass

        # Fallback jika di local folder belum ada tapi ada di CSV rekap
        if test_acc == "-" and Path("rekap_convnext_baseline.csv").exists():
            try:
                df_c = pd.read_csv("rekap_convnext_baseline.csv")
                row = df_c[df_c["Kode"] == code]
                if not row.empty:
                    r = row.iloc[0]
                    total_p = str(r["Total Params"])
                    trainable_pct = str(r["Trainable %"])
                    best_ep = int(r["Best Epoch"])
                    test_acc = str(r["Akurasi (%)"])
                    test_auc = str(r["AUC"])
                    test_hter = str(r["HTER (%)"])
                    test_apcer = str(r["APCER (%)"])
                    test_bpcer = str(r["BPCER (%)"])
            except Exception:
                pass

        if test_acc == "-" and Path("rekap_vit_baseline.csv").exists():
            try:
                df_v = pd.read_csv("rekap_vit_baseline.csv")
                row = df_v[df_v["Kode"] == code]
                if not row.empty:
                    r = row.iloc[0]
                    total_p = str(r["Total Params"])
                    trainable_pct = str(r["Trainable %"])
                    best_ep = int(r["Best Epoch"])
                    test_acc = str(r["Akurasi (%)"])
                    test_auc = str(r["AUC"])
                    test_hter = str(r["HTER (%)"])
                    test_apcer = str(r["APCER (%)"])
                    test_bpcer = str(r["BPCER (%)"])
            except Exception:
                pass

        records.append({
            "No": code,
            "Kategori": category,
            "Arsitektur Model": arch,
            "d": dim,
            "BS": bs,
            "LR (Adapter)": lr_adapter,
            "LR (Head)": lr_head,
            "WD": wd,
            "Trainable (%)": trainable_pct,
            "ES (Best Ep)": best_ep,
            "Train Loss": tr_loss,
            "Train Acc (%)": tr_acc,
            "Val Loss": val_loss,
            "Val Acc (%)": val_acc,
            "Test Acc (%)": test_acc,
            "Test AUC": test_auc,
            "Test HTER (%)": test_hter,
            "Test APCER (%)": test_apcer,
            "Test BPCER (%)": test_bpcer,
        })

    return records

def main():
    records = extract_model_data()
    df = pd.DataFrame(records)

    csv_path = "tabel_bab4_rekap_seluruh_model.csv"
    df.to_csv(csv_path, index=False)

    md_path = "tabel_bab4_rekap_seluruh_model.md"
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("# Tabel 4.1 Rekapitulasi Hasil Pelatihan Seluruh Model (Hybrid & Baseline)\n\n")
        try:
            fp.write(df.to_markdown(index=False))
        except Exception:
            # Fallback jika package tabulate belum terpasang
            headers = [str(c) for c in df.columns]
            fp.write("| " + " | ".join(headers) + " |\n")
            fp.write("| " + " | ".join([":---:" if c in ["No", "d", "BS", "ES (Best Ep)"] else ":---" for c in headers]) + " |\n")
            for _, r in df.iterrows():
                fp.write("| " + " | ".join([str(val) for val in r]) + " |\n")
        fp.write("\n")

    print("\n" + "=" * 135)
    print("TABEL REKAPITULASI BAB 4 SKRIPSI: SELURUH MODEL (HYBRID & NON-HYBRID BASELINE)")
    print("=" * 135)
    cols_preview = ["No", "Arsitektur Model", "ES (Best Ep)", "Train Loss", "Train Acc (%)", "Val Loss", "Val Acc (%)", "Test Acc (%)", "Test HTER (%)", "Test APCER (%)"]
    print(df[cols_preview].to_string(index=False))
    print("=" * 135)
    print(f"\n[SUKSES] File berhasil disimpan:")
    print(f"  1. Excel/CSV : {csv_path}")
    print(f"  2. Markdown  : {md_path}\n")

if __name__ == "__main__":
    main()
