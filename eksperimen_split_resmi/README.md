# Eksperimen Split Resmi CelebA-Spoof

Folder ini berisi program baru yang **terpisah** dari 15 model lama
(`training/`, `checkpoints_*`, `generate_bab4_evaluation_assets.py`).
Program lama tidak diubah sama sekali, sehingga masih bisa dipakai sebagai pembanding.

## Isi folder

| File | Fungsi |
|---|---|
| `buat_split.py` | Membuat `split/split.csv` dari file label resmi CelebA-Spoof |
| `common.py` | Dataset, metrik lengkap, pembuat model, optimizer, grafik |
| `train.py` | Training + validasi + test tiap epoch, evaluasi akhir |
| `run_semua.sh` | Menjalankan split + 4 eksperimen berurutan di server |
| `split/` | Hasil pembagian data (sudah dibuat, lihat tabel di bawah) |

## Aturan pembagian data

1. Label dibaca dari file resmi `metas/intra_test/train_label.txt` dan `test_label.txt`,
   **bukan** dari nama folder (2.022 citra di folder `live` berlabel resmi spoof).
2. **Test** = seluruh test resmi CelebA-Spoof (utuh).
3. **Validation** = sebagian **subjek** dari train resmi, dipilih acak (seed 42)
   sampai jumlah citranya sama dengan test.
4. **Train** = sisa subjek train resmi.
5. 3 subjek yang ada di train resmi **dan** test resmi (ID 5028, 7332, 9735) dibuang
   dari sisi train (216 citra), sehingga tidak ada orang yang sama antar bagian.
6. Konvensi label proyek: **1 = live, 0 = spoof**.

Hasil (`split/ringkasan_split.csv`):

| Bagian | Subjek | Citra | % Citra | Live | Spoof |
|---|---|---|---|---|---|
| Train | 7.064 | 427.000 | 76,07% | 140.344 | 286.656 |
| Validation | 1.125 | 67.189 | 11,97% | 22.059 | 45.130 |
| Test | 1.004 | 67.170 | 11,97% | 19.923 | 47.247 |
| Total | 9.193 | 561.359 | 100% | 182.326 | 379.033 |

## Model yang tersedia

| Perintah | Model | Parameter dilatih |
|---|---|---|
| `--model hybrid --mode freeze` | H5 (ConvNeXt-Small + ViT-Small), sama seperti model lama | 19,52 M dari 75,17 M (25,97%) |
| `--model hybrid --mode fullft` | H5 full fine-tuning | 75,17 M (100%) |
| `--model convnext --mode freeze` | B2 (ConvNeXt-Small + S-Adapter), sama seperti model lama | 15,77 M dari 49,75 M (31,70%) |
| `--model convnext --mode fullft` | B2 full fine-tuning | 49,75 M (100%) |

Learning rate:
- backbone (hanya mode `fullft`): `1e-5`
- S-Adapter + ConvNeXt stage 4: `1e-4` (sama dengan model lama)
- cross-attention + refinement + classifier: `1e-3` (sama dengan model lama)

Hyperparameter lain **sama dengan model lama**: 25 epoch, batch 64, AdamW, Cosine LR,
Focal Loss (γ=2, α=0,25), early stopping patience 5 dengan skor komposit
(AUC + Acc − HTER − Loss) pada **validation**, augmentasi yang sama, input 224×224.

## Cara menjalankan di server

Struktur yang dibutuhkan di server (dari root repo):
```
TA-TI-Kel-04/
├── models/                     (sudah ada)
├── eksperimen_split_resmi/     (folder ini, termasuk split/split.csv)
└── CelebA_Spoof/
    ├── Data/train, Data/test
    └── metas/intra_test/
```

Semua eksperimen berurutan:
```bash
bash eksperimen_split_resmi/run_semua.sh
```
Satu eksperimen saja:
```bash
python eksperimen_split_resmi/train.py --model hybrid --mode fullft --workers 8
```
Jika dataset ada di lokasi lain:
```bash
python eksperimen_split_resmi/train.py --model hybrid --mode fullft --data_dir /path/ke/CelebA_Spoof
```
Jika training terputus, jalankan perintah yang sama dengan `--resume` untuk melanjutkan
dari epoch terakhir.

Opsi lain: `--no_test_each_epoch` (lewati test tiap epoch, lebih cepat),
`--epochs`, `--batch_size`, `--patience`, `--lr_backbone`.

## Output per eksperimen (`hasil/<model>_<mode>/`)

| File | Isi |
|---|---|
| `log_training.txt` | Seluruh output terminal |
| `config.json` | Konfigurasi, jumlah parameter per grup, jumlah data |
| `training_history.csv` / `.json` | **Per epoch**: waktu train/val/test, throughput, learning rate, dan untuk train/val/test: loss, accuracy, balanced accuracy, precision/recall/F1 (live, spoof, macro), APCER, BPCER, HTER, AUC, EER, TP/TN/FP/FN, skor komposit, epoch terbaik, memori GPU |
| `training_history.png` | Grafik loss, accuracy, AUC, HTER, APCER/BPCER, waktu per epoch |
| `predictions_val.csv`, `predictions_test.csv` | Prediksi **setiap citra** (path, label asli, probabilitas live, prediksi, benar/salah) |
| `confusion_matrix_val.png`, `confusion_matrix_test.png` | Confusion matrix dinormalisasi per kelas + jumlah citra |
| `roc_curve_val.png`, `roc_curve_test.png` | Kurva ROC dari prediksi asli |
| `final_summary.json` | Hasil resmi: metrik test pada checkpoint terbaik + ringkasan waktu |
| `best_checkpoint.pth`, `last_checkpoint.pth` | Bobot model terbaik dan terakhir (untuk resume) |

**Catatan metodologi:** test dievaluasi tiap epoch hanya untuk analisis kurva.
Checkpoint terbaik dan early stopping dipilih dari **validation**. Angka yang dilaporkan
di skripsi adalah `hasil_test_epoch_terbaik` pada `final_summary.json`.
