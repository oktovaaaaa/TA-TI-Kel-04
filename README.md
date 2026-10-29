# Hybrid S-Adapter + ConvNeXt for Face Anti-Spoofing
## Dokumentasi Lengkap Proyek, Penjelasan Arsitektur, Matematika Konversi Token, dan Komparasi Baseline

Proyek ini adalah implementasi sistem **Face Anti-Spoofing** (deteksi keaslian wajah vs serangan foto cetak/layar HP) menggunakan penggabungan dua arsitektur *deep learning* modern: **Vision Transformer (ViT-B/16) dengan S-Adapter** dan **ConvNeXt-Tiny**.

---

## 📑 Daftar Isi
1. [Konsep Utama & Latar Belakang](#1-konsep-utama--latar-belakang)
2. [Bagaimana Cara Token 1D dan Feature Map 2D Digabungkan?](#2-bagaimana-cara-token-1d-dan-feature-map-2d-digabungkan)
3. [Di Mana Lokasi Penggabungan Terjadi? (Late Fusion vs Per-Stage)](#3-di-mana-lokasi-penggabungan-terjadi)
4. [Struktur File & Kode Program](#4-struktur-file--kode-program)
5. [Hasil Eksperimen & Perbandingan Model (Hybrid vs Baseline)](#5-hasil-eksperimen--perbandingan-model)
6. [Panduan Menjalankan Program & Testing Webcam](#6-panduan-menjalankan-program--testing-webcam)
7. [Pertanyaan Panduan untuk AI Lain / Dosen Penguji](#7-pertanyaan-panduan-untuk-ai-lain--dosen-penguji)

---

## 1. Konsep Utama & Latar Belakang

Sistem *Face Anti-Spoofing* membutuhkan 2 jenis penglihatan sekaligus:
1. **Konteks Spasial Global:** Memahami bentuk keseluruhan kepala, pencahayaan ruangan, dan latar belakang (Ditangani oleh **Vision Transformer ViT-B/16 + S-Adapter**).
2. **Tekstur Spasial Lokal:** Memahami pori-pori kulit, batas piksel layar HP, atau serat kertas foto (Ditangani oleh **ConvNeXt-Tiny**).

### Mengapa Digunakan S-Adapter?
`S-Adapter` (Sequential Adapter) adalah modul bottleneck kecil ($d=64$) yang disisipkan di setiap Transformer Block ViT. Tujuannya adalah **Parameter-Efficient Fine-Tuning (PEFT)**:
* Bobot utama ViT dibekukan (*frozen*), hanya parameter S-Adapter yang dilatih.
* Mencegah *overfitting* saat dilatih pada jumlah data terbatas.

---

## 2. Bagaimana Cara Token 1D dan Feature Map 2D Digabungkan?

### Masalah Utama (*Mismatch Output*):
* **ViT Output:** Berbentuk **Token 1D** `(Batch, 197, 768)` $\rightarrow$ 1 CLS Token + 196 Patch Tokens.
* **ConvNeXt Output:** Berbentuk **Feature Map 2D** `(Batch, 768, 7, 7)` $\rightarrow$ Peta Grid $7 \times 7$.

### 5 Langkah Matematika & Logika Konversi (`TokenToFeatureMapBridge`):

1. **Asal Token ViT:**
   Gambar $224 \times 224$ dipotong menjadi $14 \times 14 = 196$ kotak (patch $16 \times 16$). Jadi 196 token tersebut sebenarnya berasal dari koordinat posisi grid $14 \times 14$.

2. **Separasi CLS Token & Spatial Tokens:**
   `CLS Token` (indeks 0) dipisahkan untuk fitur semantik global `(B, 128)`. Sisa 196 token patch diisolasi `(B, 196, 768)`.

3. **Reshape 1D ke Grid 2D:**
   Karena $\sqrt{196} = 14$, 196 token 1D dilipat kembali menjadi matriks 2D:
   $$\text{Sequence 1D: } (B, 196, 768) \xrightarrow{\text{permute \& reshape}} \text{Grid 2D: } (B, 768, 14, 14)$$

4. **Spatial Bilinear Interpolation ($14 \times 14 \rightarrow 7 \times 7$):**
   Grid $14 \times 14$ ViT diperkecil menjadi $7 \times 7$ menggunakan `F.interpolate` agar ukurannya **tepat sama** dengan output ConvNeXt Stage 4 `(B, 768, 7, 7)`.

5. **Concatenation & $1 \times 1$ Conv Fusion:**
   Kedua feature map $7 \times 7$ ditumpuk pada dimensi channel:
   $$(B, 768, 7, 7) + (B, 768, 7, 7) \xrightarrow{\text{torch.cat}} (B, 1536, 7, 7) \xrightarrow{1\times 1 \text{ Conv}} (B, 512, 7, 7)$$

---

## 3. Di Mana Lokasi Penggabungan Terjadi?

Penggabungan dilakukan di **AKHIR STAGE (Late Feature-Level Fusion)** setelah **Stage 4 ConvNeXt** dan **Block 12 ViT+S-Adapter**, BUKAN di setiap intermediate stage.

### Mengapa Late Fusion Dipilih?
1. **Inkompatibilitas ViT Isotropik:** ViT-B/16 mempertahankan jumlah token konstan (196 token) dari Block 1 sampai 12. Menggabungkan per-stage dengan ConvNeXt yang ukurannya mengecil ($56\times56 \rightarrow 28\times28 \rightarrow 14\times14 \rightarrow 7\times7$) akan merusak pretrained weights bawaan ViT.
2. **High-Level Semantic Features:** Stage 4 dan Block 12 memuat abstrak semantik paling matang mengenai indikasi pemalsuan wajah.
3. **Hemat Memori GPU:** Cross-Attention pada grid $7 \times 7$ jauh lebih ringan dibanding pada grid $56 \times 56$ (mencegah *CUDA Out of Memory*).

---

## 4. Struktur File & Kode Program (Server & Jupyter Ready)

```text
Test_TA/
├── models/
│   └── s_adapter_convnext_hybrid.py    # Arsitektur: Dual-Branch Hybrid Token S-Adapter (ConvNeXt-Tiny + ViT-Tiny)
├── checkpoints/
│   └── best_token_hybrid_model.pth     # Checkpoint model Hybrid Token terbaik
├── CelebA_Spoof/                       # Dataset resmi CelebA-Spoof (Train/Test + Bounding Box txt)
├── Hybrid_Token_ConvNeXt_Tiny_ViT_Tiny.ipynb  # Notebook Jupyter interaktif (visualisasi, training, evaluasi)
├── train_hybrid_token.py               # Script training utama di server (Focal Loss, LR bertingkat, BBox scaling)
├── test_image.py                       # Skrip inferensi 1 file foto wajah
├── test_webcam.py                      # Skrip inferensi webcam real-time
├── requirements.txt                    # Daftar dependensi library
├── .gitignore                          # Filter file bobot besar & cache git
└── README.md                           # Dokumentasi proyek
```

---

## 5. Hasil Evaluasi Model

Pengujian pada dataset CelebA-Spoof dengan evaluasi ISO/IEC Face Anti-Spoofing:

| Metrik Evaluasi | Hybrid Token (ViT-Tiny + ConvNeXt-Tiny + S-Adapter) | Keterangan |
| :--- | :--- | :--- |
| **Accuracy (%)** | **83.64%** | Evaluasi pada data independen subjek |
| **AUC Score** | **0.8842** | Kemampuan memisahkan kelas Live vs Spoof |
| **HTER Error Rate (%)** | **16.36%** | Rata-rata error (APCER + BPCER) / 2 |
| **Trainable Params (M)** | **16.81M (48.5%)** | Hemat memori GPU berkat PEFT S-Adapter |
| **Total Params (M)** | 34.68M | Jauh lebih ringan dibanding ViT-Base (119M) |

---

## 6. Panduan Menjalankan Program

### Pilihan 1: Melalui Jupyter Notebook (Interaktif & Visual)
Buka file **`Hybrid_Token_ConvNeXt_Tiny_ViT_Tiny.ipynb`** menggunakan VS Code, Jupyter Lab, atau Google Colab.
* Cell 1 - 4: Eksplorasi dataset & visualisasi sampel wajah Live vs Spoof beserta Bounding Box.
* Cell 5 - 7: Arsitektur ConvNeXt-Tiny + ViT-Tiny dengan S-Adapter dan verifikasi tensor sesuai diagram.
* Cell 8 - 10: Training interaktif dengan progress bar tqdm.
* Cell 11 - 13: Plot grafik training, kurva ROC, dan inferensi visual foto wajah baru.

### Pilihan 2: Melalui Command Line di Server GPU (Background / Non-Stop)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Mode Cepat (500 sampel untuk sanity check)
python train_hybrid_token.py --samples 500 --epochs 10 --batch_size 16

# Mode Full Training di GPU Server (Gunakan tmux / nohup agar tidak putus)
nohup python train_hybrid_token.py --samples -1 --epochs 25 --batch_size 32 --workers 4 > training.log 2>&1 &

# Memantau log training realtime di server
tail -f training.log
```

### C. Pengujian Inferensi 1 Foto Wajah
```bash
python test_image.py --image path/ke/foto_wajah.jpg
```

### D. Pengujian Realtime Webcam (Lokal)
```bash
python test_webcam.py
```

---

## 7. Pertanyaan Panduan untuk AI Lain / Dosen Penguji

Jika Anda ingin menanyakan atau memverifikasi proyek ini ke AI lain (seperti ChatGPT, Claude, GPT-4o, atau DeepSeek), Anda bisa menyalin prompt berikut:

> *"Saya memiliki proyek Face Anti-Spoofing yang menggabungkan ViT-B/16 (dengan S-Adapter) dan ConvNeXt-Tiny. ViT menghasilkan token 1D (197, 768) dan ConvNeXt Stage 4 menghasilkan feature map 2D (768, 7, 7). Model ini memisahkan CLS token, lalu meng-reshape 196 patch token menjadi grid 2D 14x14, menurunkan ukurannya via bilinear interpolation menjadi 7x7, lalu menumpuknya (concat) dengan feature map ConvNeXt dan memprosesnya via 1x1 Conv serta Multi-Head Cross Attention. Apakah teknik konversi token ke 2D feature map ini secara teori deep learning valid dan efisien?"*
