"""
Skrip Evaluasi Ilmiah Dataset Testing (CelebA-Spoof Test Set Evaluation)
========================================================================
Pengujian kuantitatif resmi untuk Bab 4 Skripsi / Tugas Akhir.
Mengevaluasi checkpoint model (Base atau Tiny) pada Test Set Unseen Subjects.

Metrik yang dihitung:
  - Accuracy (%)
  - Area Under the Curve (AUC)
  - Attack Presentation Classification Error Rate (APCER %)
  - Bona Fide Presentation Classification Error Rate (BPCER %)
  - Half Total Error Rate (HTER %)
  - Confusion Matrix & ROC Curve (disimpan ke gambar)

Penggunaan:
  python eval_test_set.py
  python eval_test_set.py --checkpoint ./checkpoints_base/best_token_hybrid_base_model.pth
  python eval_test_set.py --checkpoint ./checkpoints/best_token_hybrid_model.pth
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import time
import argparse
import random
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image

from models.s_adapter_convnext_hybrid import HybridTokenSAdapterConvNeXt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

class CelebASpoofDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, use_bbox: bool = True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.use_bbox = use_bbox

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row.path)
        try:
            img = read_image(str(img_path))
            if img.shape[0] == 4:
                img = img[:3]
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)

            if self.use_bbox:
                bb_path = Path(str(img_path).replace('.jpg', '_BB.txt').replace('.png', '_BB.txt'))
                if bb_path.exists():
                    with open(bb_path, 'r') as f:
                        parts = f.read().strip().split()
                        if len(parts) >= 4:
                            bbox = [float(p) for p in parts]
                            real_h, real_w = img.shape[1], img.shape[2]
                            x1 = max(0, int(bbox[0] * (real_w / 224.0)))
                            y1 = max(0, int(bbox[1] * (real_h / 224.0)))
                            w1 = int(bbox[2] * (real_w / 224.0))
                            h1 = int(bbox[3] * (real_h / 224.0))
                            x2 = min(real_w, x1 + w1)
                            y2 = min(real_h, y1 + h1)

                            if (x2 - x1) > 10 and (y2 - y1) > 10:
                                img = img[:, y1:y2, x1:x2]
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.uint8)

        if self.transform:
            img = self.transform(img.numpy().transpose(1, 2, 0))

        return img, int(row.label)

def build_manifest(data_root="./CelebA_Spoof"):
    data_root = Path(data_root)
    records = []
    for split in ['train', 'test']:
        split_dir = data_root / split
        if not split_dir.exists():
            split_dir = data_root / 'Data' / split
        if not split_dir.exists():
            continue
        for subj in split_dir.iterdir():
            if not subj.is_dir(): continue
            for t in subj.iterdir():
                if not t.is_dir(): continue
                label = 1 if t.name.lower() in ['1', 'live'] else 0
                for img in list(t.glob('*.jpg')) + list(t.glob('*.png')):
                    records.append({
                        'path': str(img),
                        'label': label,
                        'split': split,
                        'subject': subj.name
                    })
    return pd.DataFrame(records)

def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= threshold).astype(int)

    spoof_mask = (y_true == 0)
    apcer = float(np.mean(y_pred[spoof_mask] == 1)) if spoof_mask.sum() > 0 else 0.0

    live_mask = (y_true == 1)
    bpcer = float(np.mean(y_pred[live_mask] == 0)) if live_mask.sum() > 0 else 0.0

    hter = (apcer + bpcer) / 2.0
    acc = float(np.mean(y_pred == y_true))

    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.5

    return {
        "accuracy": acc,
        "auc": auc,
        "apcer": apcer,
        "bpcer": bpcer,
        "hter": hter,
    }

def load_trained_model(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[Error] File checkpoint tidak ditemukan di: {checkpoint_path}")
        sys.exit(1)

    print(f"\n[1/4] Memuat checkpoint dari: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))

    arch = ckpt.get("arch", "")
    is_base = "base" in str(checkpoint_path).lower() or "base" in arch.lower()

    if is_base:
        print("      -> Arsitektur: BASE (ConvNeXt-Base + ViT-Base, 12 Heads)")
        model = HybridTokenSAdapterConvNeXt(
            vit_name="vit_base_patch16_224",
            convnext_name="convnext_base",
            adapter_bottleneck=64,
            num_heads=12,
            num_classes=2,
            pretrained=False,
        ).to(DEVICE)
    else:
        print("      -> Arsitektur: TINY (ConvNeXt-Tiny + ViT-Tiny, 3 Heads)")
        model = HybridTokenSAdapterConvNeXt(
            vit_name="vit_tiny_patch16_224",
            convnext_name="convnext_tiny",
            adapter_bottleneck=64,
            num_heads=3,
            num_classes=2,
            pretrained=False,
        ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()
    epoch_info = ckpt.get("epoch", "N/A")
    val_acc = ckpt.get("val_acc", None)
    if val_acc:
        print(f"      -> Checkpoint Epoch {epoch_info} (Akurasi Val: {val_acc*100:.2f}%) sukses dimuat!")
    return model, is_base

def main():
    parser = argparse.ArgumentParser(description="Evaluasi Formal Model pada Test Set CelebA-Spoof")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints_base/best_token_hybrid_base_model.pth")
    parser.add_argument("--data_dir", type=str, default="./CelebA_Spoof")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4 if os.name != 'nt' else 0)
    parser.add_argument("--max_test_samples", type=int, default=-1, help="-1 untuk seluruh test set")
    args = parser.parse_args()

    # 1. Muat Model
    model, is_base = load_trained_model(args.checkpoint)

    # 2. Siapkan Test Dataset
    print(f"\n[2/4] Membangun manifest data dari: {args.data_dir} ...")
    df = build_manifest(args.data_dir)
    print(f"      -> Total file terdeteksi: {len(df):,}")

    subjs = list(df.subject.unique())
    if len(subjs) >= 3:
        # Menjaga konsistensi protokol splitting (15% Unseen Test Subjects persis saat training)
        train_subjs, test_subjs = train_test_split(subjs, test_size=0.15, random_state=SEED)
        df_test = df[df.subject.isin(test_subjs)].reset_index(drop=True)
    else:
        _, df_test = train_test_split(df, test_size=0.15, random_state=SEED)

    if args.max_test_samples > 0 and len(df_test) > args.max_test_samples:
        df_test = df_test.sample(n=args.max_test_samples, random_state=SEED).reset_index(drop=True)

    print(f"      -> Jumlah Data Test Set (Unseen): {len(df_test):,} foto")
    print(f"         * Foto Asli (Live) : {(df_test.label == 1).sum():,} foto")
    print(f"         * Foto Palsu (Spoof): {(df_test.label == 0).sum():,} foto")

    test_ds = CelebASpoofDataset(df_test, transform=val_transform, use_bbox=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=2 if args.workers > 0 else None
    )

    # 3. Jalankan Pengujian Kuantitatif
    use_amp = (DEVICE.type == "cuda")
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    precision_str = "BFloat16 (Hopper Native)" if use_bf16 else ("Float16" if use_amp else "FP32")
    print(f"\n[3/4] Menjalankan inferensi pada Test Set menggunakan device: {DEVICE} ({precision_str}) ...")

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets, all_scores, all_preds = [], [], []

    start_time = time.time()
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Progress"):
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                out = model(images)
                logits = out["logits"]
                probs = out["probs"]
                loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_scores.extend(probs[:, 1].cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    elapsed = time.time() - start_time
    avg_loss = total_loss / len(all_targets)
    metrics = calculate_metrics(all_targets, all_scores)
    metrics["loss"] = avg_loss

    # 4. Cetak Hasil Lengkap untuk Laporan TA
    print("\n" + "="*75)
    print("HASIL EVALUASI FORMAL TEST SET (CELEBA-SPOOF)")
    print("Arsitektur: " + ("ConvNeXt-Base + ViT-Base" if is_base else "ConvNeXt-Tiny + ViT-Tiny"))
    print("="*75)
    print(f"Total Sampel Diuji : {len(all_targets):,} foto")
    print(f"Waktu Evaluasi     : {elapsed:.1f} detik ({elapsed/len(all_targets)*1000:.2f} ms/foto)")
    print(f"Test Loss          : {metrics['loss']:.4f}")
    print(f"Accuracy           : {metrics['accuracy']*100:.2f}%")
    print(f"AUC (ROC)          : {metrics['auc']:.4f}")
    print(f"APCER (Error Spoof): {metrics['apcer']*100:.2f}%")
    print(f"BPCER (Error Live) : {metrics['bpcer']*100:.2f}%")
    print(f"HTER (Total Error) : {metrics['hter']*100:.2f}%")
    print("="*75)

    # Simpan Gambar Confusion Matrix dan ROC Curve
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Spoof (0)', 'Live (1)'])
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title(f"Confusion Matrix\nAccuracy: {metrics['accuracy']*100:.2f}% | HTER: {metrics['hter']*100:.2f}%")

    # Plot 2: ROC Curve
    fpr, tpr, _ = roc_curve(all_targets, all_scores)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC Curve (AUC = {metrics['auc']:.4f})")
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (FPR)')
    axes[1].set_ylabel('True Positive Rate (TPR)')
    axes[1].set_title('Receiver Operating Characteristic (ROC)')
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    chart_output = f"./eval_test_results_{'base' if is_base else 'tiny'}.png"
    plt.savefig(chart_output, dpi=150)
    
    # Simpan juga ke folder checkpoint
    ckpt_dir = Path(args.checkpoint).parent
    plt.savefig(ckpt_dir / f"eval_test_results_{'base' if is_base else 'tiny'}.png", dpi=150)
    plt.close()

    # Simpan ringkasan ke final_test_summary.json (Poin 1 & 4 Panduan Pembimbing)
    summary_data = {
        "architecture": "ConvNeXt-Base + ViT-Base" if is_base else "ConvNeXt-Tiny + ViT-Tiny",
        "evaluated_checkpoint": str(args.checkpoint),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(all_targets),
        "live_samples": int((np.array(all_targets) == 1).sum()),
        "spoof_samples": int((np.array(all_targets) == 0).sum()),
        "evaluation_duration_seconds": round(elapsed, 2),
        "metrics_iso_30107_3": {
            "test_accuracy": round(metrics["accuracy"], 4),
            "test_accuracy_pct": f"{metrics['accuracy'] * 100:.2f}%",
            "test_auc": round(metrics["auc"], 4),
            "test_apcer": round(metrics["apcer"], 4),
            "test_apcer_pct": f"{metrics['apcer'] * 100:.2f}%",
            "test_bpcer": round(metrics["bpcer"], 4),
            "test_bpcer_pct": f"{metrics['bpcer'] * 100:.2f}%",
            "test_hter": round(metrics["hter"], 4),
            "test_hter_pct": f"{metrics['hter'] * 100:.2f}%",
            "test_loss": round(metrics["loss"], 4)
        }
    }
    summary_path = ckpt_dir / "final_test_summary.json"
    with open(summary_path, mode="w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"\n[SUCCESS] Ringkasan metrik formal disimpan di: {summary_path}")
    print(f"[SUCCESS] Grafik Confusion Matrix & ROC Curve disimpan di: {chart_output}")
    print("Gunakan tabel dan gambar ini langsung untuk Bab 4 Skripsi Anda!")

if __name__ == "__main__":
    main()

